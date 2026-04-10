"""
tof_sensor.py — Teensy serial bridge for VL53L5CX ToF sensors + IR (OUT1D).

Connects to a separate Teensy (or the same Teensy on a different port)
that runs the VL5 ToF firmware.  A daemon reader thread drains serial
frames and updates shared state that the controller and plotter can read.

Protocol (from Teensy):
  FRAME,ch,activeFlag,hz,res,d0,d1,...,d63    — 8*8 ToF distance grid
  IR,bits                                      — 2-bit IR proximity value
  MODE,{MUX|CH0|CH1|CH2|CH3}                  — current mode echo

IR 2-bit encoding (OUT1D port):
  0b00 = no detection          → clear
  0b01 = far detection         → advisory
  0b10 = close detection       → warning
  0b11 = very close / danger   → BACK AWAY (emergency)

Obstacle decision logic:
  ToF  detects obstacle within threshold → REPLAN trajectory
  IR   detects obstacle (ToF missed it)  → BACK AWAY immediately
"""

from __future__ import annotations

import os
import threading
import time
import queue
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from .imu_state import IMUState
from .serial_bridge import ensure_serial_device_path
from .constants import (
    TOF_THRESHOLDS_MM,
    SENSOR_REPLAN_CHANNELS,
    SENSOR_ADVISORY_CHANNELS,
    SENSOR_IGNORE_CHANNELS,
)

if TYPE_CHECKING:
    import serial  # pyright: ignore[reportMissingModuleSource]

_SERIAL_OK = False
_serial_mod: Any = None
try:
    import serial as _serial_mod  # pyright: ignore[reportMissingModuleSource]

    _SERIAL_OK = True
except ImportError:
    pass


# ── IR state decode ──────────────────────────────────────────────────────────

IR_LABELS = {
    0: "CLEAR",
    1: "FAR",
    2: "CLOSE",
    3: "DANGER",
}

IR_ACTIONS = {
    0: "",  # no action needed
    1: "advisory",  # log only
    2: "replan",  # same as ToF — replan trajectory
    3: "back_away",  # emergency retreat
}


# ── Obstacle response enum ───────────────────────────────────────────────────


class ObstacleResponse:
    CLEAR = "clear"
    REPLAN = "replan"  # ToF detected — replan trajectory
    BACK_AWAY = "back_away"  # IR detected (ToF missed) — retreat


class ToFState:
    """
    Thread-safe shared state for all ToF sensors + IR.

    Up to 4 ToF channels (CH0–CH3), each an 8×8 distance grid in mm.
    Plus IR 2-bit proximity value.
    """

    def __init__(self, num_channels: int = 4) -> None:
        self._lock = threading.RLock()
        self.num_channels = num_channels

        # Per-channel 8×8 grids (mm), NaN = no data yet
        self.grids = [
            np.full((8, 8), np.nan, dtype=np.float32) for _ in range(num_channels)
        ]
        self.active = [0] * num_channels
        self.last_rx = [0.0] * num_channels
        self.frame_cnt = [0] * num_channels
        self.dirty = [False] * num_channels
        self.hz = [0] * num_channels

        # IR state
        self.ir_bits: int = 0  # raw 2-bit value
        self.ir_label: str = "CLEAR"
        self.ir_action: str = ""  # '', 'advisory', 'replan', 'back_away'
        self.ir_last_rx: float = 0.0

        # Obstacle detection
        self.obstacle_response: str = ObstacleResponse.CLEAR
        self.obstacle_source: str = ""  # 'tof_chN' or 'ir'
        self.obstacle_dist_mm: float = -1.0  # closest measured distance
        self.tof_threshold_mm: float = TOF_THRESHOLDS_MM[0]  # legacy fallback
        self.tof_thresholds_mm: list = list(TOF_THRESHOLDS_MM)

        # Mode
        self.mode: str = "MUX"

        # Connection
        self.connected: bool = False
        self.port: str = ""

        # History for CSV logging (rolling deque, last N frames per channel)
        self._history_len = 600  # ~10 s at 15 Hz × 4 ch
        self.history_t: deque = deque(maxlen=self._history_len)
        self.history_ch: deque = deque(maxlen=self._history_len)
        self.history_min: deque = deque(maxlen=self._history_len)
        self.history_avg: deque = deque(maxlen=self._history_len)
        self.history_max: deque = deque(maxlen=self._history_len)

        # Per-channel parse diagnostics (last good frame; helps tell “no serial”
        # from “serial OK but every cell masked invalid by firmware rules”)
        self.diag_raw_min_mm: list = [float("nan")] * num_channels
        self.diag_raw_max_mm: list = [float("nan")] * num_channels
        self.diag_valid_cells: list = [0] * num_channels
        self.diag_zone_count: list = [0] * num_channels

    def snapshot(self) -> dict:
        """Return a copy of all display-relevant ToF/IR state."""
        with self._lock:
            return {
                "grids": [g.copy() for g in self.grids],
                "active": list(self.active),
                "last_rx": list(self.last_rx),
                "frame_cnt": list(self.frame_cnt),
                "hz": list(self.hz),
                "ir_bits": self.ir_bits,
                "ir_label": self.ir_label,
                "ir_action": self.ir_action,
                "ir_last_rx": self.ir_last_rx,
                "obstacle_response": self.obstacle_response,
                "obstacle_source": self.obstacle_source,
                "obstacle_dist_mm": self.obstacle_dist_mm,
                "tof_threshold_mm": self.tof_threshold_mm,
                "tof_thresholds_mm": list(self.tof_thresholds_mm),
                "mode": self.mode,
                "connected": self.connected,
                "port": self.port,
                "num_channels": self.num_channels,
                "diag_raw_min_mm": list(self.diag_raw_min_mm),
                "diag_raw_max_mm": list(self.diag_raw_max_mm),
                "diag_valid_cells": list(self.diag_valid_cells),
                "diag_zone_count": list(self.diag_zone_count),
            }

    def update_obstacle_status(self):
        """
        Recompute combined obstacle response from ToF + IR.

        Channel authority (defined in constants.py):
          SENSOR_REPLAN_CHANNELS   (CH0, CH1 — sides): trigger REPLAN
          SENSOR_ADVISORY_CHANNELS (CH2 — top):         advisory only, no REPLAN
          SENSOR_IGNORE_CHANNELS   (CH3 — bottom):      completely skipped

        Priority: IR DANGER > IR CLOSE > primary ToF REPLAN > IR FAR > CLEAR.
        """
        with self._lock:
            # --- IR check (second line of defense — if it fires, ToF missed) ---
            if self.ir_bits == 3:
                self.obstacle_response = ObstacleResponse.BACK_AWAY
                self.obstacle_source = "ir"
                self.obstacle_dist_mm = 0.0
                return
            if self.ir_bits == 2:
                self.obstacle_response = ObstacleResponse.BACK_AWAY
                self.obstacle_source = "ir"
                self.obstacle_dist_mm = 0.0
                return

            # --- Primary ToF check (SENSOR_REPLAN_CHANNELS only) ---
            closest = float("inf")
            closest_ch = -1
            closest_thr = -1.0
            for ch in range(self.num_channels):
                if ch in SENSOR_IGNORE_CHANNELS:   # CH3 (bottom): skip
                    continue
                if self.active[ch] == 0:
                    continue
                g = self.grids[ch]
                if np.isnan(g).all():
                    continue
                ch_min = float(np.nanmin(g))
                thr = (
                    self.tof_thresholds_mm[ch]
                    if ch < len(self.tof_thresholds_mm)
                    else self.tof_threshold_mm
                )
                if ch in SENSOR_ADVISORY_CHANNELS:
                    # Advisory channels never trigger replans
                    continue
                if ch not in SENSOR_REPLAN_CHANNELS:
                    continue
                if ch_min < thr and ch_min < closest:
                    closest = ch_min
                    closest_ch = ch
                    closest_thr = thr

            if closest_ch >= 0:
                self.obstacle_response = ObstacleResponse.REPLAN
                self.obstacle_source = f"tof_ch{closest_ch}"
                self.obstacle_dist_mm = closest
                self.tof_threshold_mm = closest_thr
                return

            # --- IR advisory (far detection, not critical) ---
            if self.ir_bits == 1:
                self.obstacle_response = ObstacleResponse.CLEAR
                self.obstacle_source = "ir_advisory"
                self.obstacle_dist_mm = -1.0
                return

            # --- All clear ---
            self.obstacle_response = ObstacleResponse.CLEAR
            self.obstacle_source = ""
            self.obstacle_dist_mm = -1.0

    @property
    def primary_threshold_mm(self) -> float:
        """Minimum threshold across primary (REPLAN) channels, for display."""
        return min(self.tof_thresholds_mm[ch] for ch in SENSOR_REPLAN_CHANNELS)


class ToFBridge:
    """
    Serial bridge to the Teensy running VL5 ToF + IR firmware.

    A daemon reader thread drains serial data and updates ToFState and IMUState.
    Supports sending mode commands (MUX, CH0–CH3).
    """

    def __init__(self, state: ToFState, imu_state: IMUState | None = None) -> None:
        self._state = state
        self._imu_state = imu_state if imu_state is not None else IMUState()
        self._ser: serial.Serial | None = None
        self._stop = threading.Event()
        self._write_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._last_open_error: str = ""

    # ── Connection ────────────────────────────────────────────────────────

    def connect(self, port: str, baud: int = 115200) -> bool:
        """Open serial port and start reader thread."""
        self._last_open_error = ""
        if not _SERIAL_OK:
            return False
        try:
            ensure_serial_device_path(port, "Teensy serial port (--teensy-port)")
        except ValueError as exc:
            self._last_open_error = str(exc)
            return False
        try:
            assert _serial_mod is not None
            self._ser = _serial_mod.Serial(port, baud, timeout=0.05)
            time.sleep(0.3)
            self._ser.reset_input_buffer()

            self._state.port = port
            self._state.connected = True

            self._stop.clear()
            self._reader = threading.Thread(
                target=self._reader_loop,
                daemon=True,
                name="tof-serial-reader",
            )
            self._reader.start()

            # Enable all 4 channels (firmware defaults to 2)
            self.send_mode("ACT 4")

            return True
        except Exception as e:
            self._state.connected = False
            self._last_open_error = str(e)
            return False

    def close(self) -> None:
        """Stop reader and close serial."""
        self._stop.set()
        ser = self._ser
        if ser is not None and ser.is_open:
            try:
                ser.close()
            except Exception:
                pass
        self._state.connected = False

    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ── Commands ──────────────────────────────────────────────────────────

    def send_mode(self, mode: str) -> None:
        """Send a mode command: MUX, CH0, CH1, CH2, CH3."""
        mode = mode.strip().upper()
        ser = self._ser
        if ser is None or not ser.is_open:
            return
        with self._write_lock:
            try:
                ser.write((mode + "\n").encode("utf-8"))
                ser.flush()
            except Exception:
                pass

    # ── Reader thread ─────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Daemon: read lines, parse FRAME/IR/MODE, update state."""
        s = self._state
        while not self._stop.is_set():
            try:
                ser_l = self._ser
                if ser_l is None or not ser_l.is_open:
                    time.sleep(0.05)
                    continue
                if ser_l.in_waiting == 0:
                    time.sleep(0.002)
                    continue

                raw = ser_l.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                # ── TF frame (vl5_tca_4x4 firmware) ──────────────────────
                if raw.startswith("TF,"):
                    self._parse_tf(raw)
                    continue

                # ── FRAME line (legacy firmware) ──────────────────────────
                if raw.startswith("FRAME,"):
                    self._parse_frame(raw)
                    continue

                # ── IR line ───────────────────────────────────────────────
                if raw.startswith("IR,"):
                    self._parse_ir(raw)
                    continue

                # ── MODE line ─────────────────────────────────────────────
                if raw.startswith("MODE,"):
                    with s._lock:
                        s.mode = raw.split(",", 1)[1].strip()
                    continue

                # ── CFG lines (grid size / active count / rate) ───────────
                if raw.startswith("CFG,"):
                    self._parse_cfg(raw)
                    continue

                # ── IMU line ───────────────────────────────────────────────
                if raw.startswith("IMU,"):
                    self._parse_imu(raw)
                    continue

            except Exception:
                if self._stop.is_set():
                    break
                time.sleep(0.01)

    def _parse_tf(self, line: str):
        """
        Parse the TF frame format from vl5_tca_4x4 firmware:
          TF,<seq>,<mcu_ms>,S<ch>,<ch>,<joint_id>,<status>,<rows>,<cols>,<d0..dN>,<v0..vN>

        The sensor_id field is 'S<ch>' (e.g. 'S0'), mux_ch is the integer
        channel (0-3), rows/cols define the grid (4×4 = 16 zones for this
        firmware), followed by N distance values then N validity flags.
        """
        s = self._state
        parts = line.split(",")
        # Minimum: TF, seq, mcu_ms, sensor_id, mux_ch, joint_id, status, rows, cols, d0
        if len(parts) < 10:
            return
        try:
            # parts[1] = seq, parts[2] = mcu_ms, parts[3] = sensor_id (S0..S3)
            # parts[4] = mux_ch (int), parts[5] = joint_id (str), parts[6] = status
            # parts[7] = rows, parts[8] = cols
            ch = int(parts[4])
            rows = int(parts[7])
            cols = int(parts[8])
            zones = rows * cols

            if ch < 0 or ch >= s.num_channels:
                return
            if zones < 1:
                return

            # Distance values start at parts[9]; validity flags follow after
            data_start = 9
            data_end = data_start + zones
            if len(parts) < data_end:
                return

            dist = np.array(parts[data_start:data_end], dtype=np.float32)
            raw_min = float(np.min(dist))
            raw_max = float(np.max(dist))

            # Apply validity mask if flags are present
            if len(parts) >= data_end + zones:
                valid = np.array(parts[data_end : data_end + zones], dtype=np.int8)
                dist[valid == 0] = np.nan

            now = time.time()
            grid = dist.reshape((rows, cols))
            n_kept = int(np.sum(~np.isnan(grid)))

            with s._lock:
                s.grids[ch] = grid
                s.active[ch] = 1
                s.last_rx[ch] = now
                s.frame_cnt[ch] += 1
                s.dirty[ch] = True
                s.diag_raw_min_mm[ch] = raw_min
                s.diag_raw_max_mm[ch] = raw_max
                s.diag_valid_cells[ch] = n_kept
                s.diag_zone_count[ch] = zones

                g_valid = grid[~np.isnan(grid)]
                if g_valid.size > 0:
                    s.history_t.append(now)
                    s.history_ch.append(ch)
                    s.history_min.append(float(np.nanmin(g_valid)))
                    s.history_avg.append(float(np.mean(g_valid)))
                    s.history_max.append(float(np.max(g_valid)))

            s.update_obstacle_status()

        except Exception:
            pass

    def _parse_cfg(self, line: str):
        """
        Parse CFG lines sent by the firmware on startup:
          CFG,ACT,<count>          — number of active channels
          CFG,GRID,<rows>,<cols>   — sensor grid dimensions
          CFG,TARGET_HZ,<hz>       — target frame rate (informational)
        """
        parts = line.split(",")
        if len(parts) < 3:
            return
        kind = parts[1].strip().upper()
        if kind == "ACT":
            pass  # informational; active count is managed by ACT 4 command
        # GRID and TARGET_HZ are informational only; no state change needed

    def _parse_frame(self, line: str):
        """Parse legacy: FRAME,ch,activeFlag,hz,res,d0,...,dN"""
        s = self._state
        parts = line.split(",")
        if len(parts) < 6:
            return
        try:
            ch = int(parts[1])
            active = int(parts[2])
            hz = int(parts[3])
            res = int(parts[4])

            if ch < 0 or ch >= s.num_channels:
                return
            if res not in (16, 64):
                return

            side = 4 if res == 16 else 8
            data = np.array(parts[5 : 5 + res], dtype=np.float32)
            if data.size != res:
                return

            now = time.time()
            grid = data.reshape((side, side))
            raw_min = float(np.min(data))
            raw_max = float(np.max(data))
            n_kept = int(np.sum(~np.isnan(grid)))

            with s._lock:
                s.grids[ch] = grid
                s.active[ch] = 1 if active else 0
                s.hz[ch] = hz
                s.last_rx[ch] = now
                s.frame_cnt[ch] += 1
                s.dirty[ch] = True
                s.diag_raw_min_mm[ch] = raw_min
                s.diag_raw_max_mm[ch] = raw_max
                s.diag_valid_cells[ch] = n_kept
                s.diag_zone_count[ch] = res

                g_valid = grid[~np.isnan(grid)]
                if g_valid.size > 0:
                    s.history_t.append(now)
                    s.history_ch.append(ch)
                    s.history_min.append(float(np.nanmin(g_valid)))
                    s.history_avg.append(float(np.mean(g_valid)))
                    s.history_max.append(float(np.max(g_valid)))

            s.update_obstacle_status()

        except Exception:
            pass

    def _parse_ir(self, line: str):
        """Parse: IR,bits  where bits is 0–3 (2-bit value)."""
        s = self._state
        parts = line.split(",")
        if len(parts) < 2:
            return
        try:
            bits = int(parts[1]) & 0x03  # mask to 2 bits
            with s._lock:
                s.ir_bits = bits
                s.ir_label = IR_LABELS.get(bits, f"?{bits}")
                s.ir_action = IR_ACTIONS.get(bits, "")
                s.ir_last_rx = time.time()

            s.update_obstacle_status()
        except Exception:
            pass

    def _parse_imu(self, line: str):
        """
        Parse IMU lines from the vl5_tca_4x4 firmware:
          IMU,<seq>,<mcu_ms>,<ax_g>,<ay_g>,<az_g>,<gx_dps>,<gy_dps>,<gz_dps>,<temp_c>,<status>

        Derives roll and pitch from the accelerometer; yaw is not available
        from an accelerometer-only MPU-6050 without magnetometer fusion.
        """
        parts = line.split(",")
        if len(parts) < 10:
            return
        try:
            # parts[1]=seq, parts[2]=mcu_ms
            ax_g = float(parts[3])
            ay_g = float(parts[4])
            az_g = float(parts[5])
            gx_dps = float(parts[6])
            gy_dps = float(parts[7])
            gz_dps = float(parts[8])
            # parts[9]=temp_c, parts[10]=status

            # Derive roll/pitch from gravity vector (radians → degrees)
            import math

            roll = math.degrees(math.atan2(ay_g, az_g))
            pitch = math.degrees(
                math.atan2(-ax_g, math.sqrt(ay_g * ay_g + az_g * az_g))
            )
            yaw = 0.0  # not available without magnetometer fusion

            self._imu_state.update(roll, pitch, yaw, ax_g, ay_g, az_g, 0.0, 0.0, 0.0)
        except Exception:
            pass
