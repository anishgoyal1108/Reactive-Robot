"""
tof_sensor.py — Teensy serial bridge for VL53L5CX ToF sensors + IR (OUT1D).

Connects to a separate Teensy (or the same Teensy on a different port)
that runs the VL5 ToF firmware.  A daemon reader thread drains serial
frames and updates shared state that the controller and plotter can read.

Protocol (from Teensy):
  FRAME,ch,activeFlag,hz,res,d0,d1,...,d63    — 8×8 ToF distance grid
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

import os
import threading
import time
import queue
from collections import deque

import numpy as np

try:
    import serial
    from serial.tools import list_ports as _list_ports
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False


# ── IR state decode ──────────────────────────────────────────────────────────

IR_LABELS = {
    0: 'CLEAR',
    1: 'FAR',
    2: 'CLOSE',
    3: 'DANGER',
}

IR_ACTIONS = {
    0: None,              # no action needed
    1: 'advisory',        # log only
    2: 'replan',          # same as ToF — replan trajectory
    3: 'back_away',       # emergency retreat
}


# ── Obstacle response enum ───────────────────────────────────────────────────

class ObstacleResponse:
    CLEAR       = 'clear'
    REPLAN      = 'replan'        # ToF detected — replan trajectory
    BACK_AWAY   = 'back_away'     # IR detected (ToF missed) — retreat


class ToFState:
    """
    Thread-safe shared state for all ToF sensors + IR.

    Up to 4 ToF channels (CH0–CH3), each an 8×8 distance grid in mm.
    Plus IR 2-bit proximity value.
    """

    def __init__(self, num_channels: int = 4):
        self._lock = threading.RLock()
        self.num_channels = num_channels

        # Per-channel 8×8 grids (mm), NaN = no data yet
        self.grids = [
            np.full((8, 8), np.nan, dtype=np.float32)
            for _ in range(num_channels)
        ]
        self.active    = [0] * num_channels
        self.last_rx   = [0.0] * num_channels
        self.frame_cnt = [0] * num_channels
        self.dirty     = [False] * num_channels
        self.hz        = [0] * num_channels

        # IR state
        self.ir_bits:   int   = 0       # raw 2-bit value
        self.ir_label:  str   = 'CLEAR'
        self.ir_action: str   = ''      # '', 'advisory', 'replan', 'back_away'
        self.ir_last_rx: float = 0.0

        # Obstacle detection
        self.obstacle_response: str  = ObstacleResponse.CLEAR
        self.obstacle_source:   str  = ''      # 'tof_chN' or 'ir'
        self.obstacle_dist_mm:  float = -1.0   # closest measured distance
        self.tof_threshold_mm:  float = 300.0  # configurable

        # Mode
        self.mode: str = 'MUX'

        # Connection
        self.connected: bool  = False
        self.port:      str   = ''

        # History for CSV logging (rolling deque, last N frames per channel)
        self._history_len = 600   # ~10 s at 15 Hz × 4 ch
        self.history_t:   deque = deque(maxlen=self._history_len)
        self.history_ch:  deque = deque(maxlen=self._history_len)
        self.history_min: deque = deque(maxlen=self._history_len)
        self.history_avg: deque = deque(maxlen=self._history_len)
        self.history_max: deque = deque(maxlen=self._history_len)

    def snapshot(self) -> dict:
        """Return a copy of all display-relevant ToF/IR state."""
        with self._lock:
            return {
                'grids':        [g.copy() for g in self.grids],
                'active':       list(self.active),
                'last_rx':      list(self.last_rx),
                'frame_cnt':    list(self.frame_cnt),
                'hz':           list(self.hz),
                'ir_bits':      self.ir_bits,
                'ir_label':     self.ir_label,
                'ir_action':    self.ir_action,
                'ir_last_rx':   self.ir_last_rx,
                'obstacle_response': self.obstacle_response,
                'obstacle_source':   self.obstacle_source,
                'obstacle_dist_mm':  self.obstacle_dist_mm,
                'tof_threshold_mm':  self.tof_threshold_mm,
                'mode':         self.mode,
                'connected':    self.connected,
                'port':         self.port,
                'num_channels': self.num_channels,
            }

    def update_obstacle_status(self):
        """
        Recompute combined obstacle response from ToF + IR.

        Priority: IR DANGER > ToF threshold > IR CLOSE > IR FAR > CLEAR.
        """
        with self._lock:
            # --- IR check (second line of defense — if it fires, ToF missed) ---
            if self.ir_bits == 3:
                self.obstacle_response = ObstacleResponse.BACK_AWAY
                self.obstacle_source   = 'ir'
                self.obstacle_dist_mm  = 0.0  # unknown exact, but very close
                return
            if self.ir_bits == 2:
                self.obstacle_response = ObstacleResponse.BACK_AWAY
                self.obstacle_source   = 'ir'
                self.obstacle_dist_mm  = 0.0
                return

            # --- ToF check (primary detection — replan trajectory) ---
            closest     = float('inf')
            closest_ch  = -1
            for ch in range(self.num_channels):
                if self.active[ch] == 0:
                    continue
                g = self.grids[ch]
                if np.isnan(g).all():
                    continue
                ch_min = float(np.nanmin(g))
                if ch_min < closest:
                    closest    = ch_min
                    closest_ch = ch

            if closest < self.tof_threshold_mm and closest_ch >= 0:
                self.obstacle_response = ObstacleResponse.REPLAN
                self.obstacle_source   = f'tof_ch{closest_ch}'
                self.obstacle_dist_mm  = closest
                return

            # --- IR advisory (far detection, not critical) ---
            if self.ir_bits == 1:
                self.obstacle_response = ObstacleResponse.CLEAR
                self.obstacle_source   = 'ir_advisory'
                self.obstacle_dist_mm  = -1.0
                return

            # --- All clear ---
            self.obstacle_response = ObstacleResponse.CLEAR
            self.obstacle_source   = ''
            self.obstacle_dist_mm  = -1.0


class ToFBridge:
    """
    Serial bridge to the Teensy running VL5 ToF + IR firmware.

    A daemon reader thread drains serial data and updates ToFState.
    Supports sending mode commands (MUX, CH0–CH3).
    """

    def __init__(self, state: ToFState):
        self._state     = state
        self._ser       = None
        self._stop      = threading.Event()
        self._write_lock = threading.Lock()
        self._reader     = None

    # ── Connection ────────────────────────────────────────────────────────

    def connect(self, port: str, baud: int = 115200) -> bool:
        """Open serial port and start reader thread."""
        if not _SERIAL_OK:
            return False
        try:
            self._ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(0.3)
            self._ser.reset_input_buffer()

            self._state.port      = port
            self._state.connected = True

            self._stop.clear()
            self._reader = threading.Thread(
                target=self._reader_loop,
                daemon=True,
                name='tof-serial-reader',
            )
            self._reader.start()
            return True
        except Exception as e:
            self._state.connected = False
            return False

    def close(self):
        """Stop reader and close serial."""
        self._stop.set()
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._state.connected = False

    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ── Commands ──────────────────────────────────────────────────────────

    def send_mode(self, mode: str):
        """Send a mode command: MUX, CH0, CH1, CH2, CH3."""
        mode = mode.strip().upper()
        if not self.is_open():
            return
        with self._write_lock:
            try:
                self._ser.write((mode + '\n').encode('utf-8'))
                self._ser.flush()
            except Exception:
                pass

    # ── Reader thread ─────────────────────────────────────────────────────

    def _reader_loop(self):
        """Daemon: read lines, parse FRAME/IR/MODE, update state."""
        s = self._state
        while not self._stop.is_set():
            try:
                if not self._ser or not self._ser.is_open:
                    time.sleep(0.05)
                    continue
                if self._ser.in_waiting == 0:
                    time.sleep(0.002)
                    continue

                raw = self._ser.readline().decode('utf-8', errors='ignore').strip()
                if not raw:
                    continue

                # ── FRAME line ────────────────────────────────────────────
                if raw.startswith('FRAME,'):
                    self._parse_frame(raw)
                    continue

                # ── IR line ───────────────────────────────────────────────
                if raw.startswith('IR,'):
                    self._parse_ir(raw)
                    continue

                # ── MODE line ─────────────────────────────────────────────
                if raw.startswith('MODE,'):
                    with s._lock:
                        s.mode = raw.split(',', 1)[1].strip()
                    continue

            except Exception:
                if self._stop.is_set():
                    break
                time.sleep(0.01)

    def _parse_frame(self, line: str):
        """Parse: FRAME,ch,activeFlag,hz,res,d0,...,d63"""
        s = self._state
        parts = line.split(',')
        if len(parts) < 6:
            return
        try:
            ch     = int(parts[1])
            active = int(parts[2])
            hz     = int(parts[3])
            res    = int(parts[4])

            if ch < 0 or ch >= s.num_channels:
                return
            if res != 64:
                return

            data = np.array(parts[5:5 + 64], dtype=np.float32)
            if data.size != 64:
                return

            now = time.time()
            grid = data.reshape((8, 8))

            with s._lock:
                s.grids[ch]    = grid
                s.active[ch]   = 1 if active else 0
                s.hz[ch]       = hz
                s.last_rx[ch]  = now
                s.frame_cnt[ch] += 1
                s.dirty[ch]    = True

                # Append to history
                g_valid = grid[~np.isnan(grid)]
                if g_valid.size > 0:
                    s.history_t.append(now)
                    s.history_ch.append(ch)
                    s.history_min.append(float(np.min(g_valid)))
                    s.history_avg.append(float(np.mean(g_valid)))
                    s.history_max.append(float(np.max(g_valid)))

            # Update obstacle status after every frame
            s.update_obstacle_status()

        except Exception:
            pass

    def _parse_ir(self, line: str):
        """Parse: IR,bits  where bits is 0–3 (2-bit value)."""
        s = self._state
        parts = line.split(',')
        if len(parts) < 2:
            return
        try:
            bits = int(parts[1]) & 0x03  # mask to 2 bits
            with s._lock:
                s.ir_bits    = bits
                s.ir_label   = IR_LABELS.get(bits, f'?{bits}')
                s.ir_action  = IR_ACTIONS.get(bits, '')
                s.ir_last_rx = time.time()

            s.update_obstacle_status()
        except Exception:
            pass
