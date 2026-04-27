"""Teensy serial bridge for ToF + IR + MPU6050 IMU telemetry."""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

try:
    import serial

    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

from sensing.tof_frame import parse_tof_line

IR_LABELS = {0: "CLEAR", 1: "FAR", 2: "CLOSE", 3: "DANGER"}
IR_ACTIONS = {0: None, 1: "advisory", 2: "replan", 3: "back_away"}


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return v / n


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def _rot_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n <= 1e-12 or abs(angle_rad) <= 1e-12:
        return np.eye(3, dtype=float)
    u = np.asarray(axis, dtype=float).reshape(3) / n
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    k = _skew(u)
    return np.eye(3, dtype=float) + s * k + (1.0 - c) * (k @ k)


def _orthonormalize(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=float).reshape(3, 3))
    out = u @ vt
    if float(np.linalg.det(out)) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _align_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = _normalize_vec(np.asarray(src, dtype=float).reshape(3))
    b = _normalize_vec(np.asarray(dst, dtype=float).reshape(3))
    dot = float(np.dot(a, b))
    dot = max(-1.0, min(1.0, dot))
    if dot >= 1.0 - 1e-9:
        return np.eye(3, dtype=float)
    if dot <= -1.0 + 1e-9:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0], dtype=float))
        if float(np.linalg.norm(axis)) <= 1e-9:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0], dtype=float))
        return _rot_axis_angle(axis, np.pi)
    axis = np.cross(a, b)
    angle = float(np.arccos(dot))
    return _rot_axis_angle(axis, angle)


def _rot_to_euler_zyx_deg(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    sy = float(np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0]))
    singular = sy < 1e-6
    if not singular:
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
        pitch = float(np.arctan2(-r[2, 0], sy))
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
    else:
        yaw = float(np.arctan2(-r[0, 1], r[1, 1]))
        pitch = float(np.arctan2(-r[2, 0], sy))
        roll = 0.0
    return np.degrees(np.array([yaw, pitch, roll], dtype=float))


class ObstacleResponse:
    CLEAR = "clear"
    CAUTION = "caution"
    REPLAN = "replan"
    BACK_AWAY = "back_away"


class ToFState:
    def __init__(self, num_channels: int = 2):
        self._lock = threading.RLock()
        self.num_channels = num_channels

        self.grids = [np.full((8, 8), np.nan, dtype=np.float32) for _ in range(num_channels)]
        self.validity = [np.zeros((8, 8), dtype=np.uint8) for _ in range(num_channels)]
        self.active = [0] * num_channels
        self.last_rx = [0.0] * num_channels
        self.frame_cnt = [0] * num_channels
        self.dirty = [False] * num_channels
        self.hz = [0] * num_channels
        self.seq = [-1] * num_channels
        self.mcu_ms = [-1] * num_channels
        self.status = [0] * num_channels
        self.sensor_id = [f"ch{ch}" for ch in range(num_channels)]
        self.frame_imu = [
            {
                "online": False,
                "last_rx": 0.0,
                "seq": -1,
                "mcu_ms": -1,
                "ax_g": 0.0,
                "ay_g": 0.0,
                "az_g": 0.0,
                "gx_dps": 0.0,
                "gy_dps": 0.0,
                "gz_dps": 0.0,
                "temp_c": 0.0,
                "status": 0,
                "calibrated": False,
                "orientation_ready": False,
                "orientation_dt_s": 0.0,
                "rot_ref_from_imu": np.eye(3, dtype=float).tolist(),
                "euler_zyx_deg": [0.0, 0.0, 0.0],
            }
            for _ in range(num_channels)
        ]

        self.ir_bits = 0
        self.ir_label = "CLEAR"
        self.ir_action = ""
        self.ir_last_rx = 0.0

        # IMU stream (MPU6050 on direct I2C, independent of MUX channel IDs).
        self.imu_online = False
        self.imu_last_rx = 0.0
        self.imu_seq = -1
        self.imu_mcu_ms = -1
        self.imu_accel_g = np.zeros(3, dtype=float)
        self.imu_gyro_dps = np.zeros(3, dtype=float)
        self.imu_temp_c = 0.0
        self.imu_status = 0
        self.imu_samples = 0
        self.imu_calibrated = False
        self.imu_lpf_alpha = 0.25
        self.imu_rot_ref_from_imu = np.eye(3, dtype=float)
        self.imu_euler_deg = np.zeros(3, dtype=float)
        self.imu_orientation_ready = False
        self.imu_orientation_dt_s = 0.0
        self._imu_prev_wall_rx = 0.0
        self._imu_prev_mcu_ms = -1

        self.obstacle_response = ObstacleResponse.CLEAR
        self.obstacle_source = ""
        self.obstacle_dist_mm = -1.0
        self.tof_threshold_mm = 200.0
        self.tof_threshold_by_ch_mm = [200.0] * num_channels

        self.mode = "MUX"
        self.connected = False
        self.port = ""
        self.configured_channels = num_channels

        self._history_len = 800
        self.history_t = deque(maxlen=self._history_len)
        self.history_ch = deque(maxlen=self._history_len)
        self.history_min = deque(maxlen=self._history_len)
        self.history_avg = deque(maxlen=self._history_len)
        self.history_max = deque(maxlen=self._history_len)

    def set_imu_calibrated(self, calibrated: bool) -> None:
        with self._lock:
            self.imu_calibrated = bool(calibrated)

    def set_imu_lpf_alpha(self, alpha: float) -> None:
        a = float(alpha)
        if not np.isfinite(a):
            a = 0.25
        a = max(0.02, min(1.0, a))
        with self._lock:
            self.imu_lpf_alpha = a

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "grids": [g.copy() for g in self.grids],
                "validity": [v.copy() for v in self.validity],
                "active": list(self.active),
                "last_rx": list(self.last_rx),
                "frame_cnt": list(self.frame_cnt),
                "hz": list(self.hz),
                "seq": list(self.seq),
                "mcu_ms": list(self.mcu_ms),
                "status": list(self.status),
                "sensor_id": list(self.sensor_id),
                "frame_imu": [dict(x) for x in self.frame_imu],
                "ir_bits": self.ir_bits,
                "ir_label": self.ir_label,
                "ir_action": self.ir_action,
                "ir_last_rx": self.ir_last_rx,
                "imu": {
                    "online": self.imu_online,
                    "last_rx": self.imu_last_rx,
                    "seq": self.imu_seq,
                    "mcu_ms": self.imu_mcu_ms,
                    "ax_g": float(self.imu_accel_g[0]),
                    "ay_g": float(self.imu_accel_g[1]),
                    "az_g": float(self.imu_accel_g[2]),
                    "gx_dps": float(self.imu_gyro_dps[0]),
                    "gy_dps": float(self.imu_gyro_dps[1]),
                    "gz_dps": float(self.imu_gyro_dps[2]),
                    "temp_c": float(self.imu_temp_c),
                    "status": int(self.imu_status),
                    "samples": int(self.imu_samples),
                    "calibrated": bool(self.imu_calibrated),
                    "lpf_alpha": float(self.imu_lpf_alpha),
                    "orientation_ready": bool(self.imu_orientation_ready),
                    "orientation_dt_s": float(self.imu_orientation_dt_s),
                    "rot_ref_from_imu": self.imu_rot_ref_from_imu.tolist(),
                    "euler_zyx_deg": [float(x) for x in self.imu_euler_deg],
                },
                "obstacle_response": self.obstacle_response,
                "obstacle_source": self.obstacle_source,
                "obstacle_dist_mm": self.obstacle_dist_mm,
                "tof_threshold_mm": self.tof_threshold_mm,
                "tof_threshold_by_ch_mm": list(self.tof_threshold_by_ch_mm),
                "mode": self.mode,
                "connected": self.connected,
                "port": self.port,
                "num_channels": self.num_channels,
                "configured_channels": self.configured_channels,
            }

    def freshness_by_channel(self, max_age_s: float = 0.4) -> dict:
        with self._lock:
            now = time.time()
            return {ch: (now - self.last_rx[ch]) <= max_age_s for ch in range(self.num_channels)}

    def update_obstacle_status(self):
        with self._lock:
            now = time.time()
            if self.ir_bits in (2, 3):
                self.obstacle_response = ObstacleResponse.BACK_AWAY
                self.obstacle_source = "ir"
                self.obstacle_dist_mm = 0.0
                return

            closest = float("inf")
            closest_ch = -1

            for ch in range(self.num_channels):
                if self.active[ch] == 0:
                    continue
                if (now - float(self.last_rx[ch])) > 0.4:
                    continue
                g = self.grids[ch]
                v = self.validity[ch]
                if np.isnan(g).all():
                    continue
                g_valid = g[v.astype(bool)]
                g_valid = g_valid[np.isfinite(g_valid)]
                g_valid = g_valid[g_valid > 0.0]
                if g_valid.size <= 0:
                    continue
                ch_min = float(np.min(g_valid))
                ch_thresh = self.tof_threshold_mm
                if ch < len(self.tof_threshold_by_ch_mm):
                    ch_thresh = float(self.tof_threshold_by_ch_mm[ch])
                if ch_min >= ch_thresh:
                    continue
                if ch_min < closest:
                    closest = ch_min
                    closest_ch = ch

            if closest_ch >= 0:
                self.obstacle_response = ObstacleResponse.REPLAN
                self.obstacle_source = f"tof_ch{closest_ch}"
                self.obstacle_dist_mm = closest
                return

            self.obstacle_response = ObstacleResponse.CLEAR
            self.obstacle_source = ""
            self.obstacle_dist_mm = -1.0

    def set_channel_threshold(self, ch: int, threshold_mm: float) -> None:
        if ch < 0 or ch >= self.num_channels:
            return
        value = max(40.0, min(3000.0, float(threshold_mm)))
        with self._lock:
            self.tof_threshold_by_ch_mm[ch] = value

    def channel_threshold(self, ch: int) -> float:
        with self._lock:
            if 0 <= ch < len(self.tof_threshold_by_ch_mm):
                return float(self.tof_threshold_by_ch_mm[ch])
            return float(self.tof_threshold_mm)


class ToFBridge:
    def __init__(self, state: ToFState):
        self._state = state
        self._ser = None
        self._stop = threading.Event()
        self._write_lock = threading.Lock()
        self._reader = None

    def connect(self, port: str, baud: int = 115200) -> bool:
        if not _SERIAL_OK:
            return False
        try:
            self._ser = serial.Serial(port, baud, timeout=0.05)
            time.sleep(0.3)
            self._ser.reset_input_buffer()

            self._state.port = port
            self._state.connected = True

            self._stop.clear()
            self._reader = threading.Thread(target=self._reader_loop, daemon=True, name="tof-serial-reader")
            self._reader.start()
            return True
        except Exception:
            self._state.connected = False
            return False

    def close(self):
        self._stop.set()
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._state.connected = False

    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def send_mode(self, mode: str):
        mode = mode.strip().upper()
        if not self.is_open():
            return
        with self._write_lock:
            try:
                self._ser.write((mode + "\n").encode("utf-8"))
                self._ser.flush()
            except Exception:
                pass

    def set_active_channels(self, count: int) -> None:
        if not self.is_open():
            return
        c = max(1, min(int(count), self._state.num_channels))
        with self._state._lock:
            self._state.configured_channels = c
        with self._write_lock:
            try:
                self._ser.write((f"ACT {c}\n").encode("utf-8"))
                self._ser.flush()
            except Exception:
                pass

    def _reader_loop(self):
        s = self._state
        while not self._stop.is_set():
            try:
                if not self._ser or not self._ser.is_open:
                    time.sleep(0.05)
                    continue
                raw = self._ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                if raw.startswith("IR,"):
                    self._parse_ir(raw)
                    continue
                if raw.startswith("IMU,"):
                    self._parse_imu(raw)
                    continue
                if raw.startswith("MODE,"):
                    with s._lock:
                        s.mode = raw.split(",", 1)[1].strip()
                    continue
                if raw.startswith("CFG,ACT,"):
                    try:
                        n = int(raw.split(",")[-1])
                        with s._lock:
                            s.configured_channels = max(1, min(n, s.num_channels))
                    except Exception:
                        pass
                    continue

                fr = parse_tof_line(raw)
                if fr is not None:
                    self._apply_frame(fr)
            except Exception:
                if self._stop.is_set():
                    break
                time.sleep(0.01)

    def _apply_frame(self, frame):
        s = self._state
        ch = frame.mux_channel
        if ch < 0 or ch >= s.num_channels:
            return

        now = time.time()
        rows = max(1, int(getattr(frame, "rows", 8)))
        cols = max(1, int(getattr(frame, "cols", 8)))
        if (rows * cols) != len(frame.distances_mm) or (rows * cols) != len(frame.validity):
            return
        grid = np.array(frame.distances_mm, dtype=np.float32).reshape((rows, cols))
        val = np.array(frame.validity, dtype=np.uint8).reshape((rows, cols))

        with s._lock:
            s.grids[ch] = grid
            s.validity[ch] = val
            s.active[ch] = 1
            s.hz[ch] = int(frame.status) if frame.status >= 0 else s.hz[ch]
            s.last_rx[ch] = now
            s.frame_cnt[ch] += 1
            s.dirty[ch] = True
            s.seq[ch] = frame.seq
            s.mcu_ms[ch] = frame.mcu_ms
            s.status[ch] = frame.status
            s.sensor_id[ch] = frame.sensor_id
            s.frame_imu[ch] = {
                "online": bool(s.imu_online),
                "last_rx": float(s.imu_last_rx),
                "seq": int(s.imu_seq),
                "mcu_ms": int(s.imu_mcu_ms),
                "ax_g": float(s.imu_accel_g[0]),
                "ay_g": float(s.imu_accel_g[1]),
                "az_g": float(s.imu_accel_g[2]),
                "gx_dps": float(s.imu_gyro_dps[0]),
                "gy_dps": float(s.imu_gyro_dps[1]),
                "gz_dps": float(s.imu_gyro_dps[2]),
                "temp_c": float(s.imu_temp_c),
                "status": int(s.imu_status),
                "calibrated": bool(s.imu_calibrated),
                "orientation_ready": bool(s.imu_orientation_ready),
                "orientation_dt_s": float(s.imu_orientation_dt_s),
                "rot_ref_from_imu": s.imu_rot_ref_from_imu.tolist(),
                "euler_zyx_deg": [float(x) for x in s.imu_euler_deg],
            }

            g_valid = grid[val.astype(bool)]
            g_valid = g_valid[np.isfinite(g_valid)]
            g_valid = g_valid[g_valid > 0.0]
            if g_valid.size > 0:
                s.history_t.append(now)
                s.history_ch.append(ch)
                s.history_min.append(float(np.min(g_valid)))
                s.history_avg.append(float(np.mean(g_valid)))
                s.history_max.append(float(np.max(g_valid)))

        s.update_obstacle_status()

    def _parse_ir(self, line: str):
        s = self._state
        parts = line.split(",")
        if len(parts) < 2:
            return
        try:
            bits = int(parts[1]) & 0x03
            with s._lock:
                s.ir_bits = bits
                s.ir_label = IR_LABELS.get(bits, f"?{bits}")
                s.ir_action = IR_ACTIONS.get(bits, "")
                s.ir_last_rx = time.time()
            s.update_obstacle_status()
        except Exception:
            pass

    def _parse_imu(self, line: str) -> None:
        # IMU,<seq>,<mcu_ms>,<ax_g>,<ay_g>,<az_g>,<gx_dps>,<gy_dps>,<gz_dps>,<temp_c>,<status>
        parts = line.split(",")
        if len(parts) < 11:
            return
        try:
            seq = int(parts[1])
            mcu_ms = int(parts[2])
            ax = float(parts[3])
            ay = float(parts[4])
            az = float(parts[5])
            gx = float(parts[6])
            gy = float(parts[7])
            gz = float(parts[8])
            temp_c = float(parts[9])
            status = int(parts[10], 0)
        except Exception:
            return

        now = time.time()
        s = self._state
        with s._lock:
            alpha = float(s.imu_lpf_alpha)
            alpha = max(0.02, min(1.0, alpha))

            s.imu_online = True
            s.imu_last_rx = now
            s.imu_seq = seq
            s.imu_mcu_ms = mcu_ms

            if s.imu_samples <= 0:
                s.imu_accel_g[:] = [ax, ay, az]
                s.imu_gyro_dps[:] = [gx, gy, gz]
                s.imu_temp_c = temp_c
            else:
                s.imu_accel_g[:] = alpha * np.array([ax, ay, az], dtype=float) + (1.0 - alpha) * s.imu_accel_g
                s.imu_gyro_dps[:] = alpha * np.array([gx, gy, gz], dtype=float) + (1.0 - alpha) * s.imu_gyro_dps
                s.imu_temp_c = alpha * float(temp_c) + (1.0 - alpha) * float(s.imu_temp_c)

            s.imu_status = status
            s.imu_samples += 1
            self._update_imu_orientation_locked(now=now, mcu_ms=mcu_ms)

    def _update_imu_orientation_locked(self, now: float, mcu_ms: int) -> None:
        s = self._state
        dt_wall = 0.0 if s._imu_prev_wall_rx <= 0.0 else max(0.0, float(now - s._imu_prev_wall_rx))
        dt_mcu = 0.0
        if s._imu_prev_mcu_ms >= 0 and mcu_ms >= s._imu_prev_mcu_ms:
            dt_mcu = max(0.0, float(mcu_ms - s._imu_prev_mcu_ms) / 1000.0)

        dt = dt_mcu if 1e-4 <= dt_mcu <= 0.08 else dt_wall
        dt = max(0.0, min(0.08, dt))
        s._imu_prev_wall_rx = float(now)
        s._imu_prev_mcu_ms = int(mcu_ms)
        s.imu_orientation_dt_s = float(dt)

        a_body = _normalize_vec(np.asarray(s.imu_accel_g, dtype=float))
        if float(np.linalg.norm(a_body)) <= 1e-6:
            return

        gravity_ref = np.array([0.0, 0.0, -1.0], dtype=float)
        if not s.imu_orientation_ready:
            s.imu_rot_ref_from_imu = _align_vectors(a_body, gravity_ref)
            s.imu_rot_ref_from_imu = _orthonormalize(s.imu_rot_ref_from_imu)
            s.imu_euler_deg[:] = _rot_to_euler_zyx_deg(s.imu_rot_ref_from_imu)
            s.imu_orientation_ready = True
            return

        if dt > 1e-4:
            omega_rad_s = np.radians(np.asarray(s.imu_gyro_dps, dtype=float))
            angle = float(np.linalg.norm(omega_rad_s) * dt)
            if angle > 1e-9:
                s.imu_rot_ref_from_imu = s.imu_rot_ref_from_imu @ _rot_axis_angle(omega_rad_s, angle)

        g_est_ref = _normalize_vec(s.imu_rot_ref_from_imu @ a_body)
        corr_axis = np.cross(g_est_ref, gravity_ref)
        corr_axis_n = float(np.linalg.norm(corr_axis))
        if corr_axis_n > 1e-9:
            corr_dot = float(np.dot(g_est_ref, gravity_ref))
            corr_dot = max(-1.0, min(1.0, corr_dot))
            corr_angle = float(np.arctan2(corr_axis_n, corr_dot))
            corr_gain = 0.12 if dt <= 1e-6 else min(0.35, max(0.05, 3.2 * dt))
            corr_angle = min(np.deg2rad(18.0), corr_angle * corr_gain)
            s.imu_rot_ref_from_imu = _rot_axis_angle(corr_axis, corr_angle) @ s.imu_rot_ref_from_imu

        s.imu_rot_ref_from_imu = _orthonormalize(s.imu_rot_ref_from_imu)
        s.imu_euler_deg[:] = _rot_to_euler_zyx_deg(s.imu_rot_ref_from_imu)





