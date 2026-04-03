"""
imu_state.py — Thread-safe IMU state container.

Holds orientation (roll/pitch/yaw), linear acceleration, and magnetometer
data streamed from the Teensy over the IMU serial line.

Expected Teensy serial line format:
  IMU,roll,pitch,yaw,ax,ay,az,mx,my,mz

Where:
  roll, pitch, yaw  — Euler angles in degrees
  ax, ay, az        — linear acceleration in m/s²
  mx, my, mz        — raw magnetometer values (uT or device units)

Calibration:
  Call record_calibration() at a known arm pose (e.g. theta=90) to store
  the yaw reference.  After calibration all yaw values are reported relative
  to the calibration pose, which removes hard-iron offset and aligns the
  coordinate frame to the arm's forward direction.
"""

import math
import threading
import time

import numpy as np


class IMUState:
    """Thread-safe shared state for the IMU (accelerometer + gyro + magnetometer)."""

    def __init__(self):
        self._lock = threading.RLock()

        # Euler angles (degrees)
        self.roll_deg:  float = 0.0
        self.pitch_deg: float = 0.0
        self.yaw_deg:   float = 0.0

        # Linear acceleration (m/s²)
        self.ax: float = 0.0
        self.ay: float = 0.0
        self.az: float = 0.0

        # Magnetometer (device units / uT)
        self.mx: float = 0.0
        self.my: float = 0.0
        self.mz: float = 0.0

        # Calibration
        self.yaw_calibration_offset: float = 0.0   # subtracted from raw yaw
        self.calibrated: bool = False

        self.last_rx: float = 0.0   # epoch time of last IMU packet
        self._rot_cache: np.ndarray | None = None
        self._rot_cache_key: tuple | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, roll: float, pitch: float, yaw: float,
               ax: float, ay: float, az: float,
               mx: float, my: float, mz: float) -> None:
        """Update all IMU fields atomically (called from serial reader thread)."""
        with self._lock:
            self.roll_deg  = roll
            self.pitch_deg = pitch
            self.yaw_deg   = yaw
            self.ax = ax;  self.ay = ay;  self.az = az
            self.mx = mx;  self.my = my;  self.mz = mz
            self.last_rx = time.time()
            self._rot_cache = None
            self._rot_cache_key = None

    def record_calibration(self) -> None:
        """
        Store the current yaw as the calibration reference.

        After calling this, yaw_relative() returns 0 at the current orientation
        and grows positive/negative as the arm rotates away from this pose.
        """
        with self._lock:
            self.yaw_calibration_offset = self.yaw_deg
            self.calibrated = True
            self._rot_cache = None
            self._rot_cache_key = None

    def yaw_relative(self) -> float:
        """
        Return yaw relative to the calibration pose (degrees).

        Wraps to [-180, +180].  Returns raw yaw if not yet calibrated.
        """
        with self._lock:
            if not self.calibrated:
                return self.yaw_deg
            delta = self.yaw_deg - self.yaw_calibration_offset
        # wrap to [-180, 180]
        return (delta + 180.0) % 360.0 - 180.0

    def rotation_matrix(self) -> np.ndarray:
        """
        Compute a 3×3 rotation matrix from roll/pitch/yaw (ZYX convention).

        Used by ObstacleMap to rotate sensor-frame points into the world frame.
        Only the yaw component is strictly necessary for the horizontal sweep
        (roll/pitch are small for a table-mounted arm), but all three are
        included for correctness.
        """
        with self._lock:
            roll = self.roll_deg
            pitch = self.pitch_deg
            # Use calibrated yaw for world-frame projection
            if self.calibrated:
                yaw = self.yaw_relative()
            else:
                yaw = self.yaw_deg
            key = (round(roll, 6), round(pitch, 6), round(yaw, 6), self.calibrated)
            if self._rot_cache is not None and self._rot_cache_key == key:
                return self._rot_cache.copy()

        r = math.radians(roll)
        p = math.radians(pitch)
        y = math.radians(yaw)

        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)

        # ZYX (yaw → pitch → roll)
        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [  -sp,            cp*sr,             cp*cr  ],
        ], dtype=np.float64)
        with self._lock:
            self._rot_cache = R
            self._rot_cache_key = key
        return R.copy()

    def snapshot(self) -> dict:
        """Return a copy of all display-relevant IMU state."""
        with self._lock:
            return {
                'roll_deg':   self.roll_deg,
                'pitch_deg':  self.pitch_deg,
                'yaw_deg':    self.yaw_deg,
                'yaw_relative': self.yaw_relative(),
                'ax': self.ax, 'ay': self.ay, 'az': self.az,
                'mx': self.mx, 'my': self.my, 'mz': self.mz,
                'calibrated': self.calibrated,
                'yaw_calibration_offset': self.yaw_calibration_offset,
                'last_rx': self.last_rx,
            }
