"""
obstacle_map.py — World-frame obstacle point cloud with Kalman tracking.

Responsibilities
----------------
1. Project each VL53L5CX ToF grid cell (4×4 or 8×8) into the inertial (world) frame,
   using the arm's current polar pose (theta, r, z) and IMU orientation.
2. Maintain a rolling cloud of obstacle points (age-capped, per-channel).
3. Run a 6-DOF constant-velocity Kalman filter on the obstacle centroid so
   that the estimated position is preserved for up to OBS_MAP_MAX_AGE_S
   seconds after the obstacle leaves the sensor FOV.
4. Expose get_obstacle_thetas() for the AutoSweeper to identify forbidden
   theta values at a given (r, z) slice.

Sensor orientation conventions (configurable via constants.py):
  CH0 = front  — primary authority, faces the sweep direction
  CH1 = back   — primary authority, opposite side
  CH2 = top    — confirmation only
  CH3 = bottom — confirmation only

Coordinate system (origin = arm base / shoulder pivot):
  +X  = arm forward at theta = 0
  +Y  = arm left at theta = 90
  +Z  = upward
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .constants import (
    OBS_MAP_MAX_AGE_S,
    OBS_MAP_TOF_FOV_DEG,
    SENSOR_MOUNT_FRONT_OFFSET,
    SENSOR_MOUNT_BACK_OFFSET,
    SENSOR_MOUNT_TOP_OFFSET,
    SENSOR_MOUNT_BOTTOM_OFFSET,
    SENSOR_REPLAN_CHANNELS,
    SENSOR_IGNORE_CHANNELS,
    TOF_THRESHOLDS_MM,
    SWEEP_OBSTACLE_MARGIN_DEG,
    L1, L2, L3,
)
from .ik_solver import polar_to_cartesian
from .obstacle_memory import PersistentObstacleMemory


# ── Sensor mount geometry ─────────────────────────────────────────────────────

def _Rx(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])

def _Ry(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[math.cos(r),0,math.sin(r)],[0,1,0],[-math.sin(r),0,math.cos(r)]])

def _Rz(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]])


_SENSOR_CONFIGS = {
    0: (np.array(SENSOR_MOUNT_FRONT_OFFSET,  dtype=np.float64), _Rz(  0.0)),
    1: (np.array(SENSOR_MOUNT_BACK_OFFSET,   dtype=np.float64), _Rz(180.0)),
    2: (np.array(SENSOR_MOUNT_TOP_OFFSET,    dtype=np.float64), _Ry(-90.0)),
    3: (np.array(SENSOR_MOUNT_BOTTOM_OFFSET, dtype=np.float64), _Ry( 90.0)),
}


# ── Kalman filter (6-state: [x,y,z,vx,vy,vz]) ────────────────────────────────

class KalmanTracker:
    def __init__(self):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e6
        self.initialized = False
        self._Q_pos  = 1.0
        self._Q_vel  = 50.0
        self._R_meas = 400.0

    def predict(self, dt: float) -> np.ndarray:
        if not self.initialized:
            return np.zeros(3)
        F = np.eye(6)
        F[0, 3] = dt;  F[1, 4] = dt;  F[2, 5] = dt
        Q = np.diag([self._Q_pos, self._Q_pos, self._Q_pos,
                     self._Q_vel, self._Q_vel, self._Q_vel])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:3].copy()

    def update(self, meas: np.ndarray) -> None:
        if not self.initialized:
            self.x[:3] = meas
            self.x[3:] = 0.0
            self.P = np.eye(6) * 1000.0
            self.initialized = True
            return
        H = np.zeros((3, 6));  H[0,0] = 1;  H[1,1] = 1;  H[2,2] = 1
        R = np.eye(3) * self._R_meas
        y  = meas - H @ self.x
        S  = H @ self.P @ H.T + R
        K  = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def position_uncertainty_mm(self) -> float:
        return float(np.sqrt(np.sum(np.diag(self.P)[:3])))


# ── Obstacle cloud dataclass ──────────────────────────────────────────────────

@dataclass
class ObstacleCloud:
    points:     np.ndarray
    centroid:   np.ndarray
    timestamp:  float
    source_ch:  int


# ── ObstacleMap ───────────────────────────────────────────────────────────────

class ObstacleMap:
    """
    Maintains a rolling world-frame obstacle point cloud and a Kalman tracker.

    Thread-safe: all public methods acquire self._lock.
    """

    def __init__(self, thresholds_mm: list = None,
                 max_age_s: float = OBS_MAP_MAX_AGE_S):
        self._lock        = threading.RLock()
        self._clouds: List[ObstacleCloud] = []
        self._tracker     = KalmanTracker()
        self._thresholds  = list(thresholds_mm or TOF_THRESHOLDS_MM)
        self._max_age_s   = max_age_s
        self._last_obs_t  = 0.0
        self._tracker_last_update = 0.0
        self._memory      = PersistentObstacleMemory()
        self._memory_last_t = time.time()

    # ── Main update call ──────────────────────────────────────────────────────

    def update(self, tof_grids: list, arm_snap: dict,
               imu_snap: dict, imu_R: Optional[np.ndarray] = None,
               tof_thresholds_mm: Optional[list] = None) -> None:
        """
        Project all active ToF grids into world frame and refresh the cloud.

        Parameters
        ----------
        tof_grids         : list of 4 np.ndarray (4×4 or 8×8, mm)
        arm_snap          : dict from ArmState.snapshot()
        imu_snap          : dict from IMUState.snapshot()
        imu_R             : optional pre-computed 3×3 rotation matrix
        tof_thresholds_mm : optional per-channel threshold override
        """
        theta = arm_snap.get('theta', 90.0)
        r     = arm_snap.get('r',     152.0)
        z_arm = arm_snap.get('z',     -50.0)

        x_ee, y_ee = polar_to_cartesian(theta, r)
        ee_pos = np.array([x_ee, y_ee, z_arm], dtype=np.float64)
        R_base = _Rz(theta)

        if imu_R is not None:
            R_imu = imu_R
        elif imu_snap.get('calibrated', False):
            R_imu = _Rz(imu_snap.get('yaw_relative', 0.0))
        else:
            R_imu = np.eye(3)

        now = time.time()
        new_clouds: List[ObstacleCloud] = []

        thresholds = tof_thresholds_mm if tof_thresholds_mm else self._thresholds
        for ch, grid in enumerate(tof_grids):
            if ch in SENSOR_IGNORE_CHANNELS:   # CH3 (bottom): skip entirely
                continue
            if grid is None or np.isnan(grid).all():
                continue

            mount_offset, R_sensor_to_ee = _SENSOR_CONFIGS[ch]
            ch_thresh = thresholds[ch] if ch < len(thresholds) else thresholds[0]
            pts_world = self._project_grid(
                grid, mount_offset, R_sensor_to_ee, ee_pos, R_base, R_imu,
                ch_thresh
            )

            if pts_world is not None and len(pts_world) > 0:
                centroid = pts_world.mean(axis=0)
                new_clouds.append(ObstacleCloud(
                    points=pts_world,
                    centroid=centroid,
                    timestamp=now,
                    source_ch=ch,
                ))

        with self._lock:
            # Decay + ingest into persistent memory
            dt_mem = now - self._memory_last_t
            if dt_mem > 0:
                self._memory.tick_decay(dt_mem)
                self._memory_last_t = now
            for c in new_clouds:
                self._memory.ingest(c.points)

            # Discard stale rolling cloud
            cutoff = now - self._max_age_s
            self._clouds = [c for c in self._clouds if c.timestamp > cutoff]
            self._clouds.extend(new_clouds)

            # Feed Kalman filter (prefer primary REPLAN channels)
            primary_clouds = [c for c in new_clouds
                              if c.source_ch in SENSOR_REPLAN_CHANNELS]
            all_new = primary_clouds if primary_clouds else new_clouds

            if all_new:
                merged_centroid = np.mean([c.centroid for c in all_new], axis=0)
                dt = now - self._tracker_last_update if self._tracker_last_update else 0.0
                if dt > 0:
                    self._tracker.predict(dt)
                self._tracker.update(merged_centroid)
                self._tracker_last_update = now
                self._last_obs_t = now
            elif self._tracker.initialized:
                dt = now - self._tracker_last_update
                if dt > 0 and dt < self._max_age_s:
                    self._tracker.predict(dt)
                    self._tracker_last_update = now

    # ── Query interface ───────────────────────────────────────────────────────

    def get_obstacle_thetas(self, sweep_r_mm: float,
                            sweep_z_mm: float) -> List[float]:
        """Return theta values (degrees) of obstacles near the given (r, z) slice."""
        z_tol = 80.0
        r_tol = 100.0
        occupied_thetas = []
        with self._lock:
            for cloud in self._clouds:
                for pt in cloud.points:
                    px, py, pz = pt
                    if abs(pz - sweep_z_mm) > z_tol:
                        continue
                    pt_r = math.sqrt(px**2 + py**2)
                    if abs(pt_r - sweep_r_mm) > r_tol:
                        continue
                    pt_theta = math.degrees(math.atan2(py, px))
                    occupied_thetas.append(max(0.0, min(180.0, pt_theta)))
        return occupied_thetas

    def estimate_lost_obstacle(self) -> Optional[np.ndarray]:
        """Return Kalman-estimated position of obstacle that left sensor FOV."""
        with self._lock:
            if not self._tracker.initialized:
                return None
            age = time.time() - self._last_obs_t
            if age < 0.1:
                return None
            if age > self._max_age_s:
                return None
            if self._tracker.position_uncertainty_mm > 200.0:
                return None
            return self._tracker.position.copy()

    def has_active_obstacle(self) -> bool:
        with self._lock:
            now = time.time()
            return any(c.timestamp > now - self._max_age_s for c in self._clouds)

    def all_points(self) -> np.ndarray:
        """Return (N, 3) array of all current world-frame obstacle points."""
        with self._lock:
            if not self._clouds:
                return np.empty((0, 3), dtype=np.float64)
            now = time.time()
            pts = [c.points for c in self._clouds
                   if c.timestamp > now - self._max_age_s]
            if not pts:
                return np.empty((0, 3), dtype=np.float64)
            return np.vstack(pts)

    # ── Persistent memory interface ───────────────────────────────────────────

    def memory_occupied_near(self, pos_xyz: tuple, r_mm: float) -> bool:
        """Fast O(k) occupancy check from persistent voxel memory."""
        return self._memory.query_radius(pos_xyz, r_mm)

    def memory_occupied_thetas(self, r_mm: float, z_mm: float) -> List[float]:
        """Return obstacle thetas from persistent memory at given (r, z) slice."""
        return self._memory.occupied_thetas(r_mm, z_mm)

    def clear_memory(self) -> None:
        """Clear persistent memory (call after IMU recalibration)."""
        self._memory.clear()

    def snapshot(self) -> dict:
        """Thread-safe snapshot for display."""
        with self._lock:
            pts = self.all_points()
            centroid = pts.mean(axis=0).tolist() if len(pts) > 0 else None
            est = (self._tracker.position.tolist()
                   if self._tracker.initialized else None)
            unc = (self._tracker.position_uncertainty_mm
                   if self._tracker.initialized else -1.0)
            return {
                'num_points':            len(pts),
                'centroid':              centroid,
                'kalman_estimate':       est,
                'kalman_uncertainty_mm': unc,
                'has_active':            self.has_active_obstacle(),
                'last_obs_age_s':        time.time() - self._last_obs_t,
                'memory':                self._memory.snapshot_stats(),
            }

    # ── Internal projection ───────────────────────────────────────────────────

    def _project_grid(self, grid: np.ndarray,
                      mount_offset: np.ndarray,
                      R_s2ee: np.ndarray,
                      ee_pos: np.ndarray,
                      R_base: np.ndarray,
                      R_imu: np.ndarray,
                      threshold_mm: float) -> Optional[np.ndarray]:
        """Project one ToF grid (any resolution) into world-frame XYZ points."""
        rows, cols = grid.shape
        half_fov = math.radians(OBS_MAP_TOF_FOV_DEG / 2.0)

        world_pts = []
        for row in range(rows):
            for col in range(cols):
                d = float(grid[row, col])
                if math.isnan(d) or d <= 0 or d >= threshold_mm:
                    continue

                h_ang = ((col - (cols - 1) / 2.0) / ((cols - 1) / 2.0)) * half_fov
                v_ang = ((row - (rows - 1) / 2.0) / ((rows - 1) / 2.0)) * half_fov

                ps = np.array([
                    d * math.cos(v_ang) * math.cos(h_ang),
                    d * math.cos(v_ang) * math.sin(h_ang),
                    d * math.sin(v_ang),
                ], dtype=np.float64)

                p_ee = R_s2ee @ ps + mount_offset
                p_world = R_imu @ (R_base @ p_ee + ee_pos)
                world_pts.append(p_world)

        if not world_pts:
            return None
        return np.array(world_pts, dtype=np.float64)
