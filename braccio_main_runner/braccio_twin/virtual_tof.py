"""
virtual_tof.py — PyBullet raycast backend for the four wrist-mounted ToF.

For each of the 4 ToF sensors we look up the world pose of its URDF
mount link, shoot a single ray along its local +X axis, and write the
resulting distance into `ToFState.grids[ch]`. The grid is filled with
a constant value across the whole 8×8 cell pattern so the existing
downstream code (min/mean/max calculations, thresholding) behaves the
same as it does with real sensor data.

Sensor → channel mapping (wired to match CLAUDE.md's authority rules):
    CH0 → tof_front_link   (primary, sweep direction)
    CH1 → tof_back_link    (primary, opposite sweep)
    CH2 → tof_up_link      (advisory)
    CH3 → tof_down_link    (ignored / floor false-positives)
"""

from __future__ import annotations

import time

import numpy as np

from braccio_ctrl.tof_sensor import ToFState

from .sim_arm import SimArm


# (channel index, URDF link name)
TOF_CHANNELS: list[tuple[int, str]] = [
    (0, "tof_front_link"),
    (1, "tof_back_link"),
    (2, "tof_up_link"),
    (3, "tof_down_link"),
]


# VL53L5CX max range is ~4 m; we cap the sim to 2 m which is already
# well past anything the Braccio can collide with from its base.
DEFAULT_MAX_RANGE_M = 2.0


class VirtualToF:
    """Samples ray distances for the four wrist-mounted ToF sensors."""

    def __init__(
        self,
        sim_arm: SimArm,
        tof_state: ToFState,
        max_range_m: float = DEFAULT_MAX_RANGE_M,
    ):
        self._arm = sim_arm
        self._state = tof_state
        self._max_range_m = max_range_m

    def sample_once(self) -> None:
        """Ray-sample all 4 channels and update ToFState in one pass."""
        # Pre-compute all 4 ray origins + directions in world frame.
        results: list[tuple[int, float]] = []
        for ch, link_name in TOF_CHANNELS:
            try:
                pose = self._arm.get_link_pose(link_name)
            except KeyError:
                continue
            origin = pose.position
            dir_world = _rotate_vec_by_quat((1.0, 0.0, 0.0), pose.orientation)
            dist_m = self._arm.raycast(origin, dir_world, self._max_range_m)
            dist_mm = dist_m * 1000.0
            results.append((ch, dist_mm))

        # Commit into ToFState.
        now = time.time()
        with self._state._lock:
            for ch, dist_mm in results:
                grid = np.full((8, 8), dist_mm, dtype=np.float32)
                self._state.grids[ch] = grid
                self._state.active[ch] = 1
                self._state.last_rx[ch] = now
                self._state.frame_cnt[ch] += 1
                self._state.dirty[ch] = True
                self._state.diag_raw_min_mm[ch] = dist_mm
                self._state.diag_raw_max_mm[ch] = dist_mm
                self._state.diag_valid_cells[ch] = 64
                self._state.diag_zone_count[ch] = 64
                self._state.hz[ch] = 20  # nominal sim sample rate
            self._state.connected = True
            self._state.port = "pybullet://sim"
            self._state.mode = "MUX"

        # Let the obstacle decision logic react to the new frame.
        self._state.update_obstacle_status()


def _rotate_vec_by_quat(
    v: tuple[float, float, float],
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """
    Rotate vector v by quaternion q (xyzw convention, matching PyBullet).
    Pure Python — no external math deps.
    """
    x, y, z = v
    qx, qy, qz, qw = q
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    # v' = v + qw * t + cross(q_xyz, t)
    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)
    return (rx, ry, rz)
