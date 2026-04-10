"""
obstacle_memory.py — Persistent spatial obstacle memory using a sparse voxel grid.

Remembers obstacle locations in the IMU-calibrated world frame so the arm
doesn't need to recompute positions every frame.  Provides O(1) occupancy
lookups for fast pre-command collision checks.

Design
------
- Sparse dict mapping (ix, iy, iz) integer voxel keys → confidence [0, 1].
- `mark_occupied(pts)`: each world-frame point confirms the voxel it falls in,
  boosting confidence by OBS_MEMORY_CONFIRM_RATE (capped at 1.0).
- `decay(dt)`: every tick, all cells lose confidence proportional to elapsed
  time.  Cells below OBS_MEMORY_PRUNE_BELOW are removed (auto-cleanup).
- `is_occupied(x, y, z)` / `is_region_occupied(x, y, z, radius)`: O(1) and
  O(k) lookups for collision checking (k = small constant for the sphere).

Thread-safety: all public methods acquire self._lock.
"""

import math
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from .constants import (
    OBS_MEMORY_CELL_MM,
    OBS_MEMORY_CONFIRM_RATE,
    OBS_MEMORY_DECAY_RATE,
    OBS_MEMORY_CONFIDENCE_MIN,
    OBS_MEMORY_CONFIDENCE_MAX,
    OBS_MEMORY_PRUNE_BELOW,
    OBS_MEMORY_OCCUPIED_THRESH,
)


class PersistentObstacleMemory:
    """
    Sparse 3D voxel grid for persistent obstacle memory.

    Each voxel stores a confidence value [0, 1].  Observations boost
    confidence; time decay reduces it.  Cells that drop below the prune
    threshold are automatically removed.
    """

    def __init__(self, cell_mm: float = OBS_MEMORY_CELL_MM):
        self._lock = threading.RLock()
        self._cell = cell_mm
        self._inv_cell = 1.0 / cell_mm
        # Sparse grid: (ix, iy, iz) → confidence float
        self._grid: dict = {}
        self._last_decay_t: float = time.monotonic()

    # ── Voxel key helpers ────────────────────────────────────────────────────

    def _to_key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert world-frame mm coordinates to integer voxel key."""
        return (
            int(math.floor(x * self._inv_cell)),
            int(math.floor(y * self._inv_cell)),
            int(math.floor(z * self._inv_cell)),
        )

    def _key_center(self, key: Tuple[int, int, int]) -> np.ndarray:
        """Return the world-frame center of a voxel cell."""
        return np.array([
            (key[0] + 0.5) * self._cell,
            (key[1] + 0.5) * self._cell,
            (key[2] + 0.5) * self._cell,
        ], dtype=np.float64)

    # ── Observation ──────────────────────────────────────────────────────────

    def mark_occupied(self, pts: np.ndarray) -> int:
        """
        Confirm obstacle presence at world-frame points.

        Parameters
        ----------
        pts : (N, 3) array of world-frame XYZ coordinates (mm)

        Returns
        -------
        Number of distinct voxels updated.
        """
        if pts is None or len(pts) == 0:
            return 0
        updated = set()
        with self._lock:
            for pt in pts:
                key = self._to_key(float(pt[0]), float(pt[1]), float(pt[2]))
                old = self._grid.get(key, 0.0)
                new = min(OBS_MEMORY_CONFIDENCE_MAX,
                          old + OBS_MEMORY_CONFIRM_RATE)
                self._grid[key] = new
                updated.add(key)
        return len(updated)

    # ── Decay + prune ────────────────────────────────────────────────────────

    def decay(self, dt: float = None) -> int:
        """
        Reduce confidence of all cells by decay_rate * dt.
        Prunes cells that fall below the threshold.

        Parameters
        ----------
        dt : elapsed seconds since last decay call.
             If None, computed automatically from monotonic clock.

        Returns
        -------
        Number of cells pruned.
        """
        now = time.monotonic()
        with self._lock:
            if dt is None:
                dt = now - self._last_decay_t
            self._last_decay_t = now
            if dt <= 0:
                return 0

            loss = OBS_MEMORY_DECAY_RATE * dt
            pruned = 0
            to_delete = []
            for key, conf in self._grid.items():
                new_conf = max(OBS_MEMORY_CONFIDENCE_MIN, conf - loss)
                if new_conf < OBS_MEMORY_PRUNE_BELOW:
                    to_delete.append(key)
                    pruned += 1
                else:
                    self._grid[key] = new_conf
            for key in to_delete:
                del self._grid[key]
            return pruned

    # ── Occupancy queries (fast-path for collision checks) ───────────────────

    def is_occupied(self, x: float, y: float, z: float) -> bool:
        """O(1) check: is the voxel at (x, y, z) occupied?"""
        key = self._to_key(x, y, z)
        with self._lock:
            return self._grid.get(key, 0.0) >= OBS_MEMORY_OCCUPIED_THRESH

    def is_region_occupied(self, x: float, y: float, z: float,
                           radius: float) -> bool:
        """
        O(k) check: is any voxel within `radius` mm of (x, y, z) occupied?

        Checks a cube of voxels inscribing the sphere.  k is typically
        very small (3×3×3 = 27 for radius < 2 * cell_mm).
        """
        n = int(math.ceil(radius * self._inv_cell))
        cx, cy, cz = self._to_key(x, y, z)
        r_sq = radius * radius
        with self._lock:
            for ix in range(cx - n, cx + n + 1):
                for iy in range(cy - n, cy + n + 1):
                    for iz in range(cz - n, cz + n + 1):
                        if self._grid.get((ix, iy, iz), 0.0) < OBS_MEMORY_OCCUPIED_THRESH:
                            continue
                        # Check actual distance from query point to cell center
                        cell_x = (ix + 0.5) * self._cell
                        cell_y = (iy + 0.5) * self._cell
                        cell_z = (iz + 0.5) * self._cell
                        dx = cell_x - x
                        dy = cell_y - y
                        dz = cell_z - z
                        if dx*dx + dy*dy + dz*dz <= r_sq:
                            return True
        return False

    # ── Bulk queries ─────────────────────────────────────────────────────────

    def occupied_centroids(self) -> np.ndarray:
        """
        Return (N, 3) array of occupied cell centers (world-frame mm).

        Only cells at or above OBS_MEMORY_OCCUPIED_THRESH are included.
        """
        with self._lock:
            centers = [
                self._key_center(k)
                for k, conf in self._grid.items()
                if conf >= OBS_MEMORY_OCCUPIED_THRESH
            ]
        if not centers:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(centers, dtype=np.float64)

    def occupied_thetas(self, r_mm: float, z_mm: float,
                        r_tol: float = 100.0, z_tol: float = 80.0) -> List[float]:
        """
        Return theta values (degrees) of occupied voxels near the given (r, z) slice.

        Faster than ObstacleMap.get_obstacle_thetas() because it iterates
        only over occupied voxel centers, not raw point clouds.
        """
        thetas = []
        with self._lock:
            for key, conf in self._grid.items():
                if conf < OBS_MEMORY_OCCUPIED_THRESH:
                    continue
                cx = (key[0] + 0.5) * self._cell
                cy = (key[1] + 0.5) * self._cell
                cz = (key[2] + 0.5) * self._cell
                if abs(cz - z_mm) > z_tol:
                    continue
                cr = math.sqrt(cx * cx + cy * cy)
                if abs(cr - r_mm) > r_tol:
                    continue
                theta = math.degrees(math.atan2(cy, cx))
                theta = max(0.0, min(180.0, theta))
                thetas.append(theta)
        return thetas

    # ── Stats for display ────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Thread-safe stats snapshot for display."""
        with self._lock:
            total = len(self._grid)
            occupied = sum(1 for c in self._grid.values()
                          if c >= OBS_MEMORY_OCCUPIED_THRESH)
            max_conf = max(self._grid.values()) if self._grid else 0.0
            return {
                'total_cells': total,
                'occupied_cells': occupied,
                'max_confidence': round(max_conf, 2),
            }

    def clear(self) -> None:
        """Remove all voxels (e.g. after recalibration)."""
        with self._lock:
            self._grid.clear()
