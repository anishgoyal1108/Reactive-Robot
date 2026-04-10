"""
obstacle_memory.py — Persistent obstacle voxel confidence memory.

Stores world-frame obstacle observations in a sparse voxel grid so recent
obstacles can be queried in O(1) average time. Confidence increases on
observations and decays over time; low-confidence cells are pruned.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import List

import numpy as np

from .constants import (
    OBS_MEM_CELL_MM,
    OBS_MEM_MAX_CELLS,
    OBS_MEM_INC,
    OBS_MEM_DECAY_PER_SEC,
    OBS_MEM_KEEP_THRESHOLD,
    OBS_MEM_OCCUPIED_THRESHOLD,
)


class PersistentObstacleMemory:
    """Sparse 3D voxel confidence memory in world coordinates."""

    def __init__(
        self,
        cell_mm: float = OBS_MEM_CELL_MM,
        max_cells: int = OBS_MEM_MAX_CELLS,
        conf_inc: float = OBS_MEM_INC,
        decay_per_sec: float = OBS_MEM_DECAY_PER_SEC,
        keep_threshold: float = OBS_MEM_KEEP_THRESHOLD,
        occupied_threshold: float = OBS_MEM_OCCUPIED_THRESHOLD,
    ):
        self._lock = threading.RLock()
        self._cell = float(cell_mm)
        self._max_cells = int(max_cells)
        self._inc = float(conf_inc)
        self._decay = float(decay_per_sec)
        self._keep = float(keep_threshold)
        self._occ = float(occupied_threshold)
        self._conf: dict[tuple[int, int, int], float] = {}

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            int(math.floor(x / self._cell)),
            int(math.floor(y / self._cell)),
            int(math.floor(z / self._cell)),
        )

    def ingest(self, points: np.ndarray) -> None:
        """Ingest world-frame points shape (N, 3)."""
        if points is None or len(points) == 0:
            return
        with self._lock:
            hits: defaultdict[tuple[int, int, int], int] = defaultdict(int)
            for p in points:
                k = self._key(float(p[0]), float(p[1]), float(p[2]))
                hits[k] += 1
            for k, n in hits.items():
                prev = self._conf.get(k, 0.0)
                self._conf[k] = min(1.0, prev + self._inc * n)
            self._prune_to_budget()

    def tick_decay(self, dt_s: float) -> None:
        """Decay confidence and prune stale cells."""
        if dt_s <= 0:
            return
        drop = self._decay * float(dt_s)
        with self._lock:
            dead = []
            for k, c in self._conf.items():
                c2 = c - drop
                if c2 < self._keep:
                    dead.append(k)
                else:
                    self._conf[k] = c2
            for k in dead:
                self._conf.pop(k, None)

    def query_radius(self, pos_xyz: tuple[float, float, float], r_mm: float) -> bool:
        """Return True if any occupied cell overlaps a sphere around pos."""
        x, y, z = pos_xyz
        rr = float(r_mm)
        kx, ky, kz = self._key(x, y, z)
        span = int(math.ceil(rr / self._cell)) + 1
        with self._lock:
            for ix in range(kx - span, kx + span + 1):
                for iy in range(ky - span, ky + span + 1):
                    for iz in range(kz - span, kz + span + 1):
                        c = self._conf.get((ix, iy, iz), 0.0)
                        if c < self._occ:
                            continue
                        cx = (ix + 0.5) * self._cell
                        cy = (iy + 0.5) * self._cell
                        cz = (iz + 0.5) * self._cell
                        if ((cx - x) ** 2 + (cy - y) ** 2 + (cz - z) ** 2) <= (rr**2):
                            return True
        return False

    def occupied_thetas(self, r_mm: float, z_mm: float,
                        r_tol: float = 100.0, z_tol: float = 80.0) -> List[float]:
        """
        Return theta values (degrees) of occupied voxels near the given (r, z) slice.
        Used by AutoSweeper to get forbidden theta bands from persistent memory.
        """
        thetas = []
        with self._lock:
            for key, conf in self._conf.items():
                if conf < self._occ:
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
                thetas.append(max(0.0, min(180.0, theta)))
        return thetas

    def snapshot_stats(self) -> dict:
        with self._lock:
            n = len(self._conf)
            mx = max(self._conf.values()) if n else 0.0
            occ = sum(1 for c in self._conf.values() if c >= self._occ)
            return {
                "cell_mm": self._cell,
                "num_cells": n,
                "num_occupied_cells": occ,
                "max_confidence": float(mx),
                "occupied_threshold": self._occ,
            }

    def clear(self) -> None:
        """Remove all voxels (call after IMU recalibration — world frame changed)."""
        with self._lock:
            self._conf.clear()

    def _prune_to_budget(self) -> None:
        if len(self._conf) <= self._max_cells:
            return
        # Drop lowest-confidence cells first.
        keep = sorted(self._conf.items(), key=lambda kv: kv[1], reverse=True)[
            : self._max_cells
        ]
        self._conf = dict(keep)
