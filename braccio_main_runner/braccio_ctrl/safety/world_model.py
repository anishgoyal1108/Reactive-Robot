"""
world_model.py — Timestamped obstacle point cloud + KDTree + coarse log-odds grid.

The world model is the single source of obstacle truth for the safety stack.
It ingests raw ToF frames, projects them into world frame, and exposes:

    cloud_points()              — current (N,3) array of obstacle points for
                                   vectorised capsule queries
    query_radius(xyz, r)        — KDTree point-radius search (O(log n))
    cell_state(xyz)             — 'free' | 'occ' | 'unknown' from the coarse grid
    polar_blocked_ranges(r, z)  — angular intervals blocked at a (r,z) slice;
                                   fed into the sweep branch's skip-ahead logic

Design (research §3):
  * Sparse point cloud stored as a single (N, 5) ndarray — columns (x, y, z, t,
    conf). Kept compact enough that `np.where(now - t > max_age)` decay is
    faster than any tree/heap trick.
  * ``scipy.spatial.KDTree`` rebuilt on a throttled cadence (default 0.5 s).
    Between rebuilds, ``cloud_points()`` still returns the raw array — that's
    what the capsule checker needs; it doesn't use the KDTree at all.
  * Coarse 20×20×20 log-odds occupancy grid at 30 mm resolution = 64 KB.
    Provides the 'unknown' distinction the research §3 identifies as critical
    for wrist-only sensing: cells the sensor has never observed stay 'unknown'
    and the planner biases paths away from them.

Thread-safety: every public method takes ``self._lock``. The ingest path is
called from the ToFBridge reader thread; readers (planner, BT tick) come from
the BT daemon; both can fire concurrently.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Literal, Optional

import numpy as np

try:
    from scipy.spatial import KDTree  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy>=1.11 is required for the safety stack (pip install scipy)."
    ) from exc


# ── Coarse grid log-odds constants ─────────────────────────────────────────
# Research §3 suggests a log-odds Bayes update. With these constants a single
# "free" observation moves -0.7 and a single "occupied" observation moves +0.9,
# so one hit outweighs one free sweep. Values clamped to [-4, +4] to stop the
# grid from locking into extreme beliefs that are hard to correct.
_LOGODDS_FREE = -0.7
_LOGODDS_OCC = 0.9
_LOGODDS_MIN = -4.0
_LOGODDS_MAX = 4.0
# Thresholds for cell_state():
#   logodds > OCC_THRESHOLD   → 'occ'
#   logodds < FREE_THRESHOLD  → 'free'
#   otherwise                 → 'unknown'
_OCC_THRESHOLD = 0.4
_FREE_THRESHOLD = -0.4


# ── WorldModel ─────────────────────────────────────────────────────────────

class WorldModel:
    """Timestamped obstacle point cloud + coarse log-odds occupancy grid.

    Parameters
    ----------
    grid_cells : int
        Edge count of the coarse grid (grid is cubic). Default 20 → 20³ = 8k cells.
    grid_cell_mm : float
        Side length of each coarse cell in millimetres. Default 30 mm gives
        a ±300 mm workspace centred on the arm base, covering the Braccio's
        reach (L1+L2+L3 = 310 mm) with one cell to spare.
    max_age_s : float
        Points older than this are purged on every ingest / decay call.
    rebuild_interval_s : float
        Minimum seconds between successive KDTree rebuilds. The capsule
        checker doesn't need the tree, so we let it lag without affecting
        collision-check fidelity.
    """

    def __init__(
        self,
        grid_cells: int = 20,
        grid_cell_mm: float = 30.0,
        max_age_s: float = 15.0,
        rebuild_interval_s: float = 0.5,
    ):
        self._lock = threading.RLock()
        # Point cloud — (N, 5) float64: x, y, z, timestamp, confidence.
        self._points = np.empty((0, 5), dtype=np.float64)
        self._max_age_s = float(max_age_s)
        self._rebuild_interval_s = float(rebuild_interval_s)
        self._last_rebuild_t = 0.0
        self._kdtree: Optional[KDTree] = None
        self._kdtree_points: Optional[np.ndarray] = None  # cached xyz view

        # Coarse grid — log-odds, shape (N, N, N). Index 0 maps to -N/2 * cell,
        # index N-1 maps to +(N/2-1) * cell. World origin (0,0,0) sits at the
        # centre of the cell at index (N/2, N/2, N/2).
        self._grid_n = int(grid_cells)
        self._grid_cell = float(grid_cell_mm)
        self._grid_half_mm = 0.5 * self._grid_n * self._grid_cell
        self._grid = np.zeros((self._grid_n,) * 3, dtype=np.float32)

    # ── Ingest path ────────────────────────────────────────────────────────

    def ingest_points(
        self,
        pts_world: np.ndarray,
        timestamp: Optional[float] = None,
        confidence: float = 1.0,
    ) -> None:
        """Append already-projected world-frame points.

        ``pts_world`` is an (N, 3) array of XYZ in millimetres.
        """
        if pts_world is None or len(pts_world) == 0:
            return
        pts = np.asarray(pts_world, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"expected (N,3); got {pts.shape}")
        now = float(time.time() if timestamp is None else timestamp)
        n = pts.shape[0]
        new_rows = np.empty((n, 5), dtype=np.float64)
        new_rows[:, :3] = pts
        new_rows[:, 3] = now
        new_rows[:, 4] = float(confidence)

        with self._lock:
            self._points = np.vstack([self._points, new_rows]) if self._points.size else new_rows
            self._bayes_update_occupied(pts)
            self._purge_stale_locked(now)

    def mark_observed_free(
        self,
        origin_world: np.ndarray,
        endpoint_world: np.ndarray,
        step_mm: Optional[float] = None,
    ) -> None:
        """Bayesian 'observed free' update along a ray.

        Ray from ``origin_world`` to ``endpoint_world`` in world-frame mm.
        Every coarse-grid cell traversed gets a log-odds decrement. The
        endpoint itself is NOT marked free — the caller should follow up with
        ``ingest_points(...)`` if the ray terminated on an obstacle.
        """
        o = np.asarray(origin_world, dtype=np.float64).reshape(3)
        e = np.asarray(endpoint_world, dtype=np.float64).reshape(3)
        delta = e - o
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            return
        step = float(step_mm if step_mm is not None else self._grid_cell)
        if step <= 0:
            step = self._grid_cell
        n = max(1, int(math.ceil(length / step)))
        # Sample evenly along the ray, excluding the endpoint.
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        samples = o + np.outer(ts, delta)
        with self._lock:
            for xyz in samples:
                idx = self._world_to_cell(xyz)
                if idx is None:
                    continue
                self._grid[idx] = np.clip(
                    self._grid[idx] + _LOGODDS_FREE, _LOGODDS_MIN, _LOGODDS_MAX
                )

    def decay(self, now: Optional[float] = None) -> int:
        """Drop points older than ``max_age_s``. Returns number dropped."""
        t = float(time.time() if now is None else now)
        with self._lock:
            before = self._points.shape[0]
            self._purge_stale_locked(t)
            return before - self._points.shape[0]

    def clear(self) -> None:
        """Reset the cloud, KDTree, and coarse grid — call on IMU recal."""
        with self._lock:
            self._points = np.empty((0, 5), dtype=np.float64)
            self._grid.fill(0.0)
            self._kdtree = None
            self._kdtree_points = None
            self._last_rebuild_t = 0.0

    # ── Read path ──────────────────────────────────────────────────────────

    def cloud_points(self) -> np.ndarray:
        """Return the current (N, 3) XYZ array (copy)."""
        with self._lock:
            if self._points.size == 0:
                return np.empty((0, 3), dtype=np.float64)
            return self._points[:, :3].copy()

    def point_count(self) -> int:
        with self._lock:
            return int(self._points.shape[0])

    def query_radius(self, point_xyz, radius_mm: float) -> np.ndarray:
        """O(log n) KDTree point-radius query. Returns matching XYZ points."""
        tree = self._ensure_tree()
        if tree is None or self._kdtree_points is None:
            return np.empty((0, 3), dtype=np.float64)
        hits = tree.query_ball_point(np.asarray(point_xyz, dtype=np.float64), float(radius_mm))
        if not hits:
            return np.empty((0, 3), dtype=np.float64)
        return self._kdtree_points[hits].copy()

    def query_nearest(
        self, point_xyz, k: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances_mm, points_xyz) of the k nearest obstacles.

        If the cloud has fewer than k points, arrays shorter than k are returned.
        """
        tree = self._ensure_tree()
        if tree is None or self._kdtree_points is None:
            return (np.empty((0,), dtype=np.float64), np.empty((0, 3), dtype=np.float64))
        dists, idxs = tree.query(np.asarray(point_xyz, dtype=np.float64), k=k)
        dists = np.atleast_1d(np.asarray(dists, dtype=np.float64))
        idxs = np.atleast_1d(np.asarray(idxs, dtype=int))
        valid = idxs < self._kdtree_points.shape[0]
        return dists[valid], self._kdtree_points[idxs[valid]].copy()

    def cell_state(
        self, xyz,
    ) -> Literal["free", "occ", "unknown"]:
        """Classification of the coarse-grid cell containing ``xyz``."""
        with self._lock:
            idx = self._world_to_cell(np.asarray(xyz, dtype=np.float64))
            if idx is None:
                return "unknown"
            v = float(self._grid[idx])
        if v > _OCC_THRESHOLD:
            return "occ"
        if v < _FREE_THRESHOLD:
            return "free"
        return "unknown"

    def polar_blocked_ranges(
        self,
        r_mm: float,
        z_mm: float,
        r_tol_mm: float = 80.0,
        z_tol_mm: float = 60.0,
        fov_half_deg: float = 22.5,
    ) -> list[tuple[float, float]]:
        """Angular intervals (degrees, in [0, 180]) blocked at the given (r,z).

        Each obstacle point within ``(r_tol, z_tol)`` of the sweep slice
        contributes a ``[theta - fov_half, theta + fov_half]`` band to the
        blocked set; overlapping bands are merged. The result feeds
        ``PolarObstacleMap.mark_blocked()`` in the sweep branch — research §7.
        """
        with self._lock:
            if self._points.shape[0] == 0:
                return []
            pts = self._points[:, :3]
            dx, dy, dz = pts[:, 0], pts[:, 1], pts[:, 2]
            pr = np.sqrt(dx * dx + dy * dy)
            mask = (np.abs(dz - z_mm) <= z_tol_mm) & (np.abs(pr - r_mm) <= r_tol_mm)
            if not np.any(mask):
                return []
            thetas = np.degrees(np.arctan2(dy[mask], dx[mask]))
            thetas = np.clip(thetas, 0.0, 180.0)
        # Sort and merge overlapping FOV bands.
        bands = sorted(
            (max(0.0, t - fov_half_deg), min(180.0, t + fov_half_deg))
            for t in thetas
        )
        merged: list[tuple[float, float]] = []
        for lo, hi in bands:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    def snapshot(self) -> dict:
        """Diagnostic summary for the display / session logger."""
        with self._lock:
            n = int(self._points.shape[0])
            oldest = float(time.time() - self._points[:, 3].min()) if n else 0.0
            grid_occ = int(np.sum(self._grid > _OCC_THRESHOLD))
            grid_free = int(np.sum(self._grid < _FREE_THRESHOLD))
        return {
            "num_points": n,
            "oldest_age_s": oldest,
            "grid_cells_occ": grid_occ,
            "grid_cells_free": grid_free,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _purge_stale_locked(self, now: float) -> None:
        if self._points.shape[0] == 0:
            return
        cutoff = now - self._max_age_s
        mask = self._points[:, 3] >= cutoff
        if not np.all(mask):
            self._points = self._points[mask]
            # Invalidate the KDTree — ``_ensure_tree`` will rebuild lazily.
            self._kdtree = None
            self._kdtree_points = None

    def _ensure_tree(self) -> Optional[KDTree]:
        now = time.time()
        with self._lock:
            needs_rebuild = (
                self._kdtree is None
                or (now - self._last_rebuild_t) > self._rebuild_interval_s
            )
            if not needs_rebuild:
                return self._kdtree
            if self._points.shape[0] == 0:
                self._kdtree = None
                self._kdtree_points = None
                self._last_rebuild_t = now
                return None
            xyz = self._points[:, :3].copy()
            self._kdtree = KDTree(xyz)
            self._kdtree_points = xyz
            self._last_rebuild_t = now
            return self._kdtree

    def _world_to_cell(self, xyz: np.ndarray) -> Optional[tuple[int, int, int]]:
        # xyz in mm → grid index, or None if out of bounds.
        ix = int(math.floor((float(xyz[0]) + self._grid_half_mm) / self._grid_cell))
        iy = int(math.floor((float(xyz[1]) + self._grid_half_mm) / self._grid_cell))
        iz = int(math.floor((float(xyz[2]) + self._grid_half_mm) / self._grid_cell))
        if 0 <= ix < self._grid_n and 0 <= iy < self._grid_n and 0 <= iz < self._grid_n:
            return (ix, iy, iz)
        return None

    def _bayes_update_occupied(self, pts_world: np.ndarray) -> None:
        # Caller holds ``self._lock``.
        for p in pts_world:
            idx = self._world_to_cell(p)
            if idx is None:
                continue
            self._grid[idx] = np.clip(
                self._grid[idx] + _LOGODDS_OCC, _LOGODDS_MIN, _LOGODDS_MAX
            )
