"""
collision.py — Full-arm capsule-vs-point-cloud collision checker.

Research §4 in one page:

    Each arm link is modelled as a capsule (line segment + radius). The distance
    from an obstacle point to a capsule reduces to "distance from point to the
    nearest point on the centerline, minus the capsule radius". That's a closed-
    form expression over the segment parameter t ∈ [0, 1], vectorisable across
    the entire obstacle cloud in a single NumPy call. Checking one full
    configuration against a 1000-point cloud takes well under a millisecond on
    commodity hardware, so BiRRT can expand thousands of configurations inside
    a 2-second planner budget.

Interfaces exposed:

    capsule_point_cloud_distance(p0, p1, r, cloud)
        Vectorised capsule-to-cloud distance. Returns an (N,) array of signed
        distances (negative = inside the capsule). Negative-value count is
        equivalent to "point in collision with this capsule".

    capsule_capsule_distance(c0, c1)
        Minimum distance between two capsules (closed-form, see
        http://geomalgorithms.com/a07-_distance.html).

    CollisionChecker(world, fk, clearance_mm)
        ``config_in_collision(q)`` — full 5-capsule + self-pair check at one
        joint configuration. ``path_in_collision(q0, q1, n_samples)`` —
        interpolates joint-space between configurations and checks each. This
        matches the Arduino's linear joint interpolation in ``Braccio.move()``,
        eliminating the polar-vs-joint-space gap the legacy MotionGuard had.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .fk import Capsule, ForwardKinematics, capsule_pair_is_adjacent
from .world_model import WorldModel


# ── Vectorised capsule primitives ──────────────────────────────────────────

def _point_segment_distance_squared(
    points: np.ndarray, p0: np.ndarray, p1: np.ndarray,
) -> np.ndarray:
    """Squared distance from each of N points to the line segment p0→p1.

    Uses the parametric form:  closest(t) = p0 + t·(p1-p0),  t clamped to [0,1].
    Vectorised: all N points share one segment.
    """
    seg = p1 - p0
    seg_len_sq = float(seg @ seg)
    if seg_len_sq < 1e-12:
        # Degenerate capsule — fall back to point-to-point distance.
        diff = points - p0
        return np.einsum("ij,ij->i", diff, diff)
    # t = dot(p - p0, seg) / |seg|²
    diffs = points - p0
    t = np.clip(diffs @ seg / seg_len_sq, 0.0, 1.0)
    # Closest points on the segment, shape (N, 3).
    closest = p0 + np.outer(t, seg)
    delta = points - closest
    return np.einsum("ij,ij->i", delta, delta)


def capsule_point_cloud_distance(
    p0, p1, radius_mm: float, cloud: np.ndarray,
) -> np.ndarray:
    """Signed distance from each cloud point to the capsule.

    ``cloud`` is expected as an (N, 3) float array in the same frame as the
    capsule endpoints. Returns an (N,) float array; values < 0 are points that
    lie inside the capsule (collision).

    Empty clouds return an empty array — the caller should special-case N == 0.
    """
    p0a = np.asarray(p0, dtype=np.float64)
    p1a = np.asarray(p1, dtype=np.float64)
    pts = np.asarray(cloud, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"cloud must be (N, 3); got {pts.shape}")
    if pts.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    d2 = _point_segment_distance_squared(pts, p0a, p1a)
    return np.sqrt(d2) - float(radius_mm)


def capsule_capsule_distance(c0: Capsule, c1: Capsule) -> float:
    """Minimum gap (mm) between two capsules — negative when interpenetrating.

    Implementation follows the classic "closest points between two 3D
    segments" formulation (Eberly / geomalgorithms). Numerical corner cases:
    parallel segments, coincident endpoints, and zero-length segments are all
    handled by the ``denom`` / ``d`` guards below.
    """
    p0 = np.asarray(c0.p0, dtype=np.float64)
    p1 = np.asarray(c0.p1, dtype=np.float64)
    q0 = np.asarray(c1.p0, dtype=np.float64)
    q1 = np.asarray(c1.p1, dtype=np.float64)
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(u @ u)
    b = float(u @ v)
    c = float(v @ v)
    d = float(u @ w)
    e = float(v @ w)
    denom = a * c - b * b
    # Compute sc (for segment u) and tc (for segment v)
    if denom < 1e-12:
        sc = 0.0
        tc = e / c if c > 1e-12 else 0.0
    else:
        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom
    sc = max(0.0, min(1.0, sc))
    tc = max(0.0, min(1.0, tc))
    # Second pass: re-clamp sc given the clamped tc, and vice versa, so the
    # result is correct even when the unconstrained optimum lies outside
    # both [0, 1] ranges.
    if c > 1e-12:
        tc2 = (b * sc + e) / c
        tc = max(0.0, min(1.0, tc2))
    if a > 1e-12:
        sc2 = (b * tc - d) / a
        sc = max(0.0, min(1.0, sc2))
    closest_u = p0 + sc * u
    closest_v = q0 + tc * v
    gap = float(np.linalg.norm(closest_u - closest_v))
    return gap - (c0.radius_mm + c1.radius_mm)


# ── Collision checker ──────────────────────────────────────────────────────

class CollisionChecker:
    """Orchestrates capsule checks for a full configuration or interpolated path.

    Parameters
    ----------
    world : WorldModel
        Source of the obstacle point cloud. The checker always reads the
        *current* cloud at the moment of each query — no caching, because
        planner calls already complete well under the cloud's update rate.
    fk : ForwardKinematics
        Link-endpoint provider.
    clearance_mm : float
        Extra safety margin added to every capsule radius before the check.
        Absorbs commanded-vs-actual servo drift, FK rounding, and sensor noise.
    self_collision : bool
        Enable capsule-capsule self-collision checks between non-adjacent
        capsule pairs. Cheap (5 capsules × ≤5 pairs = ≤10 calls).
    """

    def __init__(
        self,
        world: WorldModel,
        fk: ForwardKinematics,
        clearance_mm: float = 30.0,
        self_collision: bool = True,
    ):
        self._world = world
        self._fk = fk
        self._clearance_mm = float(clearance_mm)
        self._self_collision = bool(self_collision)

    # ── single-config check ────────────────────────────────────────────────

    def config_in_collision(self, q: list[int]) -> bool:
        """True if the full-arm pose at ``q`` intersects any obstacle point.

        Also returns True if any non-adjacent pair of link capsules interpene-
        trates (self-collision) when ``self_collision`` is enabled.
        """
        capsules = self._fk.link_endpoints(q)
        cloud = self._world.cloud_points()

        if cloud.shape[0] > 0:
            for cap in capsules:
                d = capsule_point_cloud_distance(
                    cap.p0, cap.p1,
                    cap.radius_mm + self._clearance_mm,
                    cloud,
                )
                if d.size and float(d.min()) < 0.0:
                    return True

        if self._self_collision:
            for i, ca in enumerate(capsules):
                for cb in capsules[i + 1:]:
                    if capsule_pair_is_adjacent(ca.name, cb.name):
                        continue
                    gap = capsule_capsule_distance(ca, cb)
                    if gap < 0.0:
                        return True
        return False

    # ── path check ─────────────────────────────────────────────────────────

    def path_in_collision(
        self, q_start: list[int], q_end: list[int], n_samples: int = 15,
    ) -> bool:
        """Interpolate joint-space between ``q_start`` and ``q_end`` and check
        every intermediate (plus both endpoints).

        ``n_samples`` matches the Arduino's internal joint-space interpolation
        granularity (research §1 — 10-20 samples eliminates the polar-vs-joint-
        space midpoint gap the legacy code had).
        """
        if n_samples < 2:
            n_samples = 2
        a = np.asarray(q_start, dtype=np.float64)
        b = np.asarray(q_end, dtype=np.float64)
        for k in range(n_samples + 1):
            t = k / float(n_samples)
            q = np.rint((1.0 - t) * a + t * b).astype(int).tolist()
            if self.config_in_collision(q):
                return True
        return False

    # ── integration adapter ────────────────────────────────────────────────

    def collision_fn(self) -> Callable[[np.ndarray], bool]:
        """Adapter for ``pybullet_planning.birrt``.

        pybullet-planning's BiRRT expects a callable that takes a configuration
        (any sequence) and returns True if it's in collision. It also passes
        occasional kwargs (``diagnosis=...`` from ``meta.check_direct``), so
        the wrapper accepts ``**kwargs`` and ignores them. We round to int
        degrees at the boundary so capsule FK sees the same integer joints
        the Arduino protocol uses.
        """
        def _fn(q, **_kwargs) -> bool:
            q_int = [int(round(float(x))) for x in q]
            return self.config_in_collision(q_int)
        return _fn

    # ── diagnostics ────────────────────────────────────────────────────────

    def nearest_obstacle(
        self, q: list[int],
    ) -> Optional[tuple[str, float, np.ndarray]]:
        """Return (capsule_name, gap_mm, nearest_point_xyz) or None.

        Used by the manual-mode refusal dialog (research §8) to tell the user
        *which* link would collide and how far away the nearest obstacle is.
        """
        capsules = self._fk.link_endpoints(q)
        cloud = self._world.cloud_points()
        if cloud.shape[0] == 0:
            return None
        best: Optional[tuple[str, float, np.ndarray]] = None
        for cap in capsules:
            d = capsule_point_cloud_distance(
                cap.p0, cap.p1,
                cap.radius_mm + self._clearance_mm,
                cloud,
            )
            if d.size == 0:
                continue
            idx = int(np.argmin(d))
            gap = float(d[idx])
            if best is None or gap < best[1]:
                best = (cap.name, gap, cloud[idx].copy())
        return best
