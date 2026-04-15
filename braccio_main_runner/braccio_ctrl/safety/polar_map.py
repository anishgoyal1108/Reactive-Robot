"""
polar_map.py — 1D polar obstacle map for sweep skip-ahead.

Research §7 root-cause analysis: the legacy sweep reverses direction by 30° on
obstacle hits, but 30° is smaller than the VL53L5CX 45° FOV, so the obstacle
stays in the sensor cone and the sweep re-triggers — mathematically
guaranteed oscillation.

Solution: maintain a 1D map indexed by base angle θ ∈ [0°, 180°] that
remembers *where* obstacles sit. When the sweep is blocked at θ, it looks up
the edge of the blocked band and jumps to the far side in one move, rather
than reversing by a fixed amount and running back into the same obstacle.

API:

    PolarObstacleMap(bins=180, fov_half_deg=22.5, decay_s=15.0)
        mark_blocked(theta_deg)           — expand a FOV-wide band around θ
        is_blocked(theta_deg)             — is θ inside any blocked band?
        next_unblocked(from_theta, dir)   — scan in ±1 direction for the first
                                            unblocked bin (None if saturated)
        blocked_ranges()                  — list of (lo, hi) degree intervals
        decay(now)                        — drop bins older than ``decay_s``
        clear()                           — reset

The caller decides *how* blocks are sourced. The sweep branch uses hysteresis-
filtered ToF hits at the current θ; the sequence branch pulls them from
``WorldModel.polar_blocked_ranges(r, z)``.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional


class PolarObstacleMap:
    """1D angular obstacle memory over θ ∈ [0°, 180°].

    The domain is uniformly discretised into ``bins`` buckets; each bucket
    carries a timestamp, and a block applies to every bucket whose centre
    falls inside ``[θ - fov_half_deg, θ + fov_half_deg]``. Aging is handled
    by ``decay()``: buckets older than ``decay_s`` are cleared in-place.

    Thread-safe.
    """

    def __init__(
        self,
        bins: int = 180,
        fov_half_deg: float = 22.5,
        decay_s: float = 15.0,
        theta_min: float = 0.0,
        theta_max: float = 180.0,
    ):
        if bins <= 0:
            raise ValueError("bins must be positive")
        if theta_max <= theta_min:
            raise ValueError("theta_max must exceed theta_min")
        self._lock = threading.RLock()
        self._bins = int(bins)
        self._fov_half = float(fov_half_deg)
        self._decay_s = float(decay_s)
        self._theta_min = float(theta_min)
        self._theta_max = float(theta_max)
        self._span = self._theta_max - self._theta_min
        # Per-bucket timestamp; 0.0 = never blocked.
        self._ts: list[float] = [0.0] * self._bins

    # ── Coordinate helpers ─────────────────────────────────────────────────

    def _bin_index(self, theta_deg: float) -> Optional[int]:
        t = max(self._theta_min, min(self._theta_max, float(theta_deg)))
        frac = (t - self._theta_min) / self._span
        idx = int(math.floor(frac * self._bins))
        if idx >= self._bins:
            idx = self._bins - 1
        return idx

    def _bin_center_deg(self, idx: int) -> float:
        return self._theta_min + (idx + 0.5) * self._span / self._bins

    # ── Mutation ───────────────────────────────────────────────────────────

    def mark_blocked(
        self, theta_deg: float, now: Optional[float] = None,
    ) -> None:
        """Block a ``±fov_half_deg`` band centred on ``theta_deg``."""
        t_now = float(time.time() if now is None else now)
        lo = theta_deg - self._fov_half
        hi = theta_deg + self._fov_half
        with self._lock:
            for idx in range(self._bins):
                c = self._bin_center_deg(idx)
                if lo <= c <= hi:
                    self._ts[idx] = t_now

    def mark_range(
        self, theta_lo_deg: float, theta_hi_deg: float,
        now: Optional[float] = None,
    ) -> None:
        """Block every bin whose centre falls in [lo, hi]."""
        t_now = float(time.time() if now is None else now)
        with self._lock:
            for idx in range(self._bins):
                c = self._bin_center_deg(idx)
                if theta_lo_deg <= c <= theta_hi_deg:
                    self._ts[idx] = t_now

    def decay(self, now: Optional[float] = None) -> int:
        """Clear buckets older than ``decay_s``. Returns count cleared."""
        t_now = float(time.time() if now is None else now)
        cleared = 0
        cutoff = t_now - self._decay_s
        with self._lock:
            for idx in range(self._bins):
                if 0.0 < self._ts[idx] < cutoff:
                    self._ts[idx] = 0.0
                    cleared += 1
        return cleared

    def clear(self) -> None:
        with self._lock:
            for idx in range(self._bins):
                self._ts[idx] = 0.0

    # ── Queries ────────────────────────────────────────────────────────────

    def is_blocked(self, theta_deg: float) -> bool:
        with self._lock:
            idx = self._bin_index(theta_deg)
            return idx is not None and self._ts[idx] > 0.0

    def next_unblocked(
        self, from_theta_deg: float, direction: int,
        margin_deg: float = 5.0,
    ) -> Optional[float]:
        """Scan ``dir`` from ``from_theta_deg`` and return a safe θ on the
        far side of the next blocked band.

        Semantics (the sweep use-case from research §7):
          * Walk forward in ``direction`` until we hit a blocked bin.
          * Walk past the blocked band until we re-enter unblocked bins.
          * Walk ``margin_deg`` further so we land safely past the band.
          * Return the bin centre (degrees) at that landing spot.

        Returns ``None`` when there is no blocked band ahead (nothing to
        skip — caller can just keep sweeping), or when the rest of the scan
        in ``direction`` is fully blocked.
        """
        if direction not in (-1, +1):
            raise ValueError("direction must be -1 or +1")
        margin_bins = max(1, int(math.ceil(margin_deg * self._bins / self._span)))
        with self._lock:
            start = self._bin_index(from_theta_deg)
            if start is None:
                return None
            idx = start
            # Phase 1: walk through unblocked bins looking for the first block.
            while 0 <= idx < self._bins and self._ts[idx] == 0.0:
                idx += direction
            if not (0 <= idx < self._bins):
                return None  # nothing blocked in this direction → no skip needed

            # Phase 2: walk through the blocked band until we re-enter
            # unblocked bins.
            while 0 <= idx < self._bins and self._ts[idx] > 0.0:
                idx += direction
            if not (0 <= idx < self._bins):
                return None  # blocked all the way to the domain edge

            # Phase 3: walk ``margin_bins`` further in ``direction`` so the
            # caller lands past the FOV edge.
            landing = idx + direction * margin_bins
            if not (0 <= landing < self._bins):
                return None

            # Guard: verify no block re-appears inside the margin window.
            lo, hi = (min(idx, landing), max(idx, landing))
            for probe in range(lo, hi + 1):
                if self._ts[probe] > 0.0:
                    return None

            return self._bin_center_deg(landing)

    def blocks_direction_to_edge(
        self, from_theta_deg: float, direction: int,
    ) -> bool:
        """Does a blocked band stand between ``from_theta_deg`` and the far
        edge in ``direction``?

        Distinguishes a one-sided block (user's hand in the right half-plane)
        from a narrow notch we can skip. Returns ``True`` when *any* bin
        between the starting θ and the domain edge in ``direction`` is
        blocked — i.e. the sweep can't reach the opposite edge at the
        current z without either climbing or routing around.

        Use this as the trigger for the z-ladder + BiRRT fallback: if
        ``next_unblocked`` succeeded but the resulting skip still leaves a
        blocked band ahead of us, we need to go over/under.
        """
        if direction not in (-1, +1):
            raise ValueError("direction must be -1 or +1")
        with self._lock:
            idx = self._bin_index(from_theta_deg)
            if idx is None:
                return False
            if direction > 0:
                probe_range = range(idx, self._bins)
            else:
                probe_range = range(idx, -1, -1)
            return any(self._ts[i] > 0.0 for i in probe_range)

    def blocked_ranges(self) -> list[tuple[float, float]]:
        """Contiguous blocked intervals (in degrees)."""
        out: list[tuple[float, float]] = []
        with self._lock:
            start: Optional[int] = None
            for idx in range(self._bins):
                blocked = self._ts[idx] > 0.0
                if blocked and start is None:
                    start = idx
                elif not blocked and start is not None:
                    lo = self._bin_center_deg(start) - 0.5 * self._span / self._bins
                    hi = self._bin_center_deg(idx - 1) + 0.5 * self._span / self._bins
                    out.append((lo, hi))
                    start = None
            if start is not None:
                lo = self._bin_center_deg(start) - 0.5 * self._span / self._bins
                hi = self._bin_center_deg(self._bins - 1) + 0.5 * self._span / self._bins
                out.append((lo, hi))
        return out

    def snapshot(self) -> dict:
        with self._lock:
            n_blocked = sum(1 for t in self._ts if t > 0.0)
        return {
            "bins": self._bins,
            "blocked_bins": n_blocked,
            "theta_range": (self._theta_min, self._theta_max),
            "fov_half_deg": self._fov_half,
        }
