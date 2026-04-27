"""Deterministic sector waypoint planner for the Braccio sweep demo."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


Pose = tuple[float, float, float]
ClearanceFn = Callable[[Pose, "ObstacleMemory"], float]


@dataclass
class ObstacleMemory:
    """Temporary obstacle estimate used for one avoidance episode."""

    obstacle_id: int
    source_channel: int
    centroid_m: list[float]
    theta_deg: float
    r_mm: float
    z_mm: float
    radius_mm: float
    confidence: float
    created_monotonic: float
    sample_count: int
    support_channels: list[int] = field(default_factory=list)


@dataclass
class CandidatePlan:
    name: str
    waypoints: list[Pose]
    clearance_mm: float
    path_cost: float
    accepted: bool
    reject_reason: str = ""
    start_clearance_mm: float = float("nan")
    end_clearance_mm: float = float("nan")
    egress_accepted: bool = False


@dataclass
class SectorPlan:
    selected: CandidatePlan | None
    candidates: list[CandidatePlan]
    reason: str


class SectorWaypointPlanner:
    """Small, deterministic replacement for the old planning stack.

    The planner intentionally does not solve an optimization problem. It
    generates a few conservative polar-space maneuvers, checks each waypoint
    path against the stored obstacle sector, and picks the shortest safe path.
    """

    def __init__(
        self,
        safe_clearance_mm: float = 75.0,
        step_theta_deg: float = 2.0,
        step_mm: float = 10.0,
        egress_tolerance_mm: float = 20.0,
        r_limits_mm: tuple[float, float] = (10.0, 240.0),
        z_limits_mm: tuple[float, float] = (0.0, 200.0),
        theta_limits_deg: tuple[float, float] = (0.0, 180.0),
    ) -> None:
        self.safe_clearance_mm = float(safe_clearance_mm)
        self.step_theta_deg = float(step_theta_deg)
        self.step_mm = float(step_mm)
        self.egress_tolerance_mm = float(max(0.0, egress_tolerance_mm))
        self.r_limits_mm = tuple(float(x) for x in r_limits_mm)
        self.z_limits_mm = tuple(float(x) for x in z_limits_mm)
        self.theta_limits_deg = tuple(float(x) for x in theta_limits_deg)

    def select(
        self,
        current_pose: Pose,
        goal_pose: Pose,
        obstacle: ObstacleMemory,
        clearance_fn: ClearanceFn,
        opening_pose: Pose | None = None,
    ) -> SectorPlan:
        candidates: list[CandidatePlan] = []
        start_clearance = float(clearance_fn(current_pose, obstacle))
        for name, anchors in self._candidate_anchors(current_pose, goal_pose, obstacle, opening_pose):
            path = self._path_from_anchors(current_pose, anchors)
            clearances = self._path_clearances(path, obstacle, clearance_fn)
            finite = [v for v in clearances if np.isfinite(v)]
            clearance = float(min(finite)) if finite else float("-inf")
            end_clearance = float(finite[-1]) if finite else float("-inf")
            cost = self._path_cost(current_pose, path)
            accepted = bool(np.isfinite(clearance) and clearance >= self.safe_clearance_mm)
            egress = False
            if not accepted and np.isfinite(start_clearance) and start_clearance < self.safe_clearance_mm:
                egress_floor = start_clearance - self.egress_tolerance_mm
                egress = bool(
                    finite
                    and clearance >= egress_floor
                    and max(finite) >= self.safe_clearance_mm
                    and end_clearance >= self.safe_clearance_mm
                )
                accepted = egress
            if accepted:
                reason = "egress_from_detected_sector" if egress else ""
            else:
                reason = f"clearance {clearance:.1f}mm < {self.safe_clearance_mm:.1f}mm"
            candidates.append(
                CandidatePlan(
                    name=name,
                    waypoints=path,
                    clearance_mm=float(clearance),
                    path_cost=float(cost),
                    accepted=accepted,
                    reject_reason=reason,
                    start_clearance_mm=float(start_clearance),
                    end_clearance_mm=float(end_clearance),
                    egress_accepted=bool(egress),
                )
            )

        accepted = [c for c in candidates if c.accepted]
        if not accepted:
            return SectorPlan(selected=None, candidates=candidates, reason="no_safe_sector_waypoint_candidate")

        normal = [c for c in accepted if not c.egress_accepted and c.clearance_mm >= self.safe_clearance_mm]
        if not normal:
            accepted.sort(
                key=lambda c: (
                    not c.name.startswith("vertical_opening"),
                    float(c.path_cost),
                    -float(c.end_clearance_mm),
                )
            )
            return SectorPlan(selected=accepted[0], candidates=candidates, reason="selected_egress")

        normal.sort(key=lambda c: (-float(c.clearance_mm), float(c.path_cost)))
        # Prefer paths that are comfortably safe, then shortest among that set.
        best_clearance = float(normal[0].clearance_mm)
        comfortable = [c for c in normal if c.clearance_mm >= max(self.safe_clearance_mm, best_clearance - 20.0)]
        comfortable.sort(key=lambda c: (float(c.path_cost), -float(c.clearance_mm)))
        return SectorPlan(selected=comfortable[0], candidates=candidates, reason="selected")

    def _candidate_anchors(
        self,
        current_pose: Pose,
        goal_pose: Pose,
        obstacle: ObstacleMemory,
        opening_pose: Pose | None = None,
    ) -> list[tuple[str, list[Pose]]]:
        cur_t, cur_r, cur_z = [float(v) for v in current_pose]
        goal_t, goal_r, goal_z = [float(v) for v in goal_pose]
        direction = 1.0 if goal_t >= cur_t else -1.0
        theta_gap = abs(goal_t - cur_t)
        theta_mid = self._clamp_theta(cur_t + direction * min(24.0, max(10.0, theta_gap * 0.45)))

        obstacle_z = float(obstacle.z_mm)
        z_raise = self._clamp_z(max(cur_z + 55.0, obstacle_z + 70.0, goal_z + 35.0))
        r_retract = self._clamp_r(min(cur_r - 35.0, float(obstacle.r_mm) - 40.0, goal_r - 20.0))
        r_extend = self._clamp_r(max(cur_r + 30.0, float(obstacle.r_mm) + 40.0, goal_r + 20.0))
        theta_back = self._clamp_theta(cur_t - direction * 9.0)

        candidates: list[tuple[str, list[Pose]]] = []
        if opening_pose is not None:
            _open_t, open_r, open_z = [float(v) for v in opening_pose]
            z_open = self._clamp_z(max(cur_z, open_z))
            r_open = self._clamp_r(open_r)
            candidates.extend(
                [
                    (
                        "vertical_opening_traverse",
                        [(cur_t, cur_r, z_open), (goal_t, goal_r, z_open), goal_pose],
                    ),
                    (
                        "vertical_opening_retract_traverse",
                        [
                            (cur_t, r_retract, z_open),
                            (goal_t, r_retract, z_open),
                            (goal_t, r_open, z_open),
                            goal_pose,
                        ],
                    ),
                ]
            )

        candidates.extend(
            [
                (
                    "raise_and_continue",
                    [(cur_t, cur_r, z_raise), (theta_mid, cur_r, z_raise), goal_pose],
                ),
                (
                    "retract_and_continue",
                    [(cur_t, r_retract, cur_z), (theta_mid, r_retract, cur_z), goal_pose],
                ),
                (
                    "raise_plus_retract",
                    [(cur_t, r_retract, z_raise), (theta_mid, r_retract, z_raise), goal_pose],
                ),
                (
                    "extend_then_rotate",
                    [(cur_t, r_extend, cur_z), (theta_mid, r_extend, cur_z), goal_pose],
                ),
                (
                    "backstep_raise_retract",
                    [(theta_back, r_retract, z_raise), (theta_mid, r_retract, z_raise), goal_pose],
                ),
            ]
        )
        return candidates

    def _path_from_anchors(self, start: Pose, anchors: list[Pose]) -> list[Pose]:
        out: list[Pose] = []
        prev = start
        for anchor in anchors:
            out.extend(self._segment(prev, anchor))
            prev = anchor
        return out

    def _segment(self, start: Pose, end: Pose) -> list[Pose]:
        s = np.asarray(start, dtype=float)
        e = np.asarray(end, dtype=float)
        max_steps = max(
            abs(float(e[0] - s[0])) / max(1.0, self.step_theta_deg),
            abs(float(e[1] - s[1])) / max(1.0, self.step_mm),
            abs(float(e[2] - s[2])) / max(1.0, self.step_mm),
        )
        n = max(1, int(math.ceil(max_steps)))
        out: list[Pose] = []
        for idx in range(1, n + 1):
            a = float(idx) / float(n)
            p = (1.0 - a) * s + a * e
            out.append((self._clamp_theta(p[0]), self._clamp_r(p[1]), self._clamp_z(p[2])))
        return out

    @staticmethod
    def _path_clearances(path: list[Pose], obstacle: ObstacleMemory, clearance_fn: ClearanceFn) -> list[float]:
        if not path:
            return [float("inf")]
        return [float(clearance_fn(pose, obstacle)) for pose in path]

    @staticmethod
    def _path_cost(start: Pose, path: list[Pose]) -> float:
        if not path:
            return 0.0
        cost = 0.0
        prev = np.asarray(start, dtype=float)
        for pose in path:
            cur = np.asarray(pose, dtype=float)
            cost += abs(float(cur[0] - prev[0])) * 1.6
            cost += abs(float(cur[1] - prev[1]))
            cost += abs(float(cur[2] - prev[2]))
            prev = cur
        return float(cost)

    def _clamp_theta(self, value: float) -> float:
        lo, hi = self.theta_limits_deg
        return float(max(lo, min(hi, float(value))))

    def _clamp_r(self, value: float) -> float:
        lo, hi = self.r_limits_mm
        return float(max(lo, min(hi, float(value))))

    def _clamp_z(self, value: float) -> float:
        lo, hi = self.z_limits_mm
        return float(max(lo, min(hi, float(value))))
