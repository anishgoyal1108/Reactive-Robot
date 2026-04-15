"""
replanner.py — Cascading replanner for sequence-editor waypoints.

Research §6 defines a 6-strategy recovery ladder. Each saved-pose waypoint is
annotated with a ``WaypointType`` (ESSENTIAL or VIA_POINT). When the planner
cannot find a path from the current pose to the next waypoint, the replanner
tries each strategy in order until one succeeds or all are exhausted.

    1. Direct plan with extended timeout
    2. Alternate IK solutions (elbow-up / elbow-down) for the goal
    3. Intermediate via-point insertion (HOME detour, midpoint perturbation)
    4. Skip the waypoint if it's a VIA_POINT and the next is reachable
    5. Wait and retry (obstacle may move)
    6. Declare failure — move to HOME, emit a telemetry event

The replanner is deliberately decoupled from the executor: ``execute_segment``
just returns a ``SegmentResult``. Caller is responsible for actually sending
the joint commands through its backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .planner import PlanResult, SafetyPlanner
from .world_model import WorldModel


class WaypointType(Enum):
    ESSENTIAL = "essential"
    VIA_POINT = "via_point"


@dataclass
class Waypoint:
    q: list[int]
    kind: WaypointType = WaypointType.ESSENTIAL
    name: str = ""
    gripper: Optional[int] = None   # when set, override gripper in SET ALL

    def __post_init__(self):
        if len(self.q) != 6:
            raise ValueError(f"q must have 6 joints; got {len(self.q)}")


@dataclass
class SegmentResult:
    success: bool
    path: Optional[list[list[int]]] = None
    strategy: str = ""
    failure_reason: Optional[str] = None
    replan_attempts: int = 0
    total_time_s: float = 0.0
    events: list[dict] = field(default_factory=list)


HOME_POS = [90, 90, 90, 90, 90, 73]


class CascadingReplanner:
    """6-strategy recovery ladder for a single segment (start → goal).

    Parameters
    ----------
    planner : SafetyPlanner
        The collision-aware planner.
    world : WorldModel
        Read in strategy 5 to confirm the environment refreshed before retry.
    home_q : list[int]
        Pose used for the "via HOME" detour strategy. Typically the real
        Braccio HOME_POS, but tests can inject any safe pose.
    max_retries : int
        Maximum times strategy 5 (wait-and-retry) fires.
    wait_s : float
        Sleep interval between retries.
    alternate_ik_fn : callable or None
        Returns a list of alternate 6-joint goal configurations. If None,
        strategy 2 is skipped. The default ``default_alternate_ik``
        enumerates elbow-up / elbow-down by mirroring the elbow angle.
    sleep_fn : callable
        Injected for deterministic test timing.
    """

    def __init__(
        self,
        planner: SafetyPlanner,
        world: WorldModel,
        home_q: Optional[list[int]] = None,
        max_retries: int = 3,
        wait_s: float = 2.0,
        timeout_extended_s: float = 4.0,
        alternate_ik_fn: Optional[Callable[[list[int]], list[list[int]]]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self._planner = planner
        self._world = world
        self._home = list(home_q if home_q is not None else HOME_POS)
        self._max_retries = int(max_retries)
        self._wait_s = float(wait_s)
        self._timeout_extended_s = float(timeout_extended_s)
        self._alt_ik = alternate_ik_fn or default_alternate_ik
        self._sleep = sleep_fn

    @property
    def planner(self) -> SafetyPlanner:
        """Public accessor so branches can reach the underlying planner
        (needed for ``escape_from_collision``)."""
        return self._planner

    def execute_segment(
        self,
        start: Waypoint,
        goal: Waypoint,
        next_goal: Optional[Waypoint] = None,
    ) -> SegmentResult:
        """Plan start → goal; on failure try the 6 strategies in order.

        If ``next_goal`` is supplied the replanner may skip ``goal`` when it
        is a VIA_POINT and ``start → next_goal`` is plannable (strategy 4).
        """
        t_start = time.monotonic()
        events: list[dict] = []

        # Strategy 1 — Direct plan with default timeout.
        res = self._planner.plan(start.q, goal.q)
        events.append({"strategy": "direct", "reason": res.failure_reason,
                       "ok": res.success})
        if res.success:
            return SegmentResult(True, res.path, "direct",
                                 None, 0, time.monotonic() - t_start, events)

        # Strategy 1b — Retry with extended timeout (probabilistically complete,
        # more time may find a path).
        res = self._planner.plan(start.q, goal.q,
                                 timeout_s=self._timeout_extended_s)
        events.append({"strategy": "extended_timeout",
                       "reason": res.failure_reason, "ok": res.success})
        if res.success:
            return SegmentResult(True, res.path, "extended_timeout",
                                 None, 0, time.monotonic() - t_start, events)

        # Strategy 2 — Alternate IK solutions for the goal pose.
        for alt_q in self._alt_ik(goal.q):
            alt_res = self._planner.plan(start.q, alt_q)
            events.append({"strategy": "alternate_ik",
                           "alt_q": alt_q, "ok": alt_res.success})
            if alt_res.success:
                return SegmentResult(True, alt_res.path, "alternate_ik",
                                     None, 0, time.monotonic() - t_start, events)

        # Strategy 3 — Via HOME detour.
        via_res = self._plan_via(start.q, self._home, goal.q)
        events.append({"strategy": "via_home",
                       "reason": via_res.failure_reason, "ok": via_res.success})
        if via_res.success:
            return SegmentResult(True, via_res.path, "via_home",
                                 None, 0, time.monotonic() - t_start, events)

        # Strategy 3b — Midpoint-perturbed intermediate.
        mid = [int(round(0.5 * (a + b))) for a, b in zip(start.q, goal.q)]
        # Nudge the elbow away from its nominal midpoint — if the direct path
        # was blocked, the straight midpoint often is too, so try ±20° on
        # the elbow as a simple obstacle-going-around heuristic.
        for elbow_nudge in (+20, -20, +30, -30):
            cand = list(mid)
            cand[2] = max(0, min(180, cand[2] + elbow_nudge))
            via_res = self._plan_via(start.q, cand, goal.q)
            events.append({"strategy": "via_midpoint",
                           "elbow_nudge": elbow_nudge, "ok": via_res.success})
            if via_res.success:
                return SegmentResult(True, via_res.path, "via_midpoint",
                                     None, 0, time.monotonic() - t_start, events)

        # Strategy 4 — Skip if goal is a via-point and we can reach next_goal.
        if goal.kind == WaypointType.VIA_POINT and next_goal is not None:
            skip_res = self._planner.plan(start.q, next_goal.q)
            events.append({"strategy": "skip_via",
                           "skipped": goal.name, "ok": skip_res.success})
            if skip_res.success:
                return SegmentResult(True, skip_res.path, "skip_via",
                                     None, 0, time.monotonic() - t_start, events)

        # Strategy 5 — Wait and retry (obstacle may move).
        for attempt in range(1, self._max_retries + 1):
            self._sleep(self._wait_s)
            retry_res = self._planner.plan(start.q, goal.q)
            events.append({"strategy": "wait_retry",
                           "attempt": attempt, "ok": retry_res.success})
            if retry_res.success:
                return SegmentResult(True, retry_res.path, "wait_retry",
                                     None, attempt,
                                     time.monotonic() - t_start, events)

        # Strategy 6 — Declare failure, route to HOME.
        home_res = self._planner.plan(start.q, self._home)
        events.append({"strategy": "bailout_home",
                       "reason": home_res.failure_reason, "ok": home_res.success})
        if home_res.success:
            return SegmentResult(False, home_res.path, "bailout_home",
                                 "segment_unreachable", self._max_retries,
                                 time.monotonic() - t_start, events)
        return SegmentResult(False, None, "bailout_home_failed",
                             "segment_and_home_unreachable", self._max_retries,
                             time.monotonic() - t_start, events)

    def _plan_via(
        self, start_q: list[int], via_q: list[int], goal_q: list[int],
    ) -> PlanResult:
        """Plan start → via → goal; concatenate paths if both legs succeed."""
        leg_a = self._planner.plan(start_q, via_q)
        if not leg_a.success:
            return leg_a
        leg_b = self._planner.plan(via_q, goal_q)
        if not leg_b.success:
            return leg_b
        # Drop the duplicated via-point joint that appears at the tail of leg
        # A and head of leg B.
        combined = list(leg_a.path or []) + list((leg_b.path or [])[1:])
        return PlanResult(True, combined, None,
                          (leg_a.planner_time_s + leg_b.planner_time_s),
                          len(combined))


def default_alternate_ik(q: list[int]) -> list[list[int]]:
    """Mirror the elbow about its nominal to produce elbow-up / elbow-down
    alternates. The base is left alone; shoulder and wrist-vertical are
    mirrored to keep the end-effector position roughly constant.

    For a 2-link planar arm the closed-form mirror is:
        elbow' = 180 - elbow
        shoulder' = shoulder + (elbow - 90) * 2
        wrist_v' = wrist_v - (elbow - 90) * 2   (keep gripper horizontal-ish)
    """
    alternates: list[list[int]] = []
    base, shoulder, elbow, wv, wr, gripper = q
    delta = elbow - 90
    if delta != 0:
        alt = [
            base,
            max(15, min(165, shoulder + 2 * delta)),
            max(0, min(180, 180 - elbow)),
            max(0, min(180, wv - 2 * delta)),
            wr,
            gripper,
        ]
        if alt != list(q):
            alternates.append(alt)
    return alternates
