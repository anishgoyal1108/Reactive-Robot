"""Tests for CascadingReplanner strategies.

We test against a fake planner (so we can deterministically decide which
configurations fail) rather than the real pybullet_planning BiRRT. The fake
planner's behavior per-goal is programmable, which makes each strategy
isolable and the tests fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pytest

from braccio_ctrl.safety.planner import PlanResult
from braccio_ctrl.safety.replanner import (
    CascadingReplanner,
    Waypoint,
    WaypointType,
    default_alternate_ik,
)


@dataclass
class FakePlanner:
    """Planner double: calls ``decide(start, goal) -> bool`` to decide success."""
    decide: Callable[[tuple, tuple], bool]
    calls: list[tuple] = field(default_factory=list)

    def plan(self, start, goal, timeout_s=None, smooth=True) -> PlanResult:
        self.calls.append((list(start), list(goal)))
        if self.decide(tuple(start), tuple(goal)):
            return PlanResult(True, [list(start), list(goal)], None, 0.01, 2)
        return PlanResult(False, None, "birrt_timeout", 0.01, 0)


def _make(world_decide):
    planner = FakePlanner(decide=world_decide)
    class _FakeWorld:  # tiny stub; CascadingReplanner only reads reference
        pass
    return CascadingReplanner(
        planner, _FakeWorld(),
        home_q=[90, 90, 90, 90, 90, 73],
        max_retries=2, wait_s=0.0, timeout_extended_s=1.0,
        sleep_fn=lambda _s: None,
    ), planner


def _wp(q, kind=WaypointType.ESSENTIAL, name=""):
    return Waypoint(q=list(q), kind=kind, name=name)


def test_direct_plan_succeeds():
    rep, planner = _make(lambda s, g: True)
    res = rep.execute_segment(_wp([90] * 6), _wp([95] + [90] * 5))
    assert res.success
    assert res.strategy == "direct"
    assert len(planner.calls) == 1


def test_extended_timeout_recovers():
    # Direct call fails; retry with extended timeout (same planner, same
    # signature — FakePlanner decides per-call via a counter).
    counter = {"n": 0}
    def decide(s, g):
        counter["n"] += 1
        return counter["n"] >= 2  # first call fails, second succeeds
    rep, planner = _make(decide)
    res = rep.execute_segment(_wp([90] * 6), _wp([95] + [90] * 5))
    assert res.success
    assert res.strategy == "extended_timeout"


def test_alternate_ik_succeeds():
    # Direct+extended both fail; alternate IK gives a different goal q.
    goal = [90, 120, 60, 90, 90, 73]  # elbow=60 → alt elbow=180-60=120
    alts = default_alternate_ik(goal)
    assert alts, "alt IK must produce at least one alternate"
    alt_q = tuple(alts[0])

    def decide(s, g):
        # Succeed only if goal matches the alternate IK.
        return tuple(g) == alt_q
    rep, planner = _make(decide)
    res = rep.execute_segment(_wp([90] * 6), _wp(goal))
    assert res.success
    assert res.strategy == "alternate_ik"


def test_via_home_strategy():
    HOME = (90, 90, 90, 90, 90, 73)
    goal = (95, 95, 95, 95, 95, 73)

    def decide(s, g):
        # Direct + extended fail. Alternate IK (of goal) fails.
        # Via HOME succeeds: (start→HOME) and (HOME→goal) both clear.
        s_t, g_t = tuple(s), tuple(g)
        if s_t == HOME and g_t == goal:
            return True
        if s_t == (90,) * 6 and g_t == HOME:
            return True
        return False
    rep, planner = _make(decide)
    res = rep.execute_segment(_wp([90] * 6), _wp(list(goal)))
    assert res.success
    assert res.strategy == "via_home"


def test_skip_via_point_when_next_reachable():
    start = (90,) * 6
    goal_via = (100, 90, 90, 90, 90, 73)
    next_goal = (110, 90, 90, 90, 90, 73)

    def decide(s, g):
        # Only start → next_goal is reachable.
        return tuple(s) == start and tuple(g) == next_goal
    rep, planner = _make(decide)
    res = rep.execute_segment(
        _wp(list(start)),
        _wp(list(goal_via), kind=WaypointType.VIA_POINT),
        next_goal=_wp(list(next_goal)),
    )
    assert res.success
    assert res.strategy == "skip_via"


def test_wait_retry_eventually_succeeds():
    """Strategy 5 fires after direct/extended/alt-IK/via-HOME/via-midpoint
    all fail. Goal has elbow=90 so default_alternate_ik yields zero alts —
    the pre-wait call sequence is direct, extended, via_home(leg A fails),
    via_midpoint×4 (each leg A fails). That's 7 failed calls, so succeed on
    the 8th (wait_retry attempt 1).
    """
    attempts = {"count": 0}
    goal = (95, 90, 90, 90, 90, 73)

    def decide(s, g):
        attempts["count"] += 1
        # Fail all strategies up to wait_retry, then succeed — but only on
        # calls that target the real goal (wait_retry does; bailout targets
        # HOME).
        return attempts["count"] >= 8 and tuple(g) == goal
    rep, planner = _make(decide)
    res = rep.execute_segment(_wp([90] * 6), _wp(list(goal)))
    assert res.success, f"expected success via wait_retry; got {res}"
    assert res.strategy == "wait_retry"


def test_bailout_home_when_all_fail():
    HOME = (90, 90, 90, 90, 90, 73)

    def decide(s, g):
        # Only start → HOME succeeds (for bailout leg).
        return tuple(s) == (90,) * 6 and tuple(g) == HOME
    rep, planner = _make(decide)
    res = rep.execute_segment(_wp([90] * 6), _wp([100, 100, 100, 100, 100, 73]))
    assert not res.success
    assert res.strategy == "bailout_home"
    assert res.failure_reason == "segment_unreachable"
    assert res.path is not None  # still returns the HOME escape path


def test_total_failure_returns_no_path():
    def decide(s, g):
        return False  # even HOME is unreachable
    rep, planner = _make(decide)
    res = rep.execute_segment(_wp([90] * 6), _wp([100] * 6))
    assert not res.success
    assert res.strategy == "bailout_home_failed"
    assert res.path is None


def test_waypoint_requires_6_joints():
    with pytest.raises(ValueError):
        Waypoint(q=[1, 2, 3])


def test_default_alternate_ik_elbow_mirror():
    alts = default_alternate_ik([90, 120, 60, 80, 90, 73])
    assert len(alts) == 1
    alt = alts[0]
    assert alt[2] == 120   # 180-60
    assert alt[1] == 60    # shoulder mirrored
    assert alt[3] == 140   # wrist_v compensation clipped to limit


def test_default_alternate_ik_at_straight_elbow():
    # elbow=90 → delta=0 → no alternate
    assert default_alternate_ik([90, 90, 90, 90, 90, 73]) == []
