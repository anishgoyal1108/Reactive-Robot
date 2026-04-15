"""Integration tests for the py_trees behavior tree root.

These tests use real SafetyPlanner/CollisionChecker/ForwardKinematics, so
they exercise the full stack end-to-end in ticks — catching any binding
issues between branches, blackboard, and planner.
"""

import threading

import numpy as np
import pytest
import py_trees

from braccio_ctrl.safety.behavior import (
    BehaviorTreeRunner,
    SafetyBlackboard,
    build_root,
)
from braccio_ctrl.safety.collision import CollisionChecker
from braccio_ctrl.safety.fk import ForwardKinematics
from braccio_ctrl.safety.hysteresis import SchmittTrigger
from braccio_ctrl.safety.planner import SafetyPlanner
from braccio_ctrl.safety.polar_map import PolarObstacleMap
from braccio_ctrl.safety.replanner import CascadingReplanner, Waypoint
from braccio_ctrl.safety.world_model import WorldModel


@pytest.fixture(scope="module")
def fk():
    f = ForwardKinematics()
    yield f
    f.disconnect()


@pytest.fixture
def stack(fk):
    world = WorldModel()
    checker = CollisionChecker(world, fk, clearance_mm=20.0, self_collision=False)
    planner = SafetyPlanner(checker, fk, default_timeout_s=1.0)
    replanner = CascadingReplanner(
        planner, world, max_retries=1, wait_s=0.0,
        timeout_extended_s=1.0, sleep_fn=lambda _s: None,
    )
    polar = PolarObstacleMap()
    hysteresis = {
        0: SchmittTrigger(250.0, 350.0),
        1: SchmittTrigger(250.0, 350.0),
        2: SchmittTrigger(250.0, 350.0),
    }

    bb = SafetyBlackboard(name=f"bb_{id(world)}")
    root = build_root(bb, planner, replanner, world, polar, hysteresis)
    return {
        "bb": bb,
        "root": root,
        "world": world,
        "planner": planner,
        "replanner": replanner,
        "polar": polar,
    }


def _tick(root):
    root.tick_once()


# ── Manual branch ──────────────────────────────────────────────────────────

def test_manual_intent_clears_on_success(stack):
    stack["bb"].mode = "manual"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].manual_intent = [95, 90, 90, 90, 90, 73]
    _tick(stack["root"])
    assert stack["bb"].pending_command is not None
    assert stack["bb"].last_strategy == "direct"
    assert stack["bb"].refusal is None


def test_manual_escape_fires_when_start_in_collision(stack, fk):
    """When the current config is in collision (e.g. a fresh ToF point
    landed on a capsule), the manual branch now runs
    ``escape_from_collision`` and issues a command to a nearby safe pose
    instead of deadlocking. This preempts the refusal dialog because the
    arm can still make forward progress.
    """
    caps = {c.name: c for c in fk.link_endpoints([90, 90, 90, 90, 90, 73])}
    stack["world"].ingest_points(np.array([list(caps["gripper"].p1)]))

    stack["bb"].strict_mode = True
    stack["bb"].mode = "manual"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].manual_intent = [95, 90, 90, 90, 90, 73]
    _tick(stack["root"])
    assert stack["bb"].pending_command is not None
    assert stack["bb"].last_strategy == "escape_collision"
    assert stack["bb"].last_failure == "start_in_collision"
    assert stack["bb"].refusal is None


def test_manual_refusal_silent_by_default(stack, fk):
    """With strict_mode off, a fully blocked manual intent is silently
    cleared — no refusal payload, no pending command."""
    caps = {c.name: c for c in fk.link_endpoints([90, 90, 90, 90, 90, 73])}
    stack["world"].ingest_points(np.array([list(caps["gripper"].p1)]))

    stack["bb"].strict_mode = False
    stack["bb"].mode = "manual"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].manual_intent = [95, 90, 90, 90, 90, 73]
    _tick(stack["root"])
    # Branch either found a nudge detour (→ pending_command set) or gave
    # up silently. Either way, refusal must stay None.
    assert stack["bb"].refusal is None
    assert stack["bb"].last_strategy in (
        "refused", "direct", "escape_collision"
    ) or stack["bb"].last_strategy.startswith("manual_nudge")


# ── Emergency branch ───────────────────────────────────────────────────────

def test_emergency_preempts_other_branches(stack):
    stack["bb"].mode = "manual"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].manual_intent = [95, 90, 90, 90, 90, 73]
    stack["bb"].ir_severity = 3  # DANGER
    _tick(stack["root"])
    assert stack["bb"].emergency is True
    assert stack["bb"].bt_state == "emergency_halt"
    cmd = stack["bb"].pending_command
    assert cmd is not None
    assert cmd["joints"] == [90, 90, 90, 90, 90, 73]


def test_emergency_releases_when_ir_clears(stack):
    stack["bb"].ir_severity = 2
    _tick(stack["root"])
    assert stack["bb"].emergency
    stack["bb"].ir_severity = 0
    _tick(stack["root"])
    assert not stack["bb"].emergency


# ── Sweep branch ───────────────────────────────────────────────────────────

def test_sweep_advances_when_clear(stack):
    stack["bb"].mode = "sweep"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    _tick(stack["root"])
    # Sweep advances by SWEEP_STEP_DEG = 5 in +1 direction.
    cmd = stack["bb"].pending_command
    assert cmd is not None
    assert cmd["joints"][0] in (93, 94, 95)
    assert stack["bb"].bt_state == "sweep_running"


def test_sweep_skips_blocked_band(stack):
    stack["polar"].mark_range(91.0, 115.0)  # block the next ~25°
    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    _tick(stack["root"])
    cmd = stack["bb"].pending_command
    assert cmd is not None
    # Expect the sweep to skip past the 91-115 band + margin.
    assert cmd["joints"][0] > 115


def test_sweep_one_sided_block_triggers_z_ladder(stack):
    """User's hand over the right half of the sweep blocks 91-180. Forward
    skip-via-polar fails (block extends to the domain edge), so the arm
    must climb the z-ladder instead of immediately flipping direction.
    """
    stack["polar"].mark_range(91.0, 180.0)
    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].sweep_z_index = 0
    stack["bb"].sweep_z_tries = 0

    _tick(stack["root"])

    assert stack["bb"].last_strategy.startswith("sweep_z_ladder"), (
        f"expected sweep_z_ladder, got {stack['bb'].last_strategy!r}"
    )
    assert stack["bb"].sweep_z_index == 1
    assert stack["bb"].sweep_z_tries == 1
    assert stack["bb"].bt_state == "sweep_z_ladder"


def test_sweep_escalates_to_birrt_over_when_z_ladder_exhausted(stack):
    """After len(z_ladder)-1 z-advances without clearing the block, the
    sweep fires one BiRRT to the opposite edge so the planner can lift the
    elbow over the obstacle. The world model has no points so BiRRT is
    guaranteed to succeed; we just need to verify the strategy fires and a
    command is emitted.
    """
    stack["polar"].mark_range(91.0, 180.0)
    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    # Pretend we've already climbed the z-ladder to its last rung.
    # SWEEP_Z_LADDER_MM has 5 entries, so 4 prior tries trips the BiRRT path.
    from braccio_ctrl.safety.behavior import SWEEP_Z_LADDER_MM
    stack["bb"].sweep_z_tries = len(SWEEP_Z_LADDER_MM) - 1
    stack["bb"].sweep_z_index = len(SWEEP_Z_LADDER_MM) - 1

    _tick(stack["root"])

    assert stack["bb"].last_strategy in (
        "sweep_birrt_over", "sweep_edge_reverse"
    )
    if stack["bb"].last_strategy == "sweep_birrt_over":
        cmd = stack["bb"].pending_command
        assert cmd is not None
        # Reset so follow-on ticks don't immediately re-fire BiRRT.
        assert stack["bb"].sweep_z_tries == 0


def test_sweep_narrow_notch_still_uses_polar_skip(stack):
    """A narrow notch (middle of the sweep, not touching the domain edge)
    should still take the cheap polar-skip path and not climb the z-ladder.
    Regression guard for the 'one-sided block triggers z-ladder' change.
    """
    stack["polar"].mark_range(100.0, 110.0)   # tiny block mid-sweep
    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = [95, 90, 90, 90, 90, 73]
    stack["bb"].sweep_z_index = 0
    stack["bb"].sweep_z_tries = 0

    _tick(stack["root"])

    assert stack["bb"].last_strategy == "sweep_direct"
    assert stack["bb"].sweep_z_index == 0
    assert stack["bb"].sweep_z_tries == 0


def test_sweep_descends_back_toward_origin_after_clear(stack):
    """After climbing the z-ladder to overtake an obstacle, a successful
    sweep_direct tick must step the z-index one rung back toward origin
    ('return to the right lane after an overtake')."""
    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].sweep_z_origin = 0
    stack["bb"].sweep_z_index = 2      # we're currently on the 3rd rung
    stack["bb"].sweep_z_tries = 0
    # Polar map is empty — forward path is clear.

    _tick(stack["root"])

    assert stack["bb"].sweep_z_index == 1, (
        "sweep_direct clear tick must descend one rung toward origin"
    )
    assert stack["bb"].last_strategy.startswith("sweep_descend:")


def test_sweep_stays_at_origin_when_already_there(stack):
    """No descent when z_index == z_origin; the sweep is already in the
    intended lane."""
    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].sweep_z_origin = 0
    stack["bb"].sweep_z_index = 0

    _tick(stack["root"])

    assert stack["bb"].sweep_z_index == 0
    assert stack["bb"].last_strategy == "sweep_direct"


def test_shift_z_index_bounds_clamped():
    """_shift_z_index must refuse moves that would fall outside
    [0, len(ladder)). Below 0 and above the ladder both return False."""
    from braccio_ctrl.safety.behavior import _SweepTick, SWEEP_Z_LADDER_MM

    class FakeBB:
        sweep_z_index = 0

    class FakePolar:
        cleared = 0
        def clear(self):
            self.cleared += 1

    bb = FakeBB()
    polar = FakePolar()
    # Construct a _SweepTick with the minimum kwargs and patch the two
    # attributes _shift_z_index actually touches.
    tick = _SweepTick.__new__(_SweepTick)
    tick._bb = bb
    tick._polar = polar
    tick._z_ladder = SWEEP_Z_LADDER_MM

    # Below zero — refuse.
    bb.sweep_z_index = 0
    assert tick._shift_z_index(-1) is False
    assert bb.sweep_z_index == 0
    assert polar.cleared == 0

    # Above ladder — refuse.
    bb.sweep_z_index = len(SWEEP_Z_LADDER_MM) - 1
    assert tick._shift_z_index(+1) is False
    assert bb.sweep_z_index == len(SWEEP_Z_LADDER_MM) - 1

    # Zero delta — no-op, returns False without clearing polar.
    bb.sweep_z_index = 2
    assert tick._shift_z_index(0) is False
    assert bb.sweep_z_index == 2
    assert polar.cleared == 0

    # Valid shift clears polar and updates index.
    bb.sweep_z_index = 2
    assert tick._shift_z_index(+1) is True
    assert bb.sweep_z_index == 3
    assert polar.cleared == 1

    assert tick._shift_z_index(-1) is True
    assert bb.sweep_z_index == 2
    assert polar.cleared == 2


# ── Sequence branch ────────────────────────────────────────────────────────

def test_sequence_plays_back_waypoints(stack):
    stack["bb"].mode = "sequence"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].sequence_queue = [Waypoint(q=[100, 90, 90, 90, 90, 73])]
    # First tick: plan the segment and emit the first step
    _tick(stack["root"])
    assert stack["bb"].bt_state in ("sequence_playback", "sequence_planning",
                                    "sequence_done")
    cmd = stack["bb"].pending_command
    if cmd is not None:
        assert cmd["joints"][0] >= 90


# ── Runner ─────────────────────────────────────────────────────────────────

def test_runner_sends_commands_via_callback(stack):
    sent = []
    runner = BehaviorTreeRunner(stack["bb"], stack["root"], send_cmd=sent.append,
                                 tick_hz=50.0)
    stack["bb"].mode = "manual"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].manual_intent = [95, 90, 90, 90, 90, 73]
    runner.tick_once()
    assert len(sent) == 1
    assert sent[0]["joints"][0] in (91, 92, 93, 94, 95)


def test_runner_daemon_thread_starts_and_stops(stack):
    calls = []
    runner = BehaviorTreeRunner(stack["bb"], stack["root"],
                                 send_cmd=calls.append, tick_hz=100.0)
    runner.start()
    # Let it tick a few times then stop.
    import time
    time.sleep(0.1)
    runner.stop(timeout=0.5)
    # Calls list may be empty (nothing to send at mode='idle') but the
    # thread must have exited cleanly.
    assert runner._thread is None
