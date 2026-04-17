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
def stack(fk, monkeypatch):
    # Existing tests exercise the full-planner sweep branch; the simple
    # reactive sweep has its own dedicated tests below. Force the flag off
    # so we keep regression coverage of the escalation ladder.
    import braccio_ctrl.constants as _c
    monkeypatch.setattr(_c, "SIMPLE_REPLAN_MODE", False)
    monkeypatch.setattr(_c, "SIMPLE_SWEEP_MODE", False)  # legacy alias
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


@pytest.fixture
def simple_stack(fk, monkeypatch):
    """Stack wired for the simple replan sweep (SIMPLE_REPLAN_MODE=True)."""
    import braccio_ctrl.constants as _c
    monkeypatch.setattr(_c, "SIMPLE_REPLAN_MODE", True)
    monkeypatch.setattr(_c, "SIMPLE_SWEEP_MODE", True)  # legacy alias
    world = WorldModel()
    checker = CollisionChecker(world, fk, clearance_mm=20.0, self_collision=False)
    planner = SafetyPlanner(checker, fk, default_timeout_s=1.0)
    replanner = CascadingReplanner(
        planner, world, max_retries=1, wait_s=0.0,
        timeout_extended_s=1.0, sleep_fn=lambda _s: None,
    )
    polar = PolarObstacleMap()
    hysteresis = {
        0: SchmittTrigger(100.0, 180.0),
        1: SchmittTrigger(250.0, 350.0),
        2: SchmittTrigger(250.0, 350.0),
    }
    bb = SafetyBlackboard(name=f"simple_bb_{id(world)}")
    root = build_root(bb, planner, replanner, world, polar, hysteresis)
    return {"bb": bb, "root": root, "world": world}


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


# ── Simple replan sweep (demo-safe mode) ──────────────────────────────────

def _seed_block_at(world, theta_deg, r_mm=152.0, z_mm=35.0, n_pts=12):
    """Helper: stuff a cluster of obstacle points into the world cloud at
    the sweep's (r, z) slice centred on ``theta_deg``."""
    import numpy as np
    x = r_mm * np.cos(np.radians(theta_deg))
    y = r_mm * np.sin(np.radians(theta_deg))
    pts = np.array([[x, y, z_mm]] * n_pts)
    world.ingest_points(pts)


def test_simple_replan_advances_when_clear(simple_stack, fk):
    """No ToF hit → advance to next θ on the nominal (r, z) sweep line."""
    from braccio_ctrl.constants import (
        SWEEP_STEP_DEG, SWEEP_R_DEFAULT_MM, SWEEP_Z_DEFAULT_MM,
    )
    from braccio_ctrl.ik_solver import solve_ik, polar_to_cartesian
    # Start from the nominal pose so RateClamp doesn't mask the base step.
    x, y = polar_to_cartesian(90.0, SWEEP_R_DEFAULT_MM)
    sol = solve_ik(x, y, SWEEP_Z_DEFAULT_MM)
    assert sol is not None
    start_q = [int(sol.base), int(sol.shoulder), int(sol.elbow),
               int(sol.wrist_vert), 90, 73]

    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = list(start_q)
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 1000.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])
    cmd = simple_stack["bb"].pending_command
    assert cmd is not None
    assert cmd["joints"][0] == int(round(90 + SWEEP_STEP_DEG))
    assert simple_stack["bb"].last_strategy == "simple_direct"
    assert list(simple_stack["bb"].sweep_detour_path) == []
    # Commanded pose lives on the nominal sweep z-line.
    tip_z = float(fk.link_endpoints(list(cmd["joints"]))[-1].p1[2])
    ref_tip_z = float(fk.link_endpoints(list(start_q))[-1].p1[2])
    assert abs(tip_z - ref_tip_z) <= 15.0


def test_simple_replan_snaps_back_to_nominal_z(simple_stack, fk):
    """Engage sweep from a pose that's NOT on the sweep line → every
    subsequent advance commands a pose on the nominal (r, z) line, so the
    arm slews back to the sweep's z."""
    from braccio_ctrl.constants import SWEEP_R_DEFAULT_MM, SWEEP_Z_DEFAULT_MM
    from braccio_ctrl.ik_solver import solve_ik, polar_to_cartesian

    # Build the expected nominal-line tip z from the reference pose.
    x, y = polar_to_cartesian(90.0, SWEEP_R_DEFAULT_MM)
    sol = solve_ik(x, y, SWEEP_Z_DEFAULT_MM)
    assert sol is not None
    nominal_ref = [int(sol.base), int(sol.shoulder), int(sol.elbow),
                   int(sol.wrist_vert), 90, 73]
    ref_tip_z = float(fk.link_endpoints(list(nominal_ref))[-1].p1[2])

    # Start from a deliberately OFF-line pose (shoulder pushed high).
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [90, 60, 120, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 1000.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])

    cmd = simple_stack["bb"].pending_command
    assert cmd is not None
    assert simple_stack["bb"].last_strategy == "simple_direct"
    tip_z = float(fk.link_endpoints(list(cmd["joints"]))[-1].p1[2])
    assert abs(tip_z - ref_tip_z) <= 15.0, (
        f"sweep must command a pose on the nominal z-line; "
        f"got tip z={tip_z:.1f}, nominal z={ref_tip_z:.1f}"
    )


def test_simple_replan_detour_uses_nominal_z_not_current(
    simple_stack, fk,
):
    """Detour path must target the nominal sweep z, NOT wherever the arm
    drifted to — this is what guarantees the arm returns to the sweep
    line after going around an obstacle."""
    from braccio_ctrl.constants import SWEEP_R_DEFAULT_MM, SWEEP_Z_DEFAULT_MM
    from braccio_ctrl.ik_solver import solve_ik, polar_to_cartesian

    x, y = polar_to_cartesian(90.0, SWEEP_R_DEFAULT_MM)
    sol = solve_ik(x, y, SWEEP_Z_DEFAULT_MM)
    assert sol is not None
    nominal_ref = [int(sol.base), int(sol.shoulder), int(sol.elbow),
                   int(sol.wrist_vert), 90, 73]
    ref_tip_z = float(fk.link_endpoints(list(nominal_ref))[-1].p1[2])

    _seed_block_at(simple_stack["world"], theta_deg=100.0)
    # Start from an OFF-line pose AND fire a ToF hit.
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [90, 60, 120, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 80.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])

    path = list(simple_stack["bb"].sweep_detour_path)
    assert path, "detour should have been built"
    # Final detour waypoint lives on the nominal sweep z-line, regardless
    # of the arm's starting z.
    final_tip_z = float(fk.link_endpoints(list(path[-1]))[-1].p1[2])
    assert abs(final_tip_z - ref_tip_z) <= 25.0, (
        f"detour must end on nominal sweep z={ref_tip_z:.1f}, "
        f"got {final_tip_z:.1f}"
    )


def test_replan_detour_triggers_on_tof_hit_mid_sweep(simple_stack):
    """A ToF hit mid-sweep generates a parametric detour (NOT a reverse)."""
    from braccio_ctrl.constants import SWEEP_DETOUR_STEPS
    _seed_block_at(simple_stack["world"], theta_deg=100.0)
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 80.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])

    path = list(simple_stack["bb"].sweep_detour_path)
    assert len(path) == SWEEP_DETOUR_STEPS
    assert simple_stack["bb"].sweep_detour_idx == 1, (
        "first waypoint sent this tick; next tick picks up idx=1"
    )
    cmd = simple_stack["bb"].pending_command
    assert cmd is not None
    assert cmd["joints"] == list(path[0]), (
        "pending_command must equal the first cached detour waypoint"
    )
    strat = str(simple_stack["bb"].last_strategy)
    assert strat.startswith("sweep_detour:"), (
        f"expected sweep_detour:*, got {strat!r}"
    )
    assert simple_stack["bb"].sweep_direction == +1, (
        "direction must stay the same on a mid-sweep detour"
    )


def test_replan_detour_plays_back_to_completion(simple_stack):
    """Pre-seeded detour path plays one waypoint per tick, then clears."""
    from braccio_ctrl.constants import SWEEP_STEP_DEG
    # Seed a trivially-safe 4-waypoint path around the current pose.
    path = [
        [91, 90, 90, 90, 90, 73],
        [92, 90, 90, 90, 90, 73],
        [94, 90, 90, 90, 90, 73],
        [96, 90, 90, 90, 90, 73],
    ]
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 1000.0, 1000.0, 1000.0]
    simple_stack["bb"].sweep_detour_path = list(path)
    simple_stack["bb"].sweep_detour_idx = 0

    sent = []
    for _ in range(len(path)):
        _tick(simple_stack["root"])
        cmd = simple_stack["bb"].pending_command
        assert cmd is not None
        sent.append(list(cmd["joints"]))

    assert sent == path
    assert list(simple_stack["bb"].sweep_detour_path) == []
    assert simple_stack["bb"].sweep_direction == +1

    # Next tick should resume a normal sweep advance from current_q.
    cur_theta = simple_stack["bb"].current_q[0]
    _tick(simple_stack["root"])
    cmd = simple_stack["bb"].pending_command
    assert cmd is not None
    assert cmd["joints"][0] == int(round(cur_theta + SWEEP_STEP_DEG))
    assert simple_stack["bb"].last_strategy == "simple_direct"


def test_replan_detour_aborts_on_revalidate_collision(simple_stack, fk):
    """A cached waypoint that now collides aborts the detour → reverse."""
    # Build a detour whose waypoints are safe…
    safe_path = [
        [95, 90, 90, 90, 90, 73],
        [100, 90, 90, 90, 90, 73],
    ]
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [92, 90, 90, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 1000.0, 1000.0, 1000.0]
    simple_stack["bb"].sweep_detour_path = list(safe_path)
    simple_stack["bb"].sweep_detour_idx = 0

    # …then inject a point that intersects the very first cached waypoint.
    caps = {c.name: c for c in fk.link_endpoints(safe_path[0])}
    simple_stack["world"].ingest_points(
        np.array([list(caps["gripper"].p1)]),
    )
    replan_before = int(simple_stack["bb"].replan_event)
    _tick(simple_stack["root"])

    assert simple_stack["bb"].sweep_direction == -1
    assert list(simple_stack["bb"].sweep_detour_path) == []
    assert int(simple_stack["bb"].replan_event) > replan_before
    assert str(simple_stack["bb"].last_strategy).startswith(
        "simple_reverse:"
    )


def test_replan_theta_clear_outside_domain_reverses(simple_stack):
    """Obstacle at the 180° edge → θ_clear > 180 → reverse direction."""
    _seed_block_at(simple_stack["world"], theta_deg=179.0)
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [175, 90, 90, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 80.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])

    assert simple_stack["bb"].sweep_direction == -1
    assert list(simple_stack["bb"].sweep_detour_path) == []
    strat = str(simple_stack["bb"].last_strategy)
    assert strat.startswith("simple_reverse:"), (
        f"expected simple_reverse at the edge, got {strat!r}"
    )


def test_replan_detour_z_stays_fixed(simple_stack, fk):
    """User's key constraint: the detour never leaves the sweep z-line.

    We anchor against the world-frame tip-z of a pose that actually sits
    on the sweep's nominal (r, z) slice — i.e. what the IK produces for
    (theta=90, r=SWEEP_R_DEFAULT, z=SWEEP_Z_DEFAULT). Every detour
    waypoint must stay within a few mm of that height.
    """
    from braccio_ctrl.ik_solver import solve_ik, polar_to_cartesian
    from braccio_ctrl.constants import (
        SWEEP_R_DEFAULT_MM, SWEEP_Z_DEFAULT_MM,
    )
    x, y = polar_to_cartesian(90.0, SWEEP_R_DEFAULT_MM)
    sol = solve_ik(x, y, SWEEP_Z_DEFAULT_MM)
    assert sol is not None, "sweep-nominal pose must be reachable"
    start_q = [int(sol.base), int(sol.shoulder), int(sol.elbow),
               int(sol.wrist_vert), 90, 73]

    _seed_block_at(simple_stack["world"], theta_deg=100.0)
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = list(start_q)
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 80.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])

    path = list(simple_stack["bb"].sweep_detour_path)
    assert path, "detour should have been built"
    ref_tip_z = float(fk.link_endpoints(list(start_q))[-1].p1[2])
    # Tolerance covers integer-degree IK rounding, which can produce ~20
    # mm FK tip error at the pull-in midpoint where shoulder/elbow are
    # most bent. Still well under the smallest z-ladder rung spacing, so
    # from the user's perspective the detour stays on the sweep z-line.
    for q in path:
        tip_z = float(fk.link_endpoints(list(q))[-1].p1[2])
        assert abs(tip_z - ref_tip_z) <= 25.0, (
            f"detour waypoint tip z={tip_z:.1f} drifted from sweep-line "
            f"tip z={ref_tip_z:.1f} (tolerance ±25 mm)"
        )


def test_replan_chain_skip_across_small_gap(simple_stack):
    """Two adjacent blocked bands with a narrow gap → chain-skip past both."""
    _seed_block_at(simple_stack["world"], theta_deg=100.0)
    _seed_block_at(simple_stack["world"], theta_deg=130.0)
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 80.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])

    assert simple_stack["bb"].sweep_direction == +1, (
        "chain-skip: direction stays forward"
    )
    target = float(simple_stack["bb"].sweep_target_deg)
    assert target > 130.0, (
        f"θ_clear should skip past the second block; got {target}"
    )


def test_replan_ignores_ch3_ground(simple_stack):
    """CH3 must not trigger a detour or a reverse."""
    from braccio_ctrl.constants import SWEEP_STEP_DEG
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 1000.0, 1000.0, 10.0]
    _tick(simple_stack["root"])
    cmd = simple_stack["bb"].pending_command
    assert cmd is not None
    assert cmd["joints"][0] == int(round(90 + SWEEP_STEP_DEG)), (
        "CH3 is ground-facing and must never gate the sweep"
    )
    assert list(simple_stack["bb"].sweep_detour_path) == []


def test_replan_reverses_at_domain_edge_without_hit(simple_stack):
    """No ToF hit, but next_theta out of domain → flip direction."""
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [178, 90, 90, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 1000.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])
    assert simple_stack["bb"].sweep_direction == -1
    cmd = simple_stack["bb"].pending_command
    assert cmd is not None
    assert cmd["joints"][0] < 178


def test_replan_stale_cloud_falls_back_to_fov_block(simple_stack):
    """ToF fires but the world cloud is empty → assume a default FoV block
    centred on next_theta and still build a detour."""
    from braccio_ctrl.constants import SWEEP_DETOUR_STEPS
    # Empty world cloud — only the ToF gate fires.
    simple_stack["bb"].mode = "sweep"
    simple_stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    simple_stack["bb"].sweep_direction = +1
    simple_stack["bb"].tof_filtered = [1000.0, 80.0, 1000.0, 1000.0]
    _tick(simple_stack["root"])

    path = list(simple_stack["bb"].sweep_detour_path)
    assert len(path) == SWEEP_DETOUR_STEPS, (
        "a detour should still be generated even without world points"
    )
    assert str(simple_stack["bb"].last_strategy).startswith(
        "sweep_detour:"
    )


# ── Sweep branch ───────────────────────────────────────────────────────────

def test_sweep_advances_when_clear(stack):
    stack["bb"].mode = "sweep"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    _tick(stack["root"])
    # Sweep advances by SWEEP_STEP_DEG in +1 direction.
    cmd = stack["bb"].pending_command
    assert cmd is not None
    from braccio_ctrl.constants import SWEEP_STEP_DEG
    expected_theta = int(round(90 + SWEEP_STEP_DEG))
    assert cmd["joints"][0] in (expected_theta - 1, expected_theta, expected_theta + 1)
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


def test_sweep_at_bottom_rung_climbs_past_overhead_obstacle(stack):
    """Regression (session_20260417_020606): the arm sat at the bottom of
    the Z ladder (z_index=4, z=-20) with an obstacle blocking the sweep and
    kept dropping into escape_collision instead of picking a higher rung.
    The obstacle-aware picker should now propose going UP past the obstacle
    rather than walking further down the ladder.
    """
    from braccio_ctrl.safety.behavior import SWEEP_R_DEFAULT_MM, SWEEP_Z_LADDER_MM
    # Block the forward sweep direction so the z-ladder path fires.
    stack["polar"].mark_range(91.0, 180.0)
    # Place an obstacle cluster overhead (z ≈ 20 mm), roughly at the next-θ
    # slice the sweep would hit. Kept short (25 mm span) so it doesn't count
    # as "vertical column".
    r = SWEEP_R_DEFAULT_MM
    theta = 100.0
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    zs = np.linspace(0.0, 25.0, 8)
    cluster = np.array([[x, y, z] for z in zs])
    stack["world"].ingest_points(cluster)

    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].sweep_z_index = 4       # bottom rung (z = -20)
    stack["bb"].sweep_z_tries = 0
    stack["bb"].sweep_z_visited = []

    _tick(stack["root"])

    # Must pick a rung whose z is ABOVE the obstacle envelope (obs_z_max≈25
    # + 20 mm margin = 45), not another one inside / below. Ladder z values
    # > 45: indices 0 (z=35 — borderline), 1 (60), 2 (90).
    new_idx = int(stack["bb"].sweep_z_index)
    new_z = SWEEP_Z_LADDER_MM[new_idx]
    assert new_z > 25.0 + 10.0, (
        f"at bottom rung with overhead obstacle, expected a higher rung; "
        f"got index {new_idx} (z={new_z})"
    )
    assert stack["bb"].last_strategy.startswith("sweep_z_ladder")


def test_sweep_vertical_obstacle_skips_z_ladder(stack):
    """A column of points spanning most of the ladder range can't be
    overtaken by any rung — the picker should refuse (return None) and let
    the sweep fall through to the BiRRT / direction-flip fallback rather
    than climbing rungs that won't help."""
    from braccio_ctrl.safety.behavior import SWEEP_R_DEFAULT_MM
    stack["polar"].mark_range(91.0, 180.0)
    r = SWEEP_R_DEFAULT_MM
    theta = 100.0
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    zs = np.linspace(-50.0, 95.0, 24)    # ~145 mm tall column — > 60% of ladder span (110)
    cluster = np.array([[x, y, z] for z in zs])
    stack["world"].ingest_points(cluster)

    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    stack["bb"].sweep_z_index = 0
    stack["bb"].sweep_z_tries = 0
    stack["bb"].sweep_z_visited = []

    _tick(stack["root"])

    # Strategy MUST NOT be sweep_z_ladder — a tall column can't be escaped
    # by switching rungs, so the BT must escalate past the ladder.
    assert not stack["bb"].last_strategy.startswith("sweep_z_ladder"), (
        f"vertical obstacle should bypass the z-ladder, got "
        f"{stack['bb'].last_strategy!r}"
    )


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


def test_sweep_reacts_to_close_obstacle_when_arm_reconfigured(stack, fk):
    """Regression (session_20260416_230959): user reconfigured the arm
    (wrist tilted, base rotated) and started a sweep. Covering the ToF
    sensor with a finger produced close readings that projected near the
    wrist's real world-frame position — but the old polar query was
    hardcoded to (r=SWEEP_R_DEFAULT_MM, z=SWEEP_Z_DEFAULT_MM) and missed
    them entirely. The sweep kept reporting `sweep_direct` and never
    reacted to the finger.

    After the fix, _refresh_polar_from_world queries at the gripper tip's
    actual (r, z), so the blocked range registers and config_in_collision
    triggers escape_collision.
    """
    reconfigured_q = [142, 90, 90, 37, 90, 73]
    # Place an obstacle point near the gripper tip in world frame — this
    # is where ToF projections land when the wrist is tilted and rotated.
    caps = fk.link_endpoints(reconfigured_q)
    tip = caps[-1].p1
    cloud = np.array([list(tip)] * 16)
    # Nudge points 20 mm outward along the gripper tip direction so
    # they're inside the clearance zone but outside the small self-filter
    # margin.
    stack["world"].ingest_points(cloud)

    stack["bb"].mode = "sweep"
    stack["bb"].sweep_direction = +1
    stack["bb"].current_q = reconfigured_q
    stack["bb"].sweep_z_index = 0
    stack["bb"].sweep_z_tries = 0

    _tick(stack["root"])

    # The BT must NOT silently do sweep_direct when the current config
    # has an obstacle at the wrist. It should detect the collision and
    # fire escape_collision (or a replan fallback).
    assert stack["bb"].last_strategy != "sweep_direct", (
        "sweep_direct in the presence of a near-wrist obstacle regresses "
        "the fix for session_20260416_230959"
    )


def test_sweep_tick_decays_stale_world_points(stack):
    """Regression (session_20260417_012550): stale points in the world cloud
    kept polar_blocked frozen at (0, 48) long after the obstacle left,
    trapping the sweep in a half-arc between θ≈135 and θ≈176.

    Root cause: ``WorldModel.decay()`` was never called anywhere; the only
    purge path ran inside ``ingest_points``, so when ToF readings were all
    "clear" no new points arrived and old ones lived forever. The sweep
    tick must age the cloud every tick so polar refresh sees fresh state.
    """
    import time as _time
    old_ts = _time.time() - 100.0  # well past max_age_s (15 s default)
    stale = np.array([[200.0, 50.0, 60.0], [210.0, 55.0, 65.0]])
    stack["world"].ingest_points(stale, timestamp=old_ts)
    assert stack["world"].point_count() == 2

    stack["bb"].mode = "sweep"
    stack["bb"].current_q = [90, 90, 90, 90, 90, 73]
    _tick(stack["root"])

    assert stack["world"].point_count() == 0, (
        "stale world points must be purged every sweep tick"
    )


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
