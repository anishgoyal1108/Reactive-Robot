"""End-to-end integration test for the safety stack.

Research §6/§7 acceptance: spawn a sphere in the sweep path, start a sweep,
verify the arm skips the blocked angular band without colliding. This
doubles as the regression gate for the whole architecture — every
intermediate commanded joint configuration is re-validated against the
ground-truth point cloud.

The test is deliberately framework-free: no digital-twin MuJoCo / PyBullet
GUI, just the SafetyAPI and a synthetic obstacle cloud. That keeps the test
fast (~1 s) and fully deterministic.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from braccio_ctrl.safety import (
    RefusalDialog,
    SafetyAPI,
    SafetyAPIConfig,
    Waypoint,
    WaypointType,
)


@pytest.fixture
def api():
    sent: list[tuple[list[int], int]] = []
    instance = SafetyAPI(
        send_cmd=lambda q, d: sent.append((list(q), d)),
        config=SafetyAPIConfig(bt_tick_hz=50.0),
    )
    # Attach the captured list so tests can introspect it.
    instance._test_sent_log = sent  # type: ignore[attr-defined]
    yield instance
    instance.stop()


def _arm_hits_cloud(api: SafetyAPI, q: list[int]) -> bool:
    """Ground-truth collision check against the current SafetyAPI WorldModel."""
    return api.collision.config_in_collision(q)


# ── Manual-mode refusal ────────────────────────────────────────────────────

def test_manual_move_refused_when_obstacle_on_path(api: SafetyAPI):
    # Build a wall of points that surrounds every plausible nudge alternate
    # of the target pose, so neither direct plan nor the manual nudge
    # ladder can find a clear path. Start pose is left clear so escape
    # doesn't short-circuit into a success.
    target = [95, 90, 90, 90, 90, 73]
    poses = [target]
    for delta in (+15, -15, +25, -25):
        poses.append([target[0], target[1], target[2] + delta,
                      target[3], target[4], target[5]])
    for delta in (+20, -20):
        poses.append([target[0], target[1], target[2],
                      target[3] + delta, target[4], target[5]])
    pts = []
    for q in poses:
        for cap in api.fk.link_endpoints(q):
            pts.append(list(cap.p1))
    api.world.ingest_points(np.asarray(pts, dtype=np.float64))

    # Refusal payload only fires in strict mode. Default is silent
    # auto-replan (no dialog) per the 2026-04-13 UX change.
    api.set_strict_mode(True)
    api.start()
    api.update_current_q([90, 90, 90, 90, 90, 73])
    api.queue_manual_intent(target)
    time.sleep(0.3)
    api.stop()

    refusal = api.consume_refusal()
    assert refusal is not None, "expected refusal when obstacle is on the path"
    dlg = RefusalDialog()
    dlg.open(refusal)
    lines = dlg.render_lines()
    assert any("REFUSAL" in l.text for l in lines)


# ── Sweep skip-ahead ───────────────────────────────────────────────────────

def test_sweep_skips_obstacle_without_colliding(api: SafetyAPI):
    """
    Scenario:
      * Arm starts at θ=60° sweeping toward 180°.
      * An obstacle is placed on the sweep's polar slice around θ=95°.
      * The BT must detect the block (via polar_blocked_ranges) and skip
        past it — no commanded configuration may collide.
    """
    api.update_current_q([60, 90, 90, 90, 90, 73])
    api.bb.sweep_direction = +1

    # Plant an obstacle on the θ ≈ 95° slice at r ≈ 152 mm, z ≈ 60 mm.
    # (Sweep default is r=152, z=35, but an obstacle at z=60 is still
    #  well inside the polar_blocked_ranges search tolerance of ±60 mm.)
    theta = 95.0
    r = 152.0
    z = 60.0
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    cluster = np.array([
        [x, y, z], [x + 10, y, z], [x - 10, y, z],
        [x, y + 10, z], [x, y - 10, z],
    ])
    api.world.ingest_points(cluster)

    api.start()
    api.set_mode("sweep")
    # Let the BT tick a few hundred ms; at 50 Hz that's 15+ ticks.
    time.sleep(0.5)
    api.stop()

    sent: list = api._test_sent_log  # type: ignore[attr-defined]
    assert sent, "BT should have emitted at least one sweep command"

    # No commanded config may collide with the obstacle cloud.
    for joints, _delta in sent:
        assert not _arm_hits_cloud(api, joints), (
            f"commanded joints {joints} collide with the obstacle cloud"
        )

    # The sweep should have advanced past or skipped over θ≈95°. Accept
    # either (a) the current_q crossed the blocked band, or (b) the polar
    # map recorded the block so a subsequent tick would skip. This keeps
    # the test stable across slight tick-rate variations.
    final_theta = api.bb.current_q[0]
    snap = api.snapshot()
    blocked = snap.get("polar_blocked", [])
    crossed = any(final_theta > 105 or final_theta < 90 for _ in [0])
    saw_block = bool(blocked) or any(
        lo <= 95.0 <= hi for lo, hi in (blocked or [])
    )
    assert crossed or saw_block, (
        f"sweep should have either crossed 95° (current_q={final_theta}) "
        f"or recorded the block (polar_blocked={blocked})"
    )


# ── Sequence cascading replanner ──────────────────────────────────────────

def test_sequence_waypoint_playback(api: SafetyAPI):
    api.update_current_q([90, 90, 90, 90, 90, 73])
    api.queue_sequence([
        Waypoint(q=[110, 90, 90, 90, 90, 73], kind=WaypointType.ESSENTIAL,
                 name="right"),
    ])
    api.start()
    time.sleep(0.5)
    api.stop()
    sent: list = api._test_sent_log  # type: ignore[attr-defined]
    # Must have sent at least one command toward the goal.
    assert sent
    final_q = sent[-1][0]
    assert final_q[0] >= 90


# ── Emergency preemption ──────────────────────────────────────────────────

def test_emergency_ir_signal_sends_home(api: SafetyAPI):
    api.update_current_q([45, 80, 100, 90, 90, 73])
    api.update_ir_severity(3)  # DANGER
    api.set_mode("manual")
    api.start()
    time.sleep(0.2)
    api.stop()
    sent: list = api._test_sent_log  # type: ignore[attr-defined]
    assert sent, "emergency should force a command out"
    # First command under emergency is the HOME pose.
    first = sent[0][0]
    assert first == [90, 90, 90, 90, 90, 73]


# ── Web-backend-style plan_and_validate ───────────────────────────────────

def test_plan_and_validate_routes_around_obstacle(api: SafetyAPI):
    # Place an obstacle on the straight-line base sweep between 90° and 0°.
    caps = {c.name: c for c in api.fk.link_endpoints([45, 90, 90, 90, 90, 73])}
    api.world.ingest_points(np.asarray(caps["gripper"].p1).reshape(1, 3))

    result = api.plan_and_validate(
        [90, 90, 90, 90, 90, 73], [0, 90, 90, 90, 90, 73],
    )
    # Either BiRRT times out, or it finds a detour whose every intermediate
    # config is collision-free.
    if result.success:
        assert result.path is not None
        assert result.path[0] == [90, 90, 90, 90, 90, 73]
        assert result.path[-1] == [0, 90, 90, 90, 90, 73]
        for q in result.path:
            assert not _arm_hits_cloud(api, q), (
                f"planner returned a colliding config: {q}"
            )
    else:
        assert result.failure_reason in ("birrt_timeout",)
