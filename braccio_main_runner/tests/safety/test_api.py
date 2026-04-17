"""Tests for SafetyAPI's per-channel self-filter behavior.

CH3 (bottom sensor) sits above the TCA9548A MUX breakout; when the wrist
tilts forward the bottom sensor starts seeing the MUX board / wiring at
~30-80 mm. Without a per-channel filter margin those returns would
accumulate as fake obstacles and drive the BT into permanent escape /
edge-reverse ping-pong (observed in session_20260416_235724.jsonl).
"""

from __future__ import annotations

import numpy as np

from braccio_ctrl.safety.api import SafetyAPI


def _empty_grid() -> np.ndarray:
    return np.full((8, 8), np.nan, dtype=np.float32)


def _close_grid(dist_mm: float = 35.0) -> np.ndarray:
    return np.full((8, 8), dist_mm, dtype=np.float32)


def test_ch3_wider_self_filter_drops_mount_block_returns():
    """CH3's per-channel margin filters mount-block reflections that
    CH2 (same-distance returns) keeps. Uses the exact joint pose from
    the failing session log."""
    tilted_joints = [132, 164, 145, 180, 97, 55]
    grid = _close_grid(35.0)

    api_ch3 = SafetyAPI(send_cmd=lambda q, d: None)
    api_ch3.ingest_tof_frame(
        [_empty_grid(), _empty_grid(), _empty_grid(), grid], tilted_joints,
    )
    ch3_pts = api_ch3.world.snapshot()["num_points"]

    api_ch2 = SafetyAPI(send_cmd=lambda q, d: None)
    api_ch2.ingest_tof_frame(
        [_empty_grid(), _empty_grid(), grid, _empty_grid()], tilted_joints,
    )
    ch2_pts = api_ch2.world.snapshot()["num_points"]

    assert ch3_pts < ch2_pts, (
        f"CH3 must filter more close-range returns than CH2 "
        f"(got {ch3_pts} vs {ch2_pts})"
    )
    assert ch3_pts == 0, (
        "All mount-block reflections on CH3 should be self-filtered at the "
        "tilted wrist pose that reproduces the log"
    )


def test_ch3_still_registers_distant_obstacles():
    """CH3's margin is wide (~90 mm) but finite — a real obstacle beyond
    that distance must still register so the arm reacts to the ground or
    something the user places underneath."""
    api = SafetyAPI(send_cmd=lambda q, d: None)
    # Neutral pose, distant obstacle at the bottom sensor
    far_grid = _close_grid(200.0)   # well past the 90 mm self-filter margin
    api.ingest_tof_frame(
        [_empty_grid(), _empty_grid(), _empty_grid(), far_grid],
        [90, 90, 90, 90, 90, 73],
        thresholds_mm=[100.0, 100.0, 100.0, 300.0],  # let the 200 mm read through
    )
    assert api.world.snapshot()["num_points"] > 0, (
        "CH3 should still see obstacles beyond the self-filter margin"
    )


def test_handle_command_rate_clamps_large_joint_jumps():
    """Regression (session_20260417_014307): BT escape_collision ticks
    produced single-tick joint jumps up to 49°, producing visibly sporadic
    motion. ``_handle_command`` advertises a post-EMA rate clamp but the
    ``RateClamp`` class was never actually applied. After the fix, a big
    target (e.g. full 180° flip) must be clamped to at most
    ``joint_rate_max_deg`` per tick."""
    sent: list[tuple[list[int], int]] = []
    api = SafetyAPI(send_cmd=lambda q, d: sent.append((list(q), d)))
    api.bb.current_q = [90, 90, 90, 90, 90, 73]
    api.joint_ema.reset(initial=np.array([90, 90, 90, 90, 90, 73], dtype=np.float64))

    # Planner-level pending_command asking for a 90° base jump in one tick.
    api._handle_command({"joints": [180, 90, 90, 90, 90, 73], "delta": 3})

    assert sent, "rate clamp must still forward a command"
    out_joints, _ = sent[0]
    # Delta from previous bb.current_q[0]=90 must be clamped to ≤ configured
    # JOINT_RATE_MAX_DEG (default 3°). Allow ±1° integer rounding slack.
    assert abs(out_joints[0] - 90) <= int(api._cfg.joint_rate_max_deg) + 1, (
        f"expected single-tick base jump ≤ {api._cfg.joint_rate_max_deg}° "
        f"from 90, got {out_joints[0]}"
    )


def test_ch0_ch1_ch2_keep_tight_margin():
    """The non-bottom channels keep the default tight margin. A 35 mm
    reading on CH0/CH1/CH2 must not be filtered as self — that was the
    regression from the original 40 mm blanket margin."""
    joints = [90, 90, 90, 90, 90, 73]
    grid = _close_grid(35.0)
    for ch in (0, 1, 2):
        api = SafetyAPI(send_cmd=lambda q, d: None)
        grids = [_empty_grid()] * 4
        grids[ch] = grid
        api.ingest_tof_frame(grids, joints)
        assert api.world.snapshot()["num_points"] > 0, (
            f"CH{ch} should register a 35 mm reading at neutral pose, but "
            f"the self-filter dropped it"
        )
