"""Tests for PreScanRoutine pose generation + execution."""

import pytest

from braccio_ctrl.safety.pre_scan import PreScanRoutine


CURRENT = [90, 90, 90, 90, 90, 73]


def test_plan_poses_varies_wrist_only():
    sent = []
    routine = PreScanRoutine(
        send_pose=sent.append,
        read_current_joints=lambda: list(CURRENT),
        wrist_r_offsets=[-20, 0, +20],
        wrist_v_offsets=[0],
    )
    poses = routine.plan_poses()
    # Base/shoulder/elbow/gripper must stay at their original values.
    for p in poses:
        assert p[0] == 90
        assert p[1] == 90
        assert p[2] == 90
        assert p[5] == 73
    # Wrist rotation values should span -20..+20 offsets plus the return pose.
    wrist_r_values = {p[4] for p in poses}
    assert {70, 90, 110}.issubset(wrist_r_values)


def test_plan_poses_clamps_to_joint_limits():
    routine = PreScanRoutine(
        send_pose=lambda _p: None,
        read_current_joints=lambda: [90, 90, 90, 90, 175, 73],
        wrist_r_offsets=[+20, +40],
        wrist_v_offsets=[0],
    )
    poses = routine.plan_poses()
    # +40 from 175 would be 215; must clamp to 180.
    assert max(p[4] for p in poses) == 180


def test_plan_poses_dedupes_identical_poses():
    routine = PreScanRoutine(
        send_pose=lambda _p: None,
        read_current_joints=lambda: list(CURRENT),
        wrist_r_offsets=[0, 0, 0],
        wrist_v_offsets=[0],
    )
    poses = routine.plan_poses()
    # All offsets are zero — dedup should collapse to just the current pose.
    # The routine also appends a "return to current" pose, but that's the
    # same again so the set is size 1.
    assert len(poses) == 1


def test_execute_sends_every_pose_and_sleeps_between():
    sent: list = []
    slept: list = []
    routine = PreScanRoutine(
        send_pose=sent.append,
        read_current_joints=lambda: list(CURRENT),
        settle_s=0.01,
        wrist_r_offsets=[-20, 0, +20],
        wrist_v_offsets=[0],
    )
    routine.execute(sleep_fn=slept.append)
    expected_n = len(routine.plan_poses())
    assert len(sent) == expected_n
    assert len(slept) == expected_n
    assert all(abs(s - 0.01) < 1e-9 for s in slept)


def test_execute_with_base_joints_overrides_reader():
    sent = []
    called = {"reader": False}
    def reader():
        called["reader"] = True
        return [0] * 6
    routine = PreScanRoutine(
        send_pose=sent.append,
        read_current_joints=reader,
        wrist_r_offsets=[0],
        wrist_v_offsets=[0],
        settle_s=0.0,
    )
    routine.execute(base_joints=[90, 90, 90, 90, 90, 73], sleep_fn=lambda _s: None)
    assert not called["reader"]
    # Poses reflect the override
    for pose in sent:
        assert pose[0] == 90
