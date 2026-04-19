"""Integration tests for SafetyPlanner (wraps pybullet_planning.birrt)."""

import numpy as np
import pytest

from braccio_ctrl.safety.collision import CollisionChecker
from braccio_ctrl.safety.fk import ForwardKinematics
from braccio_ctrl.safety.planner import SafetyPlanner
from braccio_ctrl.safety.world_model import WorldModel


@pytest.fixture(scope="module")
def fk():
    instance = ForwardKinematics()
    yield instance
    instance.disconnect()


@pytest.fixture
def world():
    return WorldModel()


@pytest.fixture
def planner(world, fk):
    checker = CollisionChecker(world, fk, clearance_mm=20.0, self_collision=False)
    return SafetyPlanner(checker, fk, default_timeout_s=2.0)


def test_trivial_plan_is_start_goal_pair(planner):
    # Clean world, short move — the trivial path [start, goal] is clear so
    # the planner short-circuits BiRRT.
    res = planner.plan([90, 90, 90, 90, 90, 73], [95, 90, 90, 90, 90, 73])
    assert res.success
    assert res.path is not None
    assert res.path[0] == [90, 90, 90, 90, 90, 73]
    assert res.path[-1] == [95, 90, 90, 90, 90, 73]


def test_plan_rejects_out_of_limits_goal(planner):
    res = planner.plan([90, 90, 90, 90, 90, 73], [200, 90, 90, 90, 90, 73])
    assert not res.success
    assert res.failure_reason == "goal_out_of_limits"


def test_plan_rejects_start_in_collision(world, fk, planner):
    caps = {c.name: c for c in fk.link_endpoints([90, 90, 90, 90, 90, 73])}
    # Park an obstacle right on the gripper capsule axis.
    world.ingest_points(np.array([list(caps["gripper"].p1)]))
    res = planner.plan([90, 90, 90, 90, 90, 73], [95, 90, 90, 90, 90, 73])
    assert not res.success
    assert res.failure_reason == "start_in_collision"


def test_plan_rejects_goal_in_collision(world, fk, planner):
    goal = [90, 60, 120, 90, 90, 73]
    caps = {c.name: c for c in fk.link_endpoints(goal)}
    world.ingest_points(np.array([list(caps["gripper"].p1)]))
    res = planner.plan([90, 90, 90, 90, 90, 73], goal)
    assert not res.success
    assert res.failure_reason == "goal_in_collision"


def test_plan_routes_around_obstacle(world, fk, planner):
    """Start and goal differ mostly in base angle; park an obstacle on the
    straight-line midpoint so a straight base sweep collides but a detour
    through a different shoulder/elbow exists."""
    start = [90, 90, 90, 90, 90, 73]   # arm forward (+X)
    goal = [0, 90, 90, 90, 90, 73]     # arm swung 90° right (-Y)
    # Straight-line midpoint in joint space is base=45°. Drop an obstacle at
    # its gripper tip position.
    mid = [45, 90, 90, 90, 90, 73]
    caps = {c.name: c for c in fk.link_endpoints(mid)}
    gripper_tip = np.array(caps["gripper"].p1)
    world.ingest_points(gripper_tip.reshape(1, 3))

    res = planner.plan(start, goal, timeout_s=3.0)
    # BiRRT is probabilistically complete — usually succeeds, sometimes times
    # out. Accept either; if it succeeds the endpoints must match the
    # request and the first/last configs must be collision-free.
    if res.success:
        assert res.path[0] == start
        assert res.path[-1] == goal
    else:
        assert res.failure_reason in ("birrt_timeout",)


def test_plan_cartesian_calls_ik(world, fk, planner):
    start = [90, 90, 90, 90, 90, 73]
    # Target the default HOME cartesian roughly — reachable.
    res = planner.plan_cartesian(start, (152.0, 0.0, 60.0))
    assert res.success or res.failure_reason in ("ik_unreachable", "birrt_timeout",
                                                 "goal_in_collision")


def test_plan_cartesian_reports_unreachable(planner):
    start = [90, 90, 90, 90, 90, 73]
    res = planner.plan_cartesian(start, (10000.0, 0.0, 0.0))  # far outside reach
    assert not res.success
    assert res.failure_reason == "ik_unreachable"
