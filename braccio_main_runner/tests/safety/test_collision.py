"""Unit tests for braccio_ctrl.safety.collision.

The capsule primitives are tested directly against analytic expectations; the
CollisionChecker is tested end-to-end against a real PyBullet FK load of the
Braccio URDF so the test doubles as a sanity check that the URDF is intact.
"""

import numpy as np
import pytest

from braccio_ctrl.safety.collision import (
    CollisionChecker,
    capsule_capsule_distance,
    capsule_point_cloud_distance,
)
from braccio_ctrl.safety.fk import Capsule, ForwardKinematics, LINK_CAPSULES
from braccio_ctrl.safety.world_model import WorldModel


# ── capsule_point_cloud_distance ───────────────────────────────────────────

class TestCapsulePointCloud:
    def test_empty_cloud_returns_empty(self):
        out = capsule_point_cloud_distance(
            [0, 0, 0], [0, 0, 100], 20.0,
            np.empty((0, 3), dtype=np.float64),
        )
        assert out.shape == (0,)

    def test_point_on_axis_midpoint(self):
        # Capsule from (0,0,0) to (0,0,100), radius 20. Point at (0,0,50) is
        # on the axis → distance = -20 (fully inside).
        pts = np.array([[0.0, 0.0, 50.0]])
        d = capsule_point_cloud_distance([0, 0, 0], [0, 0, 100], 20.0, pts)
        np.testing.assert_allclose(d, [-20.0])

    def test_point_far_away(self):
        pts = np.array([[500.0, 0.0, 0.0]])
        d = capsule_point_cloud_distance([0, 0, 0], [0, 0, 100], 20.0, pts)
        # Closest point on segment is (0,0,0) → raw dist 500 → 500-20 = 480
        np.testing.assert_allclose(d, [480.0])

    def test_beyond_endpoint_uses_endpoint(self):
        # Point at (0,0,200) is past the segment end → closest point is (0,0,100)
        pts = np.array([[0.0, 0.0, 200.0]])
        d = capsule_point_cloud_distance([0, 0, 0], [0, 0, 100], 20.0, pts)
        np.testing.assert_allclose(d, [100.0 - 20.0])

    def test_degenerate_capsule(self):
        # p0 == p1 → sphere behavior
        pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        d = capsule_point_cloud_distance([50, 0, 0], [50, 0, 0], 10.0, pts)
        # dist from (50,0,0) to points: 50 and 50, minus 10 → [40, 40]
        np.testing.assert_allclose(d, [40.0, 40.0])

    def test_vectorised_over_cloud(self):
        pts = np.array([
            [0.0, 0.0, 50.0],   # on axis → -20
            [30.0, 0.0, 50.0],  # 30 mm off → 10
            [100.0, 0.0, 50.0], # 100 mm off → 80
        ])
        d = capsule_point_cloud_distance([0, 0, 0], [0, 0, 100], 20.0, pts)
        np.testing.assert_allclose(d, [-20.0, 10.0, 80.0])


# ── capsule_capsule_distance ───────────────────────────────────────────────

class TestCapsuleCapsule:
    def test_identical_capsules_overlap(self):
        c = Capsule("a", (0, 0, 0), (100, 0, 0), 10.0)
        assert capsule_capsule_distance(c, c) < 0

    def test_parallel_separated(self):
        c0 = Capsule("a", (0, 0, 0), (100, 0, 0), 10.0)
        c1 = Capsule("b", (0, 50, 0), (100, 50, 0), 10.0)
        # Axis separation 50, sum of radii 20 → gap 30
        assert abs(capsule_capsule_distance(c0, c1) - 30.0) < 1e-6

    def test_perpendicular_touching(self):
        c0 = Capsule("a", (-50, 0, 0), (50, 0, 0), 10.0)
        c1 = Capsule("b", (0, -50, 20), (0, 50, 20), 10.0)
        # Axes cross at origin with 20 mm vertical gap, sum radii 20 → gap 0
        assert abs(capsule_capsule_distance(c0, c1)) < 1e-6

    def test_endpoints_only_case(self):
        c0 = Capsule("a", (0, 0, 0), (10, 0, 0), 5.0)
        c1 = Capsule("b", (100, 0, 0), (110, 0, 0), 5.0)
        # Closest points are endpoints — (10,0,0) and (100,0,0) → 90 - 10 = 80
        assert abs(capsule_capsule_distance(c0, c1) - 80.0) < 1e-6


# ── ForwardKinematics smoke test ───────────────────────────────────────────

@pytest.fixture(scope="module")
def fk():
    instance = ForwardKinematics()
    yield instance
    instance.disconnect()


class TestForwardKinematics:
    def test_urdf_loads_and_has_every_capsule_link(self, fk):
        # If the URDF loaded and link_endpoints runs without raising, all link
        # names in LINK_CAPSULES resolved correctly (the constructor checks).
        caps = fk.link_endpoints([90, 90, 90, 90, 90, 73])
        assert len(caps) == len(LINK_CAPSULES)
        names = [c.name for c in caps]
        assert names == [c.name for c in LINK_CAPSULES]

    def test_home_pose_has_expected_reach(self, fk):
        # HOME = [90,90,90,90,90,73]: upper arm up, forearm forward.
        # The gripper capsule should extend forward (+X) from around z≈260 mm.
        caps = {c.name: c for c in fk.link_endpoints([90, 90, 90, 90, 90, 73])}
        gripper = caps["gripper"]
        # Gripper tip should be ~60 mm past the wrist_roll origin along +X.
        # Exact values depend on URDF offsets; sanity-check magnitudes only.
        dx = gripper.p1[0] - gripper.p0[0]
        assert dx > 40.0, f"expected forward-pointing gripper; got dx={dx}"

    def test_base_rotation_moves_gripper(self, fk):
        c0 = {c.name: c for c in fk.link_endpoints([90, 90, 90, 90, 90, 73])}["gripper"]
        c1 = {c.name: c for c in fk.link_endpoints([0, 90, 90, 90, 90, 73])}["gripper"]
        # Gripper world position should differ significantly when base rotates.
        diff = np.linalg.norm(np.array(c0.p1) - np.array(c1.p1))
        assert diff > 50.0


# ── CollisionChecker end-to-end ────────────────────────────────────────────

@pytest.fixture
def world():
    return WorldModel()


@pytest.fixture
def checker(world, fk):
    return CollisionChecker(world, fk, clearance_mm=20.0, self_collision=False)


class TestCollisionChecker:
    def test_clean_world_is_collision_free(self, checker):
        assert checker.config_in_collision([90, 90, 90, 90, 90, 73]) is False

    def test_obstacle_at_gripper_collides(self, world, fk, checker):
        # Put an obstacle right at the home-pose gripper tip.
        caps = {c.name: c for c in fk.link_endpoints([90, 90, 90, 90, 90, 73])}
        gripper_tip = np.array(caps["gripper"].p1)
        world.ingest_points(gripper_tip.reshape(1, 3))
        assert checker.config_in_collision([90, 90, 90, 90, 90, 73]) is True

    def test_obstacle_far_from_arm_is_free(self, world, checker):
        world.ingest_points(np.array([[0.0, 500.0, 0.0]]))
        assert checker.config_in_collision([90, 90, 90, 90, 90, 73]) is False

    def test_path_in_collision_catches_midpoint(self, world, fk, checker):
        # Path from home to a swung-out pose. Place an obstacle midway.
        q_start = [90, 90, 90, 90, 90, 73]
        q_end = [90, 60, 120, 90, 90, 73]  # swing forearm out
        # Obstacle at the forearm midpoint of an intermediate config.
        mid_q = [90, 75, 105, 90, 90, 73]
        caps = {c.name: c for c in fk.link_endpoints(mid_q)}
        forearm_mid = 0.5 * (
            np.array(caps["forearm"].p0) + np.array(caps["forearm"].p1)
        )
        world.ingest_points(forearm_mid.reshape(1, 3))
        assert checker.path_in_collision(q_start, q_end) is True

    def test_collision_fn_adapter(self, checker):
        fn = checker.collision_fn()
        # Should accept np.ndarray or list and return bool
        assert fn([90, 90, 90, 90, 90, 73]) is False
        assert fn(np.array([90, 90, 90, 90, 90, 73])) is False

    def test_nearest_obstacle_diagnostic(self, world, fk, checker):
        caps = {c.name: c for c in fk.link_endpoints([90, 90, 90, 90, 90, 73])}
        gripper_tip = np.array(caps["gripper"].p1)
        world.ingest_points((gripper_tip + np.array([0, 100, 0])).reshape(1, 3))
        result = checker.nearest_obstacle([90, 90, 90, 90, 90, 73])
        assert result is not None
        name, gap, pt = result
        assert name in {c.name for c in LINK_CAPSULES}
        assert gap > 0  # obstacle is outside the capsule
