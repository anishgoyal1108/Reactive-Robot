import numpy as np

from session_plotter import SessionPlotter


class _FakeKin:
    def joint_transform(self, joints, link_name):
        tf = np.eye(4, dtype=float)
        tf[:3, 3] = [float(joints[0]) / 100.0, float(joints[1]) / 100.0, float(joints[2]) / 100.0]
        return tf


def test_session_plotter_does_not_load_feasible_cloud():
    plotter = SessionPlotter.__new__(SessionPlotter)

    assert plotter._load_feasible_cloud() is None


def test_session_plotter_workspace_limits_keep_minimum_scene_span():
    points = np.array([[0.02, 0.02, 0.02], [0.04, 0.05, 0.03]], dtype=float)

    lo, hi, aspect = SessionPlotter._workspace_limits_for_points(points)

    assert hi[0] - lo[0] >= 0.62
    assert hi[1] - lo[1] >= 0.62
    assert hi[2] - lo[2] >= 0.42
    assert aspect[0] >= 0.62


def test_session_plotter_nearest_tof_handles_missing_channels():
    tof = np.array([[np.nan, 250.0, np.nan], [np.nan, np.nan, np.nan], [800.0, 600.0, 900.0]])

    nearest = SessionPlotter._nearest_tof_m(tof)

    assert np.isclose(nearest[0], 0.25)
    assert np.isnan(nearest[1])
    assert np.isclose(nearest[2], 0.6)


def test_future_plan_points_prefer_joint_transform_over_ik_target_eef():
    plotter = SessionPlotter.__new__(SessionPlotter)
    plotter.kin = _FakeKin()

    xyz = plotter._future_plan_point_m(
        {
            "joints_deg": [10, 20, 30, 40, 50, 60],
            "eef_m": [9.0, 9.0, 9.0],
        }
    )

    assert np.allclose(xyz, [0.1, 0.2, 0.3])


def test_tool_points_prefer_joint_transform_over_recorded_ik_target_eef():
    plotter = SessionPlotter.__new__(SessionPlotter)
    plotter.kin = _FakeKin()
    plotter.records = [
        {
            "joints_deg": [10, 20, 30, 40, 50, 60],
            "eef_m": [0.09, 0.08, 0.07],
        }
    ]

    points = plotter._precompute_tool_points()

    assert np.allclose(points[0], [0.1, 0.2, 0.3])


def test_session_plotter_builds_replan_history_only_for_replan_windows():
    records = [
        {"mode": "detect_sweep", "planner_mode": "detect_sweep", "obstacle": {"response": "clear"}},
        {"mode": "vertical_probe", "planner_mode": "vertical_probe", "obstacle": {"response": "replan"}},
        {"mode": "avoid_execute", "planner_mode": "avoid_execute", "future_plan": [{"eef_m": [0.2, 0.0, 0.1]}]},
        {"mode": "detect_sweep", "planner_mode": "detect_sweep", "obstacle": {"response": "clear"}},
    ]
    tool_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        dtype=float,
    )

    history = SessionPlotter._build_replan_history_points(records, tool_points)

    assert history.shape == (4, 3)
    assert np.isnan(history[0]).all()
    assert np.allclose(history[1], tool_points[1])
    assert np.allclose(history[2], tool_points[2])
    assert np.isnan(history[3]).all()
