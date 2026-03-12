import numpy as np

from planning.linear_relaxed_replanner import LinearRelaxedReplanner


def test_linear_relaxed_replanner_obstacle_projection():
    replanner = LinearRelaxedReplanner()

    theta_cmd, r_cmd, z_cmd, debug = replanner.step(
        desired_theta=90.0,
        desired_r=130.0,
        desired_z=20.0,
        prev_theta=85.0,
        prev_r=130.0,
        prev_z=20.0,
        obstacle_response="replan",
        obstacle_dist_mm=120.0,
        obstacle_threshold_mm=300.0,
        dt_s=0.04,
        r_bounds=(10.0, 240.0),
        z_bounds=(-250.0, 200.0),
        obstacle_point_m=np.array([0.13, 0.0, 0.02], dtype=float),
        endpoint_lock=False,
    )

    assert np.isfinite(theta_cmd)
    assert np.isfinite(r_cmd)
    assert np.isfinite(z_cmd)
    assert 10.0 <= r_cmd <= 240.0
    assert -250.0 <= z_cmd <= 200.0
    assert debug["mode"] in {"replan", "clear", "back_away"}


def test_linear_relaxed_replanner_endpoint_lock():
    replanner = LinearRelaxedReplanner()
    theta_cmd, r_cmd, z_cmd, _ = replanner.step(
        desired_theta=180.0,
        desired_r=112.0,
        desired_z=20.0,
        prev_theta=175.0,
        prev_r=90.0,
        prev_z=80.0,
        obstacle_response="replan",
        obstacle_dist_mm=150.0,
        obstacle_threshold_mm=300.0,
        dt_s=0.04,
        r_bounds=(10.0, 240.0),
        z_bounds=(-250.0, 200.0),
        obstacle_point_m=np.array([0.1, 0.0, 0.05], dtype=float),
        endpoint_lock=True,
    )

    assert r_cmd == 112.0
    assert z_cmd == 20.0
    assert 0.0 <= theta_cmd <= 180.0
