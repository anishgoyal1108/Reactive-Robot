from planning.qp_replanner import QPReplanner


def test_qp_replanner_nominal_or_fallback():
    qp = QPReplanner(max_step_deg=5.0, min_clearance_m=0.2)
    res = qp.solve(
        current_joints=[90, 90, 90, 90, 90, 60],
        nominal_joints=[100, 100, 100, 90, 90, 60],
        obstacle_distance_m=0.4,
        speed_scale=0.7,
        time_budget_ms=50.0,
    )
    assert len(res.command) == 6
    assert res.solve_time_ms >= 0.0


def test_qp_replanner_close_obstacle_fallback_hold():
    qp = QPReplanner(max_step_deg=5.0, min_clearance_m=0.2)
    res = qp.solve(
        current_joints=[90, 90, 90, 90, 90, 60],
        nominal_joints=[110, 110, 110, 90, 90, 60],
        obstacle_distance_m=0.05,
        speed_scale=0.5,
        time_budget_ms=50.0,
    )
    assert res.fallback_used
    assert res.command == [90, 90, 90, 90, 90, 60]
