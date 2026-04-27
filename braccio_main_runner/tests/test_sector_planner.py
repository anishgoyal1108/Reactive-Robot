from planning.sector_planner import ObstacleMemory, SectorWaypointPlanner


def _memory() -> ObstacleMemory:
    return ObstacleMemory(
        obstacle_id=1,
        source_channel=0,
        centroid_m=[0.11, 0.0, 0.02],
        theta_deg=0.0,
        r_mm=110.0,
        z_mm=20.0,
        radius_mm=45.0,
        confidence=0.8,
        created_monotonic=1.0,
        sample_count=3,
        support_channels=[0],
    )


def test_sector_planner_selects_safe_waypoint_candidate():
    planner = SectorWaypointPlanner(safe_clearance_mm=50.0)

    def clearance(pose, obstacle):
        _theta, r_mm, z_mm = pose
        if _theta >= 30.0 or z_mm >= 30.0 or r_mm <= 80.0:
            return 95.0
        return 15.0

    result = planner.select((0.0, 112.0, 20.0), (45.0, 112.0, 20.0), _memory(), clearance)

    assert result.selected is not None
    assert result.selected.accepted
    assert result.selected.clearance_mm >= 50.0


def test_sector_planner_reports_hold_when_all_candidates_blocked():
    planner = SectorWaypointPlanner(safe_clearance_mm=50.0)

    result = planner.select(
        (0.0, 112.0, 20.0),
        (45.0, 112.0, 20.0),
        _memory(),
        lambda _pose, _obstacle: 10.0,
    )

    assert result.selected is None
    assert result.reason == "no_safe_sector_waypoint_candidate"
    assert all(not c.accepted for c in result.candidates)


def test_sector_planner_accepts_vertical_opening_egress():
    planner = SectorWaypointPlanner(safe_clearance_mm=50.0)

    def clearance(pose, obstacle):
        theta, _r_mm, z_mm = pose
        return 80.0 if z_mm >= 60.0 or theta >= 40.0 else -20.0

    result = planner.select(
        (0.0, 112.0, 20.0),
        (45.0, 112.0, 20.0),
        _memory(),
        clearance,
        opening_pose=(0.0, 112.0, 80.0),
    )

    assert result.selected is not None
    assert result.selected.accepted
    assert result.selected.egress_accepted
    assert result.selected.name.startswith("vertical_opening")
