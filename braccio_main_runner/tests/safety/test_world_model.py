"""Unit tests for braccio_ctrl.safety.world_model."""

import time

import numpy as np
import pytest

from braccio_ctrl.safety.world_model import WorldModel


@pytest.fixture
def world():
    return WorldModel(grid_cells=20, grid_cell_mm=30.0, max_age_s=10.0)


def test_empty_world_returns_empty_cloud(world):
    assert world.point_count() == 0
    assert world.cloud_points().shape == (0, 3)


def test_ingest_adds_points(world):
    pts = np.array([[100.0, 0.0, 50.0], [150.0, 20.0, 60.0]])
    world.ingest_points(pts)
    assert world.point_count() == 2
    np.testing.assert_allclose(np.sort(world.cloud_points(), axis=0),
                               np.sort(pts, axis=0))


def test_decay_drops_stale_points(world):
    t0 = time.time() - 100.0  # 100 s ago → far past max_age_s=10
    pts = np.array([[100.0, 0.0, 0.0]])
    world.ingest_points(pts, timestamp=t0)
    assert world.point_count() == 1
    dropped = world.decay()
    assert dropped == 1
    assert world.point_count() == 0


def test_ingest_rejects_malformed(world):
    with pytest.raises(ValueError):
        world.ingest_points(np.zeros((3, 2)))  # needs 3 columns


def test_query_radius_finds_nearby(world):
    world.ingest_points(np.array([
        [100.0, 0.0, 0.0],
        [500.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
    ]))
    hits = world.query_radius([0.0, 0.0, 0.0], 150.0)
    # [100,0,0] and [0,100,0] within 150 mm; [500,0,0] far away
    assert hits.shape[0] == 2


def test_query_nearest(world):
    world.ingest_points(np.array([
        [100.0, 0.0, 0.0],
        [500.0, 0.0, 0.0],
    ]))
    dists, pts = world.query_nearest([0.0, 0.0, 0.0], k=1)
    assert len(dists) == 1
    np.testing.assert_allclose(dists[0], 100.0)
    np.testing.assert_allclose(pts[0], [100.0, 0.0, 0.0])


def test_cell_state_bayesian_updates(world):
    # Never-observed cell → unknown
    assert world.cell_state([0.0, 0.0, 0.0]) == "unknown"
    # Ingest a point at that cell — should flip to 'occ' after enough hits
    for _ in range(3):
        world.ingest_points(np.array([[0.0, 0.0, 0.0]]))
    assert world.cell_state([0.0, 0.0, 0.0]) == "occ"


def test_observed_free_ray_drops_cells_to_free(world):
    # Ingest nothing; mark a ray free from origin along +X
    world.mark_observed_free([0.0, 0.0, 0.0], [300.0, 0.0, 0.0])
    # Multiple passes needed to drive log-odds below FREE_THRESHOLD
    for _ in range(2):
        world.mark_observed_free([0.0, 0.0, 0.0], [300.0, 0.0, 0.0])
    # Midpoint along the ray should be 'free'
    assert world.cell_state([150.0, 0.0, 0.0]) == "free"


def test_out_of_bounds_cell_is_unknown(world):
    # Way outside the 600 mm-wide grid → unknown (not raise)
    assert world.cell_state([5000.0, 0.0, 0.0]) == "unknown"


def test_polar_blocked_ranges_merges_overlaps(world):
    # Place two obstacles 15° apart at r=152, z=60 — each generates a 45°
    # band (±22.5°), so they should merge into one.
    r = 152.0
    z = 60.0
    theta_a = 85.0
    theta_b = 95.0
    pts = []
    for th in (theta_a, theta_b):
        x = r * np.cos(np.radians(th))
        y = r * np.sin(np.radians(th))
        pts.append([x, y, z])
    world.ingest_points(np.array(pts))
    ranges = world.polar_blocked_ranges(r, z)
    assert len(ranges) == 1
    lo, hi = ranges[0]
    assert lo < theta_a - 10
    assert hi > theta_b + 10


def test_polar_ignores_far_slices(world):
    # Obstacle at z=200 shouldn't pollute a slice at z=60
    world.ingest_points(np.array([[100.0, 0.0, 200.0]]))
    ranges = world.polar_blocked_ranges(r_mm=152.0, z_mm=60.0)
    assert ranges == []


def test_obstacle_z_extent_captures_tall_column(world):
    # Vertical column at θ=90°, r=152: points stacked along z from -60 → 90.
    r = 152.0
    theta = 90.0
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
    zs = np.linspace(-60.0, 90.0, 16)
    pts = np.array([[x, y, z] for z in zs])
    world.ingest_points(pts)

    extent = world.obstacle_z_extent(r_mm=r, theta_deg=theta)
    assert extent is not None
    z_min, z_max = extent
    assert z_min <= -55.0
    assert z_max >= 85.0


def test_obstacle_z_extent_empty_slice_returns_none(world):
    # Point at theta≈0°; query at theta=180° — opposite side, far outside FoV.
    world.ingest_points(np.array([[152.0, 0.0, 0.0]]))
    assert world.obstacle_z_extent(r_mm=152.0, theta_deg=180.0) is None


def test_clear_wipes_everything(world):
    world.ingest_points(np.array([[100.0, 0.0, 0.0]]))
    world.mark_observed_free([0.0, 0.0, 0.0], [200.0, 0.0, 0.0])
    world.clear()
    assert world.point_count() == 0
    assert world.cell_state([0.0, 0.0, 0.0]) == "unknown"


def test_snapshot_reports_state(world):
    world.ingest_points(np.array([[100.0, 0.0, 0.0], [120.0, 0.0, 0.0]]))
    snap = world.snapshot()
    assert snap["num_points"] == 2
    assert snap["oldest_age_s"] >= 0
    assert snap["grid_cells_occ"] >= 0
