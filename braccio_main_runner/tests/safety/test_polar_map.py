"""Unit tests for PolarObstacleMap."""

import pytest

from braccio_ctrl.safety.polar_map import PolarObstacleMap


def test_new_map_is_empty():
    m = PolarObstacleMap()
    assert m.blocked_ranges() == []
    assert not m.is_blocked(90.0)


def test_mark_blocked_expands_by_fov():
    m = PolarObstacleMap(bins=180, fov_half_deg=22.5)
    m.mark_blocked(90.0)
    # FOV band is [67.5, 112.5]. Inside should be blocked, outside clear.
    assert m.is_blocked(90.0)
    assert m.is_blocked(70.0)
    assert m.is_blocked(110.0)
    assert not m.is_blocked(50.0)
    assert not m.is_blocked(130.0)


def test_mark_range_blocks_inclusive_band():
    m = PolarObstacleMap(bins=180, fov_half_deg=5.0)
    m.mark_range(50.0, 80.0)
    assert m.is_blocked(60.0)
    assert m.is_blocked(79.0)
    assert not m.is_blocked(45.0)
    assert not m.is_blocked(85.0)


def test_blocked_ranges_merges_contiguous_bins():
    m = PolarObstacleMap(bins=180, fov_half_deg=10.0)
    m.mark_blocked(90.0)
    m.mark_blocked(105.0)
    ranges = m.blocked_ranges()
    assert len(ranges) == 1
    lo, hi = ranges[0]
    assert lo < 85.0
    assert hi > 110.0


def test_blocked_ranges_keeps_gaps_apart():
    m = PolarObstacleMap(bins=180, fov_half_deg=5.0)
    m.mark_blocked(30.0)
    m.mark_blocked(150.0)
    ranges = m.blocked_ranges()
    assert len(ranges) == 2


def test_next_unblocked_goes_around_obstacle():
    m = PolarObstacleMap(bins=180, fov_half_deg=22.5)
    m.mark_blocked(90.0)  # blocks ~[67.5, 112.5]
    # Scanning forward from 60° should skip past the blocked region.
    target = m.next_unblocked(60.0, direction=+1, margin_deg=5.0)
    assert target is not None
    assert target > 112.5

    # Scanning backward from 130° should skip backward past the same band.
    target = m.next_unblocked(130.0, direction=-1, margin_deg=5.0)
    assert target is not None
    assert target < 67.5


def test_next_unblocked_returns_none_when_saturated():
    m = PolarObstacleMap(bins=180, fov_half_deg=5.0)
    m.mark_range(0.0, 180.0)
    assert m.next_unblocked(90.0, direction=+1) is None
    assert m.next_unblocked(90.0, direction=-1) is None


def test_decay_clears_stale_bins():
    m = PolarObstacleMap(bins=180, fov_half_deg=22.5, decay_s=5.0)
    m.mark_blocked(90.0, now=100.0)
    # Pretend the clock advanced 100 s.
    cleared = m.decay(now=200.0)
    assert cleared > 0
    assert m.blocked_ranges() == []


def test_rejects_bad_direction():
    m = PolarObstacleMap()
    with pytest.raises(ValueError):
        m.next_unblocked(90.0, direction=0)


def test_clear_resets():
    m = PolarObstacleMap()
    m.mark_blocked(90.0)
    m.clear()
    assert not m.is_blocked(90.0)


def test_blocks_direction_to_edge_detects_one_sided_block():
    """Hand-shaped block spanning [120, 180] must report 'blocks forward
    toward 180' when the sweep is heading right, and 'does not block
    forward toward 0' when the sweep is heading left."""
    m = PolarObstacleMap(bins=180, fov_half_deg=0.0)   # exact-bin marking
    m.mark_range(120.0, 180.0)

    assert m.blocks_direction_to_edge(80.0, +1) is True    # right → hits block
    assert m.blocks_direction_to_edge(80.0, -1) is False   # left → clear


def test_blocks_direction_to_edge_false_when_no_block_ahead():
    """Empty map always returns False in either direction."""
    m = PolarObstacleMap()
    assert m.blocks_direction_to_edge(90.0, +1) is False
    assert m.blocks_direction_to_edge(90.0, -1) is False


def test_blocks_direction_to_edge_rejects_invalid_direction():
    m = PolarObstacleMap()
    with pytest.raises(ValueError):
        m.blocks_direction_to_edge(90.0, 0)
