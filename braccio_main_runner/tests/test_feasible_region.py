import numpy as np

from planning.feasible_region import FeasibleRegion


def test_feasible_region_membership_and_distance():
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ],
        dtype=float,
    )
    region = FeasibleRegion(pts, membership_tol_m=0.06)

    assert region.contains(np.array([0.02, 0.01, 0.0], dtype=float))
    assert not region.contains(np.array([0.3, 0.3, 0.3], dtype=float))

    d = region.nearest_distance(np.array([0.1, 0.1, 0.1], dtype=float))
    assert d >= 0.0
