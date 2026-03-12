"""Conservative obstacle model for local replanning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from planning.sensor_config import SENSOR_CONFIG


@dataclass
class ObstacleModel:
    center_m: np.ndarray
    radius_m: float
    points_count: int


class ObstacleModelBuilder:
    def __init__(self):
        self._prev_center: Optional[np.ndarray] = None

    def build(self, points: np.ndarray) -> Optional[ObstacleModel]:
        if points.size == 0:
            return None

        center = np.nanmean(points, axis=0)
        if self._prev_center is not None:
            a = SENSOR_CONFIG["smoothing_alpha"]
            center = (1.0 - a) * self._prev_center + a * center
        self._prev_center = center

        d = np.linalg.norm(points - center[None, :], axis=1)
        radius = float(np.nanpercentile(d, 80)) + float(SENSOR_CONFIG["inflation_m"])
        radius = max(radius, float(SENSOR_CONFIG["inflation_m"]))
        return ObstacleModel(center_m=center, radius_m=radius, points_count=int(points.shape[0]))
