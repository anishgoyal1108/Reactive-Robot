"""Obstacle trigger and threat scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from planning.obstacle_model import ObstacleModel


@dataclass
class TriggerResult:
    optimizer_active: bool
    threat_level: str
    nearest_obstacle_m: float
    source_sensor: str
    speed_scale: float


class ObstacleTrigger:
    def __init__(self, replanning_dist_m: float = 0.35, slow_dist_m: float = 0.6):
        self._replan = replanning_dist_m
        self._slow = slow_dist_m

    def evaluate(self, model: Optional[ObstacleModel], source_sensor: str = "") -> TriggerResult:
        if model is None:
            return TriggerResult(False, "clear", float("inf"), source_sensor, 1.0)

        d = float(np.linalg.norm(model.center_m) - model.radius_m)
        if d <= self._replan:
            return TriggerResult(True, "high", d, source_sensor, 0.35)
        if d <= self._slow:
            return TriggerResult(True, "medium", d, source_sensor, 0.65)
        return TriggerResult(False, "low", d, source_sensor, 0.85)


def blend_joint_command(nominal: list[int], safe: list[int], alpha: float) -> list[int]:
    a = max(0.0, min(1.0, alpha))
    out = []
    for n, s in zip(nominal, safe):
        out.append(int(round((1.0 - a) * n + a * s)))
    return out
