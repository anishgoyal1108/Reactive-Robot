import numpy as np

from planning.obstacle_model import ObstacleModelBuilder
from planning.obstacle_trigger import ObstacleTrigger


def test_obstacle_model_and_trigger():
    builder = ObstacleModelBuilder()
    pts = np.array([[0.2, 0.0, 0.0], [0.22, 0.01, 0.0], [0.21, -0.01, 0.0]])
    model = builder.build(pts)
    assert model is not None
    trig = ObstacleTrigger(replanning_dist_m=0.3, slow_dist_m=0.5)
    out = trig.evaluate(model, "tof_ch0")
    assert out.optimizer_active
    assert out.threat_level in {"high", "medium"}
