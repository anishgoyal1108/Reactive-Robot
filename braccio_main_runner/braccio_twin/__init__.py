"""
braccio_twin — PyBullet digital twin of the Braccio V2 arm.

This package provides a simulated Braccio arm, ToF sensors, IR pocket
sensors, IMU, and an obstacle world that together implement the
BraccioBackend / sensor-reader contracts used by braccio_ctrl.

Entry points:

    python -m braccio_twin                    # native sim + curses
    python -m braccio_twin --gui              # with PyBullet GUI window
    python -m braccio_twin --world world.json # with pre-populated obstacles

The top-level imports here are deliberately light so that a user who only
wants ``SimBackend`` does not pay the PyBullet import cost until they
actually call ``SimBackend.connect()``.
"""

from .sim_backend import SimBackend, SimSensorBridge
from .obstacle_world import ObstacleWorld, ObstacleSpec

__all__ = [
    "SimBackend",
    "SimSensorBridge",
    "ObstacleWorld",
    "ObstacleSpec",
]
