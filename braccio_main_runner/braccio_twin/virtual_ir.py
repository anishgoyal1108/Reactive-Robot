"""
virtual_ir.py — PyBullet raycast backend for the 4 base-pocket IR sensors.

Each IR sensor is a boolean "something is close" detector. In the real
hardware they are active-LOW binary outputs debounced by firmware; the
count of firing sensors is encoded as a 2-bit severity (0..3) that gets
fed into ToFState.ingest_ir_raw().

    0 = CLEAR   (no sensors firing)
    1 = FAR     (1 sensor firing)
    2 = CLOSE   (2 sensors firing)
    3 = DANGER  (3 or 4 sensors firing)

Sensor positions come from the URDF (ir_left_front_link,
ir_left_back_link, ir_right_front_link, ir_right_back_link). Each one
shoots a short ray along its local +X axis — the URDF sets that axis
to point outward from the side of the base it is mounted on.
"""

from __future__ import annotations

from braccio_ctrl.tof_sensor import ToFState

from .sim_arm import SimArm
from .virtual_tof import _rotate_vec_by_quat


# Order determines which pocket maps to which "side", but the hardware
# treats them symmetrically — only the count matters.
IR_LINKS: list[str] = [
    "ir_left_front_link",
    "ir_left_back_link",
    "ir_right_front_link",
    "ir_right_back_link",
]


# Real IR sensors fire up to ~8 cm. Use the same threshold in the sim.
IR_DETECTION_RANGE_M = 0.08


class VirtualIR:
    """4 boolean pocket sensors encoded into the ToFState IR 2-bit value."""

    def __init__(
        self,
        sim_arm: SimArm,
        tof_state: ToFState,
        detection_range_m: float = IR_DETECTION_RANGE_M,
    ):
        self._arm = sim_arm
        self._state = tof_state
        self._range_m = detection_range_m

    def sample_once(self) -> None:
        firing = 0
        for link_name in IR_LINKS:
            try:
                pose = self._arm.get_link_pose(link_name)
            except KeyError:
                continue
            direction = _rotate_vec_by_quat((1.0, 0.0, 0.0), pose.orientation)
            dist_m = self._arm.raycast(pose.position, direction, self._range_m)
            if dist_m < self._range_m - 1e-4:
                firing += 1

        # Hardware encoding: 0=none, 1=FAR, 2=CLOSE, 3=DANGER (3 or 4)
        if firing >= 3:
            bits = 3
        elif firing == 2:
            bits = 2
        elif firing == 1:
            bits = 1
        else:
            bits = 0

        self._state.ingest_ir_raw(bits)
        self._state.update_obstacle_status()
