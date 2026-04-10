"""
virtual_imu.py — Derive IMU roll/pitch/yaw from PyBullet link orientation.

The real IMU is mounted on the wrist. Its roll/pitch come from the
accelerometer's gravity reading, and yaw is always 0 (no magnetometer
fusion — documented in CLAUDE.md). We replicate exactly that here by
pulling the world orientation of wrist_roll_link and converting to Euler
angles.

VirtualIMU.sample_once() is safe to call on every tick — IMUState caches
its rotation matrix and skips recomputation when nothing changes.
"""

from __future__ import annotations

import math

from braccio_ctrl.imu_state import IMUState

from .sim_arm import SimArm

IMU_LINK = "wrist_roll_link"


class VirtualIMU:
    def __init__(self, sim_arm: SimArm, imu_state: IMUState):
        self._arm = sim_arm
        self._state = imu_state

    def sample_once(self) -> None:
        try:
            pose = self._arm.get_link_pose(IMU_LINK)
        except KeyError:
            return
        qx, qy, qz, qw = pose.orientation
        roll, pitch, yaw = _quat_to_euler_deg(qx, qy, qz, qw)

        # Derive gravity components in body frame so ax/ay/az look like a
        # stationary MPU-6050 reading (used by downstream code for
        # sanity-check displays). Yaw is intentionally held at 0 to match
        # the real system's no-magnetometer behaviour.
        ax = -math.sin(math.radians(pitch))
        ay = math.sin(math.radians(roll)) * math.cos(math.radians(pitch))
        az = math.cos(math.radians(roll)) * math.cos(math.radians(pitch))

        self._state.update(
            roll=roll,
            pitch=pitch,
            yaw=0.0,
            ax=ax,
            ay=ay,
            az=az,
            mx=0.0,
            my=0.0,
            mz=0.0,
        )


def _quat_to_euler_deg(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[float, float, float]:
    """
    Quaternion (xyzw, PyBullet convention) → ZYX Euler degrees.

    Returns (roll, pitch, yaw) in degrees.
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)  # gimbal-lock clamp
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
