"""Thread-safe shared state for the arm controller."""

from __future__ import annotations

import threading

from .constants import (
    DEFAULT_R,
    DEFAULT_THETA,
    DEFAULT_Z,
    DELTA_DEFAULT,
    GRIPPER_CLOSE,
    HOME_POS,
)


class ArmState:
    def __init__(self):
        self._lock = threading.RLock()

        self.joints: list[int] = list(HOME_POS)

        self.theta: float = DEFAULT_THETA
        self.r: float = DEFAULT_R
        self.z: float = DEFAULT_Z

        self.wrist_offset: float = 0.0
        self.wrist_rot: int = 90
        self.gripper: int = GRIPPER_CLOSE

        self.delta: int = DELTA_DEFAULT

        self.equil_theta: float = DEFAULT_THETA
        self.equil_r: float = DEFAULT_R
        self.equil_z: float = DEFAULT_Z
        self.equil_wrist_offset: float = 0.0
        self.equil_wrist_rot: int = 90
        self.equil_gripper: int = GRIPPER_CLOSE

        self.connected: bool = False
        self.last_cmd: str = ""
        self.last_resp: str = ""
        self.last_error: str = ""

        self.obstacle_response: str = "clear"
        self.obstacle_source: str = ""
        self.obstacle_dist_mm: float = -1.0
        self.tof_snapshot: dict = {}

        self.control_mode: str = "nominal_tracking"
        self.optimizer_active: bool = False
        self.fallback_active: bool = False
        self.qp_status: str = ""
        self.qp_solve_ms: float = 0.0
        self.predicted_clearance_m: float = -1.0
        self.current_speed_scale: float = 1.0
        self.deadline_miss_count: int = 0
        self.last_latency_ms: float = 0.0

        self.profile_active: bool = False
        self.recording_active: bool = False
        self.record_samples: int = 0
        self.saving_recording: bool = False

        self.obstacle_memory_count: int = 0
        self.obstacle_decay_s: float = 3.0
        self.tof_channels_enabled: int = 0
        self.persistent_memory_active: bool = False
        self.imu_lpf_alpha: float = 0.25
        self.future_plan_points: int = 0

    def update_joints_from_ik(self, ik_result, wrist_offset: float, wrist_rot: int, gripper: int) -> None:
        from .ik_solver import apply_wrist_offset, wrist_level_angle

        with self._lock:
            self.joints[0] = ik_result.base
            self.joints[1] = ik_result.shoulder
            self.joints[2] = ik_result.elbow
            level = wrist_level_angle(ik_result.shoulder, ik_result.elbow)
            self.joints[3] = apply_wrist_offset(level, wrist_offset)
            self.joints[4] = wrist_rot
            self.joints[5] = gripper

    def set_equil_from_current(self) -> None:
        with self._lock:
            self.equil_theta = self.theta
            self.equil_r = self.r
            self.equil_z = self.z
            self.equil_wrist_offset = self.wrist_offset
            self.equil_wrist_rot = self.wrist_rot
            self.equil_gripper = self.gripper

    def restore_equil(self) -> None:
        with self._lock:
            self.theta = self.equil_theta
            self.r = self.equil_r
            self.z = self.equil_z
            self.wrist_offset = self.equil_wrist_offset
            self.wrist_rot = self.equil_wrist_rot
            self.gripper = self.equil_gripper

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "joints": list(self.joints),
                "theta": self.theta,
                "r": self.r,
                "z": self.z,
                "wrist_offset": self.wrist_offset,
                "wrist_rot": self.wrist_rot,
                "gripper": self.gripper,
                "delta": self.delta,
                "equil_theta": self.equil_theta,
                "equil_r": self.equil_r,
                "equil_z": self.equil_z,
                "connected": self.connected,
                "last_cmd": self.last_cmd,
                "last_resp": self.last_resp,
                "last_error": self.last_error,
                "obstacle_response": self.obstacle_response,
                "obstacle_source": self.obstacle_source,
                "obstacle_dist_mm": self.obstacle_dist_mm,
                "tof_snapshot": dict(self.tof_snapshot) if self.tof_snapshot else {},
                "control_mode": self.control_mode,
                "optimizer_active": self.optimizer_active,
                "fallback_active": self.fallback_active,
                "qp_status": self.qp_status,
                "qp_solve_ms": self.qp_solve_ms,
                "predicted_clearance_m": self.predicted_clearance_m,
                "current_speed_scale": self.current_speed_scale,
                "deadline_miss_count": self.deadline_miss_count,
                "last_latency_ms": self.last_latency_ms,
                "profile_active": self.profile_active,
                "recording_active": self.recording_active,
                "record_samples": self.record_samples,
                "saving_recording": self.saving_recording,
                "obstacle_memory_count": self.obstacle_memory_count,
                "obstacle_decay_s": self.obstacle_decay_s,
                "tof_channels_enabled": self.tof_channels_enabled,
                "persistent_memory_active": self.persistent_memory_active,
                "imu_lpf_alpha": self.imu_lpf_alpha,
                "future_plan_points": self.future_plan_points,
            }


