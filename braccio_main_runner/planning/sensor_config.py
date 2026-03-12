"""Configuration defaults for ToF mounts and projection assumptions."""

from __future__ import annotations

SENSOR_CONFIG = {
    # CH0/CH1 are active ToF sensors. CH2 is MPU6050 (IMU-only stream).
    "active_channels": [0, 1],
    "supported_channels": [0, 1],
    # Home-pose reference used by host stack:
    # - CH0 points +Y in base XY plane
    # - CH1 points -Y in base XY plane
    # Mounting note: sensors are mounted on the wrist_pitch body midpoint,
    # not exactly at the wrist_pitch joint origin.
    "channels": {
        0: {
            "sensor_id": "tof_ch0",
            "joint_id": "wrist_pitch",
            "origin_m": [-0.018, 0.000, 0.030],
            "axis_dir": [-1.0, 0.0, 0.0],
            "up_hint": [0.0, 0.0, 1.0],
            "fov_x_deg": 63.0,
            "fov_y_deg": 63.0,
        },
        1: {
            "sensor_id": "tof_ch1",
            "joint_id": "wrist_pitch",
            "origin_m": [0.018, 0.000, 0.030],
            "axis_dir": [1.0, 0.0, 0.0],
            "up_hint": [0.0, 0.0, 1.0],
            "fov_x_deg": 63.0,
            "fov_y_deg": 63.0,
        },
    },
    # IMU mount (MPU6050 on Teensy CH2 stream). This is used for host-side
    # orientation fusion and calibration bookkeeping.
    "imu": {
        "mux_channel": 2,
        "joint_id": "wrist_pitch",
        "origin_m": [0.000, 0.000, 0.036],
        "axis_hint": [0.0, 0.0, 1.0],
    },
    "valid_range_mm": [40.0, 3000.0],
    "inflation_m": 0.08,
    "smoothing_alpha": 0.3,
}


def set_active_channel_count(count: int) -> list[int]:
    """Update active ToF channel list to first `count` supported channels."""
    supported = SENSOR_CONFIG["supported_channels"]
    n = max(1, min(int(count), len(supported)))
    SENSOR_CONFIG["active_channels"] = list(supported[:n])
    return list(SENSOR_CONFIG["active_channels"])


def get_active_channels() -> list[int]:
    return list(SENSOR_CONFIG.get("active_channels", []))
