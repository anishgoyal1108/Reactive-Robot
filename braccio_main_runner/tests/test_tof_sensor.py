import numpy as np
import time

from braccio_ctrl.tof_sensor import ObstacleResponse, ToFBridge, ToFState
from sensing.tof_frame import ToFFrameV1


def test_update_obstacle_status_ignores_invalid_negative_ranges():
    state = ToFState(num_channels=2)
    state.active[0] = 1
    state.grids[0][:] = -1.0
    state.validity[0][:] = 1
    state.tof_threshold_mm = 200.0

    state.update_obstacle_status()

    assert state.obstacle_response == ObstacleResponse.CLEAR
    assert state.obstacle_dist_mm == -1.0


def test_apply_frame_freezes_current_imu_snapshot_per_channel():
    state = ToFState(num_channels=2)
    bridge = ToFBridge(state)
    with state._lock:
        state.imu_online = True
        state.imu_last_rx = 12.5
        state.imu_seq = 42
        state.imu_mcu_ms = 314
        state.imu_accel_g[:] = [0.1, 0.2, 0.3]
        state.imu_gyro_dps[:] = [1.0, 2.0, 3.0]
        state.imu_temp_c = 25.0
        state.imu_status = 7
        state.imu_calibrated = True

    frame = ToFFrameV1(
        seq=5,
        mcu_ms=100,
        sensor_id="tof_ch0",
        mux_channel=0,
        joint_id="wrist_pitch",
        status=0,
        distances_mm=[150.0] * 64,
        validity=[1] * 64,
    )
    bridge._apply_frame(frame)

    snap = state.snapshot()
    frame_imu = snap["frame_imu"][0]
    assert frame_imu["online"] is True
    assert frame_imu["seq"] == 42
    assert frame_imu["mcu_ms"] == 314
    assert np.isclose(frame_imu["ax_g"], 0.1)
    assert np.isclose(frame_imu["ay_g"], 0.2)
    assert np.isclose(frame_imu["az_g"], 0.3)


def test_parse_imu_updates_relative_orientation_estimate():
    state = ToFState(num_channels=2)
    bridge = ToFBridge(state)

    bridge._parse_imu("IMU,1,10,0.0,0.0,-1.0,0.0,0.0,0.0,25.0,0")
    bridge._parse_imu("IMU,2,20,0.0,0.0,-1.0,0.0,0.0,90.0,25.0,0")

    snap = state.snapshot()["imu"]
    assert snap["orientation_ready"] is True
    assert len(snap["rot_ref_from_imu"]) == 3
    assert len(snap["rot_ref_from_imu"][0]) == 3
    assert abs(float(snap["euler_zyx_deg"][0])) > 0.1


def test_apply_frame_freezes_orientation_snapshot_per_channel():
    state = ToFState(num_channels=2)
    bridge = ToFBridge(state)

    bridge._parse_imu("IMU,1,10,0.0,0.0,-1.0,0.0,0.0,0.0,25.0,0")
    bridge._parse_imu("IMU,2,20,0.0,0.0,-1.0,0.0,0.0,90.0,25.0,0")

    frame = ToFFrameV1(
        seq=5,
        mcu_ms=100,
        sensor_id="tof_ch0",
        mux_channel=0,
        joint_id="wrist_pitch",
        status=0,
        distances_mm=[150.0] * 64,
        validity=[1] * 64,
    )
    bridge._apply_frame(frame)

    frame_imu = state.snapshot()["frame_imu"][0]
    assert frame_imu["orientation_ready"] is True
    assert len(frame_imu["rot_ref_from_imu"]) == 3
    assert abs(float(frame_imu["euler_zyx_deg"][0])) > 0.1


def test_update_obstacle_status_ignores_stale_channel_frames():
    state = ToFState(num_channels=2)
    state.active[0] = 1
    state.grids[0][:] = 80.0
    state.validity[0][:] = 1
    state.last_rx[0] = time.time() - 1.0
    state.tof_threshold_mm = 200.0

    state.update_obstacle_status()

    assert state.obstacle_response == ObstacleResponse.CLEAR


def test_apply_frame_accepts_hyperion_4x4_grid():
    state = ToFState(num_channels=4)
    bridge = ToFBridge(state)

    frame = ToFFrameV1(
        seq=8,
        mcu_ms=200,
        sensor_id="tof_ch3",
        mux_channel=3,
        joint_id="wrist_pitch",
        status=0,
        distances_mm=[220.0] * 16,
        validity=[1] * 16,
        rows=4,
        cols=4,
    )
    bridge._apply_frame(frame)

    snap = state.snapshot()
    assert snap["grids"][3].shape == (4, 4)
    assert snap["validity"][3].shape == (4, 4)
    assert np.isclose(float(np.nanmin(snap["grids"][3])), 220.0)


def test_update_obstacle_status_respects_per_channel_thresholds():
    state = ToFState(num_channels=4)
    state.active[2] = 1
    state.grids[2] = np.full((4, 4), 170.0, dtype=np.float32)
    state.validity[2] = np.ones((4, 4), dtype=np.uint8)
    state.last_rx[2] = time.time()
    state.tof_threshold_mm = 200.0
    state.tof_threshold_by_ch_mm[2] = 150.0

    state.update_obstacle_status()

    assert state.obstacle_response == ObstacleResponse.CLEAR
