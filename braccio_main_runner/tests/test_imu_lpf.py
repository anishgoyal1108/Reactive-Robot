import numpy as np

from braccio_ctrl.tof_sensor import ToFBridge, ToFState


def test_imu_lpf_clamps_alpha_bounds():
    state = ToFState(num_channels=2)
    state.set_imu_lpf_alpha(5.0)
    assert state.imu_lpf_alpha == 1.0

    state.set_imu_lpf_alpha(-1.0)
    assert state.imu_lpf_alpha == 0.02


def test_imu_lpf_applies_exponential_filter():
    state = ToFState(num_channels=2)
    state.set_imu_lpf_alpha(0.5)
    bridge = ToFBridge(state)

    bridge._parse_imu('IMU,1,100,1.0,0.0,0.0,10.0,0.0,0.0,20.0,0')
    np.testing.assert_allclose(state.imu_accel_g, np.array([1.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(state.imu_gyro_dps, np.array([10.0, 0.0, 0.0]), atol=1e-6)
    assert abs(state.imu_temp_c - 20.0) < 1e-6

    bridge._parse_imu('IMU,2,110,0.0,1.0,0.0,0.0,20.0,0.0,30.0,0')
    np.testing.assert_allclose(state.imu_accel_g, np.array([0.5, 0.5, 0.0]), atol=1e-6)
    np.testing.assert_allclose(state.imu_gyro_dps, np.array([5.0, 10.0, 0.0]), atol=1e-6)
    assert abs(state.imu_temp_c - 25.0) < 1e-6
