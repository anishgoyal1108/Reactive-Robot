"""Guard-rails for the controller's startup auto-calibration hook.

The hardware path itself (`BraccioController._curses_main`) is interactive,
so we exercise it only up to the `wait_for_frame` + `record_calibration`
contract. The actual end-to-end integration is covered by smoke testing on
real hardware.
"""

import threading
import time

import pytest

from braccio_ctrl.imu_state import IMUState


def test_wait_for_frame_returns_false_when_no_frame_ever_arrives():
    imu = IMUState()
    t0 = time.time()
    assert imu.wait_for_frame(timeout_s=0.15, poll_interval_s=0.02) is False
    # Upper bound: shouldn't hang far past the deadline.
    assert time.time() - t0 < 0.6


def test_wait_for_frame_returns_true_once_update_runs():
    imu = IMUState()

    def feed():
        time.sleep(0.05)
        imu.update(0.0, 0.0, 0.0, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0)

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    assert imu.wait_for_frame(timeout_s=1.0, poll_interval_s=0.02) is True
    t.join(timeout=0.5)


def test_record_calibration_sets_flag_and_offset():
    imu = IMUState()
    imu.update(0.0, 0.0, 42.5, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0)
    assert imu.calibrated is False
    imu.record_calibration()
    assert imu.calibrated is True
    assert imu.yaw_calibration_offset == pytest.approx(42.5)
    # After calibration, yaw_relative measures drift from the reference.
    imu.update(0.0, 0.0, 45.0, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0)
    assert imu.yaw_relative() == pytest.approx(2.5)
