import json
import queue
import time

import numpy as np

from braccio_ctrl.controller import BraccioController, ChannelSample, FSM_DETECT, FSM_PLAN, FSM_PROBE


class FakeMega:
    def __init__(self):
        self.commands = []
        self.response_queue = queue.Queue()

    def send_cmd(self, cmd):
        self.commands.append(cmd.strip())

    def connect(self):
        return True

    def ping(self, timeout=1.5):
        return True

    def close(self):
        pass


class FakeToFBridge:
    def __init__(self):
        self.counts = []

    def set_active_channels(self, count):
        self.counts.append(int(count))

    def is_open(self):
        return True

    def close(self):
        pass


def _controller():
    ctrl = BraccioController("COM_MEGA", teensy_port="COM_TOF")
    ctrl._bridge = FakeMega()
    ctrl._tof_bridge = FakeToFBridge()
    ctrl._project_channel_points = lambda snap, ch, threshold: [np.array([0.11, 0.0, 0.02], dtype=float)]
    return ctrl


def _snap(hit_channels=()):
    now = time.time()
    grids = []
    validity = []
    for ch in range(4):
        value = 150.0 if ch in hit_channels else 600.0
        grids.append(np.full((4, 4), value, dtype=np.float32))
        validity.append(np.ones((4, 4), dtype=np.uint8))
    return {
        "grids": grids,
        "validity": validity,
        "active": [1, 1, 1, 1],
        "last_rx": [now, now, now, now],
        "seq": [1, 1, 1, 1],
        "mcu_ms": [10, 10, 10, 10],
        "status": [0, 0, 0, 0],
        "sensor_id": ["S0", "S1", "S2", "S3"],
        "imu": {"online": False},
    }


def test_ch2_ch3_do_not_initiate_obstacle_events():
    ctrl = _controller()

    assert ctrl._detect_authoritative_hit(_snap(hit_channels={2, 3})) is None


def test_confirmed_obstacle_switches_act_2_to_4_then_resets_to_2():
    ctrl = _controller()
    ctrl._set_tof_active_count(2, reason="test_start")
    hit = ctrl._detect_authoritative_hit(_snap(hit_channels={0}))

    ctrl._begin_confirmation(hit)
    for _ in range(3):
        ctrl._confirm_step(time.monotonic(), _snap(hit_channels={0}))

    assert ctrl._obstacle_memory is not None
    assert ctrl._fsm_mode == FSM_PROBE
    assert ctrl._tof_bridge.counts[:2] == [2, 4]

    ctrl._reset_obstacle_episode("test_done")

    assert ctrl._fsm_mode == FSM_DETECT
    assert ctrl._tof_bridge.counts[-1] == 2


def test_vertical_probe_uses_source_clearance_and_support_channels():
    ctrl = _controller()
    ctrl._set_tof_active_count(2, reason="test_start")
    hit = ctrl._detect_authoritative_hit(_snap(hit_channels={1}))

    ctrl._begin_confirmation(hit)
    for _ in range(3):
        ctrl._confirm_step(time.monotonic(), _snap(hit_channels={1}))

    assert ctrl._fsm_mode == FSM_PROBE
    ctrl._vertical_probe_step(100.0, _snap(hit_channels={1}))
    target_z = ctrl._vertical_probe_target_z
    assert target_z == 0.0

    ctrl._vertical_probe_step(101.0, _snap(hit_channels=set()))

    assert ctrl._fsm_mode == FSM_PLAN
    assert ctrl._vertical_opening_pose is not None
    assert ctrl._vertical_opening_pose[2] == target_z


def test_vertical_probe_targets_sweep_low_to_high_from_floor():
    ctrl = _controller()

    assert ctrl._build_vertical_probe_targets(20.0) == [0.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0]
    assert ctrl._build_vertical_probe_targets(120.0) == [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 140.0, 160.0]
    assert 180.0 not in ctrl._build_vertical_probe_targets(20.0)


def test_vertical_probe_pose_regenerates_from_xy_lock():
    ctrl = _controller()
    ctrl._vertical_probe_xy_lock = ctrl._vertical_xy_lock_from_pose((74.5, 112.0, 20.0))
    ctrl._last_sent_theta = 10.0
    ctrl._last_sent_r = 200.0

    pose = ctrl._vertical_probe_pose_for_z(120.0)

    assert np.allclose(pose, (74.5, 112.0, 120.0))


def test_auto_wrist_offset_opens_tight_high_z_probe_pose_without_accumulating():
    ctrl = _controller()
    ctrl._fsm_mode = FSM_PROBE
    ctrl._auto_wrist_base_offset = 0.0

    high_offset = ctrl._auto_wrist_offset_for_pose(112.0, 112.0, 140.0)
    low_offset = ctrl._auto_wrist_offset_for_pose(112.0, 112.0, 20.0)

    assert high_offset >= 35.0
    assert low_offset == 0.0


def test_vertical_support_uses_filtered_median_not_single_minimum_cell():
    ctrl = _controller()
    snap = _snap()
    snap["grids"][3][0, 0] = 45.0

    support = ctrl._vertical_support_status(snap)

    assert support["ok"]
    assert support["channels"]["tof_ch3"]["min_mm"] == 45.0
    assert support["channels"]["tof_ch3"]["median_mm"] > support["clear_mm"]


def test_obstacle_memory_radius_is_capped_to_physical_sector_scale():
    ctrl = _controller()
    samples = [
        ChannelSample(
            channel=1,
            min_mm=160.0,
            median_near_mm=180.0,
            points_m=[[0.0, 0.0, 0.0], [0.30, 0.0, 0.0], [0.0, 0.28, 0.0]],
            seq=idx,
            rx_wall=time.time(),
        )
        for idx in range(3)
    ]

    memory = ctrl._memory_from_samples(time.monotonic(), samples)

    assert memory.radius_mm <= 80.0
    assert memory.radius_mm >= 35.0


def test_avoidance_completion_finishes_sweep_leg_without_returning_to_replan_origin():
    ctrl = _controller()
    ctrl._profile_side_active = True
    ctrl._profile_target_theta = 5.0
    ctrl._profile_setpoint_theta = 119.5
    ctrl._last_sent_theta = 5.0
    ctrl._last_sent_r = 112.0
    ctrl._last_sent_z = 20.0
    ctrl._committed_plan = []

    ctrl._avoid_step(100.0)

    assert ctrl._fsm_mode == FSM_DETECT
    assert ctrl._profile_setpoint_theta == 5.0
    assert ctrl._profile_target_theta == 175.0
    assert ctrl._profile_hold_until > 100.0


def test_recording_schema_v2_uses_planner_fields():
    ctrl = _controller()
    ctrl._recording_active = True
    ctrl._record_buffer = []
    ctrl._last_planner_debug = {"schema": "braccio_session_v2", "mode": "detect_sweep", "debug": {}}

    ctrl._record_tick(time.monotonic(), _snap())

    row = json.loads(ctrl._record_buffer[-1])
    assert row["schema"] == "braccio_session_v2"
    assert "planner_active" in row
    assert "optimizer_active" not in row
