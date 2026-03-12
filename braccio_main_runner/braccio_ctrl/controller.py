
"""Main control loop integrating keyboard control and obstacle-aware replanning."""

from __future__ import annotations

import curses
import json
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from planning.feasible_region import FeasibleRegionBuilder, FeasibleRegionConfig
from planning.linear_relaxed_replanner import (
    LinearRelaxedReplanner,
    RelaxedLimits,
    RelaxedWeights,
)
from planning.sensor_config import set_active_channel_count
from planning.tof_projection import frame_to_points_base
from planning.urdf_kinematics import BraccioURDFKinematics
from sensing.tof_frame import ToFFrameV1

from .arm_state import ArmState
from .constants import (
    BAUD_RATE,
    CONTROL_LOOP_HZ,
    DELTA_MAX,
    DELTA_MIN,
    DISPLAY_LOOP_HZ,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    JOINT_GRIPPER,
    JOINT_WRIST_ROT,
    OBSTACLE_DECAY_S_DEFAULT,
    IMU_LPF_ALPHA_DEFAULT,
    PERSISTENT_MEMORY_DEFAULT,
    PERSISTENT_CLEAR_REQUIRED,
    PLANNING_HORIZON_S,
    PLANNING_PREVIEW_DT_S,
    PROFILE_SWEEP_DWELL_S,
    PROFILE_SWEEP_GRIPPER_DEG,
    PROFILE_SWEEP_R_MM,
    PROFILE_SWEEP_STEP_DEG,
    PROFILE_SWEEP_WRIST_OFFSET_DEG,
    PROFILE_SWEEP_WRIST_ROT_DEG,
    PROFILE_SWEEP_Z_MM,
    R_MAX,
    R_MIN,
    R_STEP,
    THETA_STEP,
    TOF_BAUD_RATE,
    TOF_CHANNELS_DEFAULT,
    TOF_CHANNELS_MAX,
    TOF_FRESHNESS_S,
    TOF_THRESHOLD_MM,
    WRIST_R_STEP,
    WRIST_V_STEP,
    Z_MAX,
    Z_MIN,
    Z_STEP,
)
from .display import CursesDisplay
from .keyboard_handler import KeyboardHandler
from .protocol import (
    cmd_get_pos,
    cmd_home,
    cmd_set_all,
    cmd_set_delta,
    cmd_set_ik_polar,
    cmd_set_joint,
)
from .serial_bridge import SerialBridge
from .state_library import StateLibrary
from .ik_solver import apply_wrist_offset, polar_to_cartesian, solve_ik
from .tof_sensor import ObstacleResponse, ToFBridge, ToFState



def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return v / n


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def _rot_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n <= 1e-12 or abs(angle_rad) <= 1e-12:
        return np.eye(3, dtype=float)
    u = axis / n
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    k = _skew(u)
    return np.eye(3, dtype=float) + s * k + (1.0 - c) * (k @ k)
class BraccioController:
    """Top-level controller."""

    def __init__(
        self,
        port: str,
        baud: int = BAUD_RATE,
        teensy_port: str | None = None,
        teensy_baud: int = TOF_BAUD_RATE,
        tof_channels_enabled: int = TOF_CHANNELS_DEFAULT,
        imu_lpf_alpha: float = IMU_LPF_ALPHA_DEFAULT,
    ):
        self._bridge = SerialBridge(port, baud)
        self._state = ArmState()
        self._state_lib = StateLibrary()
        self._plotter = None
        self._tof_plotter = None
        self._stdscr = None
        self._kb = None

        self._tof_channel_count = max(1, min(int(tof_channels_enabled), TOF_CHANNELS_MAX))
        set_active_channel_count(self._tof_channel_count)

        self._tof_state = ToFState(num_channels=self._tof_channel_count)
        self._tof_state.tof_threshold_mm = TOF_THRESHOLD_MM
        self._tof_state.set_imu_lpf_alpha(float(imu_lpf_alpha))
        self._tof_bridge = ToFBridge(self._tof_state)
        self._teensy_port = teensy_port
        self._teensy_baud = teensy_baud

        self._last_sent_theta = self._state.theta
        self._last_sent_r = self._state.r
        self._last_sent_z = self._state.z
        self._last_send_monotonic = time.monotonic()

        self._control_period_s = 1.0 / max(1.0, CONTROL_LOOP_HZ)
        self._display_period_s = 1.0 / max(1.0, DISPLAY_LOOP_HZ)

        self._profile_side_active = False
        self._profile_target_theta: float | None = None
        self._profile_setpoint_theta = float(self._state.theta)
        self._profile_hold_until = 0.0

        self._recording_active = False
        self._record_buffer: list[str] = []
        self._record_start_wall = 0.0
        self._save_thread: threading.Thread | None = None
        self._save_lock = threading.Lock()

        self._obstacle_decay_s = OBSTACLE_DECAY_S_DEFAULT
        self._persistent_memory_active = bool(PERSISTENT_MEMORY_DEFAULT)
        self._persistent_clear_required = int(PERSISTENT_CLEAR_REQUIRED)
        self._memory_voxel_m = 0.03
        self._memory_match_radius_m = 0.08
        self._obstacle_memory: list[dict] = []

        self._weights = RelaxedWeights(
            w_track_theta=3.0,
            w_smooth_theta=1.5,
            w_track_r=2.2,
            w_smooth_r=2.0,
            w_avoid_r=4.0,
            w_track_z=2.2,
            w_smooth_z=2.0,
            w_avoid_z=4.0,
        )
        self._limits = RelaxedLimits(
            theta_vel_nominal_deg_s=55.0,
            theta_vel_avoid_deg_s=22.0,
            r_vel_nominal_mm_s=120.0,
            r_vel_avoid_mm_s=55.0,
            z_vel_nominal_mm_s=110.0,
            z_vel_avoid_mm_s=50.0,
            replan_retreat_mm=28.0,
            replan_lift_mm=55.0,
            backaway_retreat_mm=38.0,
            backaway_lift_mm=70.0,
        )
        self._replanner = LinearRelaxedReplanner(weights=self._weights, limits=self._limits)

        self._future_plan_horizon_s = float(PLANNING_HORIZON_S)
        self._future_plan_dt_s = float(PLANNING_PREVIEW_DT_S)
        self._future_plan_preview: list[dict] = []

        self._workspace_dir = Path(__file__).resolve().parents[1]
        self._repo_root = Path(__file__).resolve().parents[2]
        self._feasible_region_dir = self._workspace_dir / "sensing" / "feasible_regions"
        self._feasible_region = None
        self._kinematics = None

        self._imu_cal_lock = threading.RLock()
        self._imu_calibrating = False
        self._imu_calibrated = False
        self._imu_cal_rms_deg = 0.0
        self._imu_rot_wrist_from_imu = np.eye(3, dtype=float)
        self._imu_calib_thread: threading.Thread | None = None
        self._imu_fuse_weight = 0.8
        self._imu_max_tilt_correction_deg = 35.0

        self._init_feasible_region()
        self._tof_state.set_imu_calibrated(False)

    def _init_feasible_region(self) -> None:
        urdf_path = self._repo_root / "Tinkerkit_model" / "tinkerkit.urdf"
        if not urdf_path.exists():
            self._feasible_region = None
            self._kinematics = None
            return

        try:
            self._kinematics = BraccioURDFKinematics(urdf_path)
        except Exception:
            self._kinematics = None

        try:
            builder = FeasibleRegionBuilder(urdf_path=urdf_path, output_dir=self._feasible_region_dir)
            self._feasible_region = builder.load_or_build(
                FeasibleRegionConfig(sample_count=30000, seed=7, membership_tol_m=0.07)
            )
        except Exception:
            self._feasible_region = None

    def attach_plotter(self, plotter) -> None:
        self._plotter = plotter

    def attach_tof_plotter(self, tof_plotter) -> None:
        self._tof_plotter = tof_plotter

    @property
    def tof_state(self) -> ToFState:
        return self._tof_state

    def run(self) -> None:
        curses.wrapper(self._curses_main)

    def _curses_main(self, stdscr) -> None:
        self._stdscr = stdscr
        self._kb = KeyboardHandler(stdscr)
        display = CursesDisplay(stdscr)

        self._connect()

        if self._teensy_port:
            ok = self._tof_bridge.connect(self._teensy_port, self._teensy_baud)
            if not ok:
                with self._state._lock:
                    self._state.last_error = f"ToF Teensy: cannot open {self._teensy_port}"
            else:
                self._tof_bridge.set_active_channels(self._tof_channel_count)

        if self._plotter is not None:
            self._plotter.start()
        if self._tof_plotter is not None:
            self._tof_plotter.start()

        next_control = time.monotonic()
        next_display = next_control

        try:
            while True:
                now = time.monotonic()

                action = self._kb.get_action()
                if action == "quit":
                    break
                if action is not None:
                    self._handle_action(action)

                if now >= next_control:
                    while now >= next_control:
                        next_control += self._control_period_s
                    self._control_step(now)

                if now >= next_display:
                    while now >= next_display:
                        next_display += self._display_period_s
                    display.render(self._state.snapshot())
                    if self._plotter is not None:
                        self._plotter.pump()
                    if self._tof_plotter is not None:
                        self._tof_plotter.pump()

                sleep_until = min(next_control, next_display)
                sleep_for = sleep_until - time.monotonic()
                if sleep_for > 0.001:
                    time.sleep(min(0.01, sleep_for))
        finally:
            if self._recording_active:
                self._finish_recording_and_home(async_save=False)
            self._wait_for_save_completion(timeout_s=4.0)

            with self._imu_cal_lock:
                t_imu = self._imu_calib_thread
            if t_imu is not None and t_imu.is_alive():
                t_imu.join(timeout=2.0)

            self._bridge.close()
            self._tof_bridge.close()
            if self._plotter is not None:
                self._plotter.stop()
            if self._tof_plotter is not None:
                self._tof_plotter.stop()

    def _wait_for_save_completion(self, timeout_s: float = 0.0) -> None:
        with self._save_lock:
            t = self._save_thread
        if t is None:
            return
        t.join(timeout=timeout_s)

    def _control_step(self, now: float) -> None:
        self._drain_responses()
        self._check_obstacle()

        tof_snap = self._tof_state.snapshot()
        self._update_obstacle_memory(now, tof_snap)
        self._run_side_profile_step(now)

        with self._state._lock:
            self._state.tof_snapshot = tof_snap
            self._state.profile_active = self._profile_side_active
            self._state.recording_active = self._recording_active
            self._state.obstacle_memory_count = len(self._obstacle_memory)
            self._state.obstacle_decay_s = self._obstacle_decay_s
            self._state.tof_channels_enabled = self._tof_channel_count
            self._state.persistent_memory_active = bool(self._persistent_memory_active)
            self._state.imu_lpf_alpha = float(tof_snap.get("imu", {}).get("lpf_alpha", 0.25))
            self._state.future_plan_points = int(len(self._future_plan_preview))

        self._record_tick(now, tof_snap)

    def _connect(self) -> None:
        ok = self._bridge.connect()
        if ok:
            pong = self._bridge.ping()
            with self._state._lock:
                self._state.connected = pong
                if pong:
                    self._state.last_resp = "PONG"
                    self._state.last_error = ""
                else:
                    self._state.last_error = f"No response from {self._bridge._port}"
            if pong:
                self._bridge.send_cmd(cmd_get_pos())
        else:
            err_msg = self._bridge.connect_error
            with self._state._lock:
                self._state.connected = False
                self._state.last_error = f"Cannot open {self._bridge._port}: {err_msg}"

    def _check_obstacle(self) -> None:
        snap = self._tof_state.snapshot()
        response = snap["obstacle_response"]
        source = snap["obstacle_source"]
        dist = snap["obstacle_dist_mm"]

        with self._state._lock:
            self._state.obstacle_response = response
            self._state.obstacle_source = source
            self._state.obstacle_dist_mm = dist

            if response == ObstacleResponse.BACK_AWAY:
                self._state.last_error = f"IR DANGER - BACK AWAY! (src={source})"
            elif response == ObstacleResponse.REPLAN:
                self._state.last_error = f"ToF obstacle {dist:.0f} mm (src={source}) - REPLAN"
            else:
                if ("BACK AWAY" in self._state.last_error) or ("REPLAN" in self._state.last_error):
                    self._state.last_error = ""

    def _drain_responses(self) -> None:
        while True:
            try:
                resp = self._bridge.response_queue.get_nowait()
            except queue.Empty:
                break

            rtype = resp.get("type", "unknown")
            with self._state._lock:
                if rtype in ("pos", "ok_all", "ok_ik"):
                    if "positions" in resp and len(resp["positions"]) >= 6:
                        self._state.joints = list(resp["positions"][:6])
                    self._state.last_resp = resp.get("raw", rtype)
                    self._state.last_error = ""
                elif rtype == "error":
                    self._state.last_error = resp.get("message", "")
                    self._state.last_resp = resp.get("raw", "ERR")
                elif rtype == "ready":
                    self._state.connected = True
                    self._state.last_resp = "READY"
                    self._state.last_error = ""
                elif rtype in ("ok_joint", "ok_home", "ok_delta", "pong", "ack", "stat"):
                    self._state.last_resp = resp.get("raw", rtype)
                elif rtype == "unknown":
                    self._state.last_resp = resp.get("raw", "")

    def _handle_action(self, action: str) -> None:
        if action == "states_menu":
            self._open_states_menu()
            return
        if action == "seq_editor":
            self._open_seq_editor()
            return
        if action == "tuning_menu":
            self._open_tuning_menu()
            return

        if action == "plot_main_toggle":
            if self._plotter is not None:
                self._plotter.toggle_main()
            return
        if action.startswith("plot_joint_") and action.endswith("_toggle"):
            if self._plotter is not None:
                idx = int(action[len("plot_joint_")]) - 1
                self._plotter.toggle_joint(idx)
            return
        if action == "plot_reset":
            if self._plotter is not None:
                self._plotter.reset()
            return
        if action == "plot_screenshot":
            if self._plotter is not None:
                self._plotter.save_screenshot()
            return

        if action == "tof_view_toggle":
            if self._tof_plotter is not None:
                self._tof_plotter.toggle_main()
            return
        if action == "tof_screenshot":
            if self._tof_plotter is not None:
                self._tof_plotter.save_screenshot()
            return
        if action == "tof_threshold_inc":
            with self._tof_state._lock:
                self._tof_state.tof_threshold_mm = min(3000.0, self._tof_state.tof_threshold_mm + 50.0)
            return
        if action == "tof_threshold_dec":
            with self._tof_state._lock:
                self._tof_state.tof_threshold_mm = max(50.0, self._tof_state.tof_threshold_mm - 50.0)
            return
        if action in ("tof_ch1_toggle", "tof_ch2_toggle", "tof_ch3_toggle", "tof_ch4_toggle"):
            if self._tof_plotter is not None:
                ch = int(action[6]) - 1
                self._tof_plotter.toggle_channel(ch)
            return

        if action == "profile_side_toggle":
            self._toggle_side_profile()
            return
        if action == "record_toggle":
            self._toggle_recording()
            return
        if action == "imu_calibrate":
            self._begin_imu_calibration()
            return

        state = self._state
        ik_dirty = False
        send_fn = None

        with self._imu_cal_lock:
            if self._imu_calibrating:
                with state._lock:
                    state.last_resp = "IMU calibration running"
                return

        manual_override_actions = {
            "theta_inc",
            "theta_dec",
            "r_inc",
            "r_dec",
            "z_inc",
            "z_dec",
            "wv_inc",
            "wv_dec",
            "wr_inc",
            "wr_dec",
            "grip_open",
            "grip_close",
            "go_home",
            "set_equil",
        }
        if self._profile_side_active and action in manual_override_actions:
            self._disable_side_profile("manual override")

        with state._lock:
            if action == "theta_inc":
                state.theta = min(180.0, state.theta + THETA_STEP)
                ik_dirty = True
            elif action == "theta_dec":
                state.theta = max(0.0, state.theta - THETA_STEP)
                ik_dirty = True
            elif action == "r_inc":
                state.r = min(R_MAX, state.r + R_STEP)
                ik_dirty = True
            elif action == "r_dec":
                state.r = max(R_MIN, state.r - R_STEP)
                ik_dirty = True
            elif action == "z_inc":
                state.z = min(Z_MAX, state.z + Z_STEP)
                ik_dirty = True
            elif action == "z_dec":
                state.z = max(Z_MIN, state.z - Z_STEP)
                ik_dirty = True
            elif action == "wv_inc":
                state.wrist_offset = min(90.0, state.wrist_offset + WRIST_V_STEP)
                ik_dirty = True
            elif action == "wv_dec":
                state.wrist_offset = max(-90.0, state.wrist_offset - WRIST_V_STEP)
                ik_dirty = True
            elif action == "wr_inc":
                state.wrist_rot = min(180, state.wrist_rot + int(WRIST_R_STEP))
                send_fn = self._send_wrist_rot
            elif action == "wr_dec":
                state.wrist_rot = max(0, state.wrist_rot - int(WRIST_R_STEP))
                send_fn = self._send_wrist_rot
            elif action == "grip_open":
                state.gripper = GRIPPER_OPEN
                send_fn = self._send_gripper
            elif action == "grip_close":
                state.gripper = GRIPPER_CLOSE
                send_fn = self._send_gripper
            elif action == "delta_inc":
                state.delta = min(DELTA_MAX, state.delta + 1)
                send_fn = self._send_delta
            elif action == "delta_dec":
                state.delta = max(DELTA_MIN, state.delta - 1)
                send_fn = self._send_delta
            elif action == "go_home":
                send_fn = self._send_home
            elif action == "set_equil":
                state.set_equil_from_current()
                state.last_cmd = "(equil saved)"
                state.last_resp = ""
                return

        if ik_dirty:
            self._send_ik_move()
        elif send_fn is not None:
            send_fn()

    def _send_ik_move(self, endpoint_lock: bool = False) -> None:
        state = self._state
        with state._lock:
            theta = float(state.theta)
            r = float(state.r)
            z = float(state.z)
            wrist_offset = float(state.wrist_offset)
            wrist_rot = int(state.wrist_rot)
            gripper = int(state.gripper)
            joints = [float(x) for x in state.joints]
            obs = state.obstacle_response
            obs_dist = float(state.obstacle_dist_mm)
            connected = bool(state.connected)

        thresh = float(self._tof_state.tof_threshold_mm)
        now = time.monotonic()
        dt_s = max(1e-3, now - self._last_send_monotonic)

        fresh_ok = True
        if self._teensy_port and self._tof_bridge.is_open():
            freshness = self._tof_state.freshness_by_channel(max_age_s=TOF_FRESHNESS_S)
            fresh_ok = any(freshness.values())

        mem_dist_mm, mem_source, mem_point = self._memory_nearest(joints)
        if obs == ObstacleResponse.REPLAN and not np.isfinite(mem_dist_mm):
            obs = ObstacleResponse.CLEAR
            obs_dist = -1.0
        if obs == ObstacleResponse.CLEAR and np.isfinite(mem_dist_mm) and mem_dist_mm <= thresh:
            obs = ObstacleResponse.REPLAN
            obs_dist = mem_dist_mm

        if not fresh_ok:
            theta_cmd = float(self._last_sent_theta)
            r_cmd = float(self._last_sent_r)
            z_cmd = float(self._last_sent_z)
            debug = {
                "mode": "stale_tof_hold",
                "severity": 1.0,
                "speed_scale": 0.0,
            }
            mode = "fallback_hold"
            fallback_active = True
            optimizer_active = False
            self._future_plan_preview = []
        else:
            plan = self._plan_future_trajectory(
                desired_theta=theta,
                desired_r=r,
                desired_z=z,
                start_theta=float(self._last_sent_theta),
                start_r=float(self._last_sent_r),
                start_z=float(self._last_sent_z),
                wrist_offset=wrist_offset,
                wrist_rot=wrist_rot,
                gripper=gripper,
                base_obstacle_response=obs,
                obstacle_threshold_mm=thresh,
                dt_first_s=dt_s,
                endpoint_lock=endpoint_lock,
            )

            if plan:
                first = plan[0]
                theta_cmd = float(first["theta"])
                r_cmd = float(first["r_mm"])
                z_cmd = float(first["z_mm"])
                debug = dict(first.get("debug", {}))
                self._future_plan_preview = plan
            else:
                theta_cmd, r_cmd, z_cmd, debug = self._replanner.step(
                    desired_theta=theta,
                    desired_r=r,
                    desired_z=z,
                    prev_theta=float(self._last_sent_theta),
                    prev_r=float(self._last_sent_r),
                    prev_z=float(self._last_sent_z),
                    obstacle_response=obs,
                    obstacle_dist_mm=obs_dist,
                    obstacle_threshold_mm=thresh,
                    dt_s=dt_s,
                    r_bounds=(R_MIN, R_MAX),
                    z_bounds=(Z_MIN, Z_MAX),
                    obstacle_point_m=mem_point,
                    endpoint_lock=endpoint_lock,
                )
                self._future_plan_preview = []

            fallback_active = False
            optimizer_active = bool(debug.get("severity", 0.0) > 0.0)
            if obs in (ObstacleResponse.REPLAN, ObstacleResponse.BACK_AWAY):
                mode = "obstacle_aware_tracking"
            else:
                mode = "nominal_tracking"

        if not connected:
            mode = "comms_fault"
            fallback_active = True

        cmd = cmd_set_ik_polar(theta_cmd, r_cmd, z_cmd, wrist_offset, wrist_rot, gripper)
        with state._lock:
            state.last_cmd = cmd.strip()
            state.last_error = ""
            state.control_mode = mode
            state.optimizer_active = optimizer_active
            state.fallback_active = fallback_active
            state.qp_status = f"linear_relaxed:{debug.get('mode', 'unknown')}"
            state.qp_solve_ms = 0.0
            state.current_speed_scale = float(debug.get("speed_scale", 1.0))
            pred_mm = obs_dist
            if (not np.isfinite(pred_mm) or pred_mm < 0.0) and np.isfinite(mem_dist_mm):
                pred_mm = mem_dist_mm
            state.predicted_clearance_m = max(0.0, pred_mm) / 1000.0 if pred_mm >= 0.0 else -1.0
            if obs == ObstacleResponse.REPLAN and mem_source:
                state.obstacle_source = mem_source

        self._bridge.send_cmd(cmd)

        self._last_sent_theta = float(theta_cmd)
        self._last_sent_r = float(r_cmd)
        self._last_sent_z = float(z_cmd)
        self._last_send_monotonic = now

    def _run_side_profile_step(self, now: float) -> None:
        if not self._profile_side_active:
            return
        if now < self._profile_hold_until:
            return
        if self._profile_target_theta is None:
            return

        target = float(self._profile_target_theta)
        delta = target - self._profile_setpoint_theta
        if abs(delta) > 1e-9:
            step = math.copysign(min(abs(delta), PROFILE_SWEEP_STEP_DEG), delta)
            self._profile_setpoint_theta += step

        reached_setpoint = abs(self._profile_setpoint_theta - target) <= 1e-6

        with self._state._lock:
            self._state.theta = self._profile_setpoint_theta
            self._state.r = PROFILE_SWEEP_R_MM
            self._state.z = PROFILE_SWEEP_Z_MM
            self._state.wrist_offset = PROFILE_SWEEP_WRIST_OFFSET_DEG
            self._state.wrist_rot = PROFILE_SWEEP_WRIST_ROT_DEG
            self._state.gripper = PROFILE_SWEEP_GRIPPER_DEG

        self._send_ik_move(endpoint_lock=reached_setpoint)

        if reached_setpoint:
            if abs(self._last_sent_theta - target) <= 1.0:
                self._profile_hold_until = now + PROFILE_SWEEP_DWELL_S
                self._profile_target_theta = 0.0 if target >= 90.0 else 180.0

    def _toggle_side_profile(self) -> None:
        if self._profile_side_active:
            self._disable_side_profile("profile disabled")
            return

        with self._state._lock:
            self._state.r = PROFILE_SWEEP_R_MM
            self._state.z = PROFILE_SWEEP_Z_MM
            self._state.wrist_offset = PROFILE_SWEEP_WRIST_OFFSET_DEG
            self._state.wrist_rot = PROFILE_SWEEP_WRIST_ROT_DEG
            self._state.gripper = PROFILE_SWEEP_GRIPPER_DEG
            self._state.theta = max(0.0, min(180.0, self._state.theta))
            self._state.last_resp = "Side profile: ON"
            self._state.last_error = ""
            start_theta = float(self._state.theta)

        self._profile_setpoint_theta = start_theta
        self._profile_target_theta = 180.0 if start_theta < 90.0 else 0.0
        self._profile_side_active = True
        self._profile_hold_until = 0.0
        self._send_ik_move(endpoint_lock=False)

    def _disable_side_profile(self, reason: str) -> None:
        self._profile_side_active = False
        self._profile_hold_until = 0.0
        self._profile_target_theta = None
        with self._state._lock:
            self._state.last_resp = f"Side profile: OFF ({reason})"

    def _toggle_recording(self) -> None:
        if self._recording_active:
            self._finish_recording_and_home(async_save=True)
            return

        with self._save_lock:
            if self._save_thread is not None and self._save_thread.is_alive():
                with self._state._lock:
                    self._state.last_error = "Previous recording is still being saved"
                return

        self._record_buffer.clear()
        self._recording_active = True
        self._record_start_wall = time.time()
        with self._state._lock:
            self._state.recording_active = True
            self._state.record_samples = 0
            self._state.saving_recording = False
            self._state.last_resp = "Recording started"
            self._state.last_error = ""

    def _finish_recording_and_home(self, async_save: bool) -> None:
        if not self._recording_active:
            return

        self._recording_active = False
        self._disable_side_profile("recording stop")
        self._send_home()

        lines = list(self._record_buffer)
        self._record_buffer.clear()
        start_wall = self._record_start_wall
        end_wall = time.time()

        with self._state._lock:
            self._state.recording_active = False
            self._state.record_samples = 0
            self._state.saving_recording = True
            self._state.last_resp = "[SAVING RECORDING...]"
            self._state.last_error = ""

        if async_save:
            th = threading.Thread(
                target=self._save_recording_worker,
                args=(lines, start_wall, end_wall),
                daemon=True,
                name="recording-save-worker",
            )
            with self._save_lock:
                self._save_thread = th
            th.start()
        else:
            saved = self._save_recording_lines(lines, start_wall, end_wall)
            with self._state._lock:
                self._state.saving_recording = False
                self._state.last_resp = f"Recording saved -> {saved}" if saved else "Recording stopped"

    def _save_recording_worker(self, lines: list[str], start_wall: float, end_wall: float) -> None:
        saved = self._save_recording_lines(lines, start_wall, end_wall)
        with self._state._lock:
            self._state.saving_recording = False
            self._state.last_resp = f"Recording saved -> {saved}" if saved else "Recording stopped"

    def _save_recording_lines(self, lines: list[str], start_wall: float, end_wall: float) -> str | None:
        if not lines:
            return None

        end_stamp = datetime.fromtimestamp(end_wall).strftime("%Y%m%d_%H%M%S")
        sessions_dir = Path("logs") / "sessions"
        session_dir = sessions_dir / f"session_{end_stamp}"
        session_dir.mkdir(parents=True, exist_ok=True)

        data_path = session_dir / "session.jsonl"
        with data_path.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line)
                fh.write("\n")

        meta = {
            "start_wall": float(start_wall),
            "end_wall": float(end_wall),
            "duration_s": max(0.0, float(end_wall - start_wall)),
            "samples": int(len(lines)),
            "tof_channels_enabled": int(self._tof_channel_count),
            "threshold_mm": float(self._tof_state.tof_threshold_mm),
            "obstacle_decay_s": float(self._obstacle_decay_s),
            "persistent_memory_active": bool(self._persistent_memory_active),
            "imu_lpf_alpha": float(self._tof_state.imu_lpf_alpha),
            "imu_calibrated": bool(self._imu_calibrated),
            "imu_calibration_rms_deg": float(self._imu_cal_rms_deg),
            "imu_rot_wrist_from_imu": self._imu_rot_wrist_from_imu.tolist(),
            "feasible_region_dir": str(self._feasible_region_dir),
            "feasible_cloud_path": str(self._feasible_region_dir / "workspace_cloud.npy"),
            "urdf_path": str(self._repo_root / "Tinkerkit_model" / "tinkerkit.urdf"),
        }
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return str(data_path)

    def _record_tick(self, now: float, tof_snap: dict) -> None:
        if not self._recording_active:
            return

        min_by_ch = []
        grids = tof_snap.get("grids", [])
        validity = tof_snap.get("validity", [])
        for ch, grid in enumerate(grids):
            try:
                if ch < len(validity):
                    mask = validity[ch].astype(bool)
                    vals = grid[mask]
                else:
                    vals = grid[~np.isnan(grid)]
                if vals.size == 0:
                    min_by_ch.append(None)
                else:
                    min_by_ch.append(float(np.min(vals)))
            except Exception:
                min_by_ch.append(None)

        with self._state._lock:
            joints_fk = [float(x) for x in self._state.joints]
            theta_fk = float(self._state.theta)
            r_fk = float(self._state.r)
            z_fk = float(self._state.z)
            eef = self._tool_joint_point_base(joints_fk, theta_fk, r_fk, z_fk)

            obstacles = []
            for m in self._obstacle_memory:
                age = max(0.0, now - float(m.get("t", now)))
                if self._persistent_memory_active:
                    clear_required = max(1, int(self._persistent_clear_required))
                    miss_count = max(0, int(m.get("miss_count", 0)))
                    opacity = max(0.25, 1.0 - (min(clear_required, miss_count) / float(clear_required)))
                else:
                    decay = max(1e-3, self._obstacle_decay_s)
                    opacity = max(0.0, 1.0 - (age / decay))
                p = np.asarray(m.get("point", [0.0, 0.0, 0.0]), dtype=float)
                obstacles.append(
                    {
                        "x_m": float(p[0]),
                        "y_m": float(p[1]),
                        "z_m": float(p[2]),
                        "source": str(m.get("source", "")),
                        "age_s": float(age),
                        "opacity": float(opacity),
                        "persistent": bool(self._persistent_memory_active),
                        "miss_count": int(m.get("miss_count", 0)),
                    }
                )

            future_plan = []
            for fp in self._future_plan_preview:
                try:
                    e = fp.get("eef_m", [0.0, 0.0, 0.0])
                    jv = fp.get("joints_deg", [])
                    future_plan.append(
                        {
                            "t_s": float(fp.get("t_s", 0.0)),
                            "theta": float(fp.get("theta", 0.0)),
                            "r_mm": float(fp.get("r_mm", 0.0)),
                            "z_mm": float(fp.get("z_mm", 0.0)),
                            "joints_deg": [int(v) for v in jv],
                            "eef_m": [float(e[0]), float(e[1]), float(e[2])],
                            "waypoint": bool(fp.get("waypoint", False)),
                        }
                    )
                except Exception:
                    continue

            row = {
                "host_monotonic": round(now, 6),
                "wall_time": time.time(),
                "theta": round(self._state.theta, 3),
                "r_mm": round(self._state.r, 3),
                "z_mm": round(self._state.z, 3),
                "wrist_offset_deg": round(self._state.wrist_offset, 3),
                "wrist_rot_deg": int(self._state.wrist_rot),
                "gripper_deg": int(self._state.gripper),
                "joints_deg": [int(x) for x in self._state.joints],
                "eef_m": [float(eef[0]), float(eef[1]), float(eef[2])],
                "last_sent": {
                    "theta": round(self._last_sent_theta, 3),
                    "r_mm": round(self._last_sent_r, 3),
                    "z_mm": round(self._last_sent_z, 3),
                },
                "mode": self._state.control_mode,
                "optimizer_active": self._state.optimizer_active,
                "fallback_active": self._state.fallback_active,
                "obstacle": {
                    "response": self._state.obstacle_response,
                    "source": self._state.obstacle_source,
                    "distance_mm": self._state.obstacle_dist_mm,
                },
                "obstacles": obstacles,
                "persistent_memory_active": bool(self._persistent_memory_active),
                "future_plan": future_plan,
                "tof_min_by_ch_mm": min_by_ch,
                "profile_active": self._profile_side_active,
                "memory_count": len(self._obstacle_memory),
                "decay_s": float(self._obstacle_decay_s),
                "imu": {
                    "online": bool(tof_snap.get("imu", {}).get("online", False)),
                    "calibrated": bool(tof_snap.get("imu", {}).get("calibrated", False)),
                    "lpf_alpha": float(tof_snap.get("imu", {}).get("lpf_alpha", 0.25)),
                    "ax_g": float(tof_snap.get("imu", {}).get("ax_g", 0.0)),
                    "ay_g": float(tof_snap.get("imu", {}).get("ay_g", 0.0)),
                    "az_g": float(tof_snap.get("imu", {}).get("az_g", 0.0)),
                    "gx_dps": float(tof_snap.get("imu", {}).get("gx_dps", 0.0)),
                    "gy_dps": float(tof_snap.get("imu", {}).get("gy_dps", 0.0)),
                    "gz_dps": float(tof_snap.get("imu", {}).get("gz_dps", 0.0)),
                    "rms_deg": float(self._imu_cal_rms_deg),
                },
            }
            self._state.record_samples += 1

        self._record_buffer.append(json.dumps(row, separators=(",", ":")))

    def _wrist_origin_base_m(self, theta_deg: float, r_mm: float, z_mm: float) -> np.ndarray:
        th = math.radians(theta_deg)
        return np.array([r_mm * math.cos(th), r_mm * math.sin(th), z_mm], dtype=float) / 1000.0

    def _arm_pitch_deg(self, r_mm: float, z_mm: float) -> float:
        return math.degrees(math.atan2(z_mm, max(abs(r_mm), 1e-3)))

    def _wrist_pitch_pose_base(
        self,
        joints_deg: list[float],
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        use_imu: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        origin = None
        rot = None

        if self._kinematics is not None:
            try:
                tf = self._kinematics.joint_transform(joints_deg, "wrist_pitch")
                origin = tf[:3, 3].copy()
                rot = tf[:3, :3].copy()
            except Exception:
                origin = None
                rot = None

        if origin is None or rot is None:
            # Fallback approximation if URDF FK is unavailable.
            origin = self._wrist_origin_base_m(theta_deg, r_mm, z_mm)
            th = math.radians(theta_deg)
            pitch = math.radians(self._arm_pitch_deg(r_mm, z_mm))
            base_rot = np.array(
                [[math.cos(th), -math.sin(th), 0.0], [math.sin(th), math.cos(th), 0.0], [0.0, 0.0, 1.0]],
                dtype=float,
            )
            arm_rot = np.array(
                [[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]],
                dtype=float,
            )
            rot = base_rot @ arm_rot

        if use_imu:
            rot = self._imu_correct_wrist_rotation(rot)

        return origin, rot

    def _tool_joint_point_base(
        self,
        joints_deg: list[float],
        theta_deg: float,
        r_mm: float,
        z_mm: float,
    ) -> np.ndarray:
        if self._kinematics is not None:
            try:
                tf = self._kinematics.joint_transform(joints_deg, "wrist_roll")
                return tf[:3, 3].copy()
            except Exception:
                pass
        return self._wrist_origin_base_m(theta_deg, r_mm, z_mm)

    def _memory_key(self, point_m: np.ndarray) -> tuple[int, int, int]:
        p = np.asarray(point_m, dtype=float).reshape(3)
        q = np.round(p / max(1e-6, float(self._memory_voxel_m))).astype(int)
        return int(q[0]), int(q[1]), int(q[2])

    def _parse_memory_channel(self, source: str) -> int:
        if not isinstance(source, str):
            return -1
        if source.startswith("tof_ch"):
            try:
                return int(source[6:])
            except Exception:
                return -1
        return -1

    def _prune_obstacle_memory(self, now: float) -> None:
        if self._persistent_memory_active:
            self._obstacle_memory = [
                m for m in self._obstacle_memory if int(m.get("miss_count", 0)) < int(self._persistent_clear_required)
            ]
        else:
            self._obstacle_memory = [m for m in self._obstacle_memory if (now - float(m.get("t", now))) <= self._obstacle_decay_s]

    def _upsert_memory_point(self, point_m: np.ndarray, source: str, now: float) -> None:
        p = np.asarray(point_m, dtype=float).reshape(3)
        key = self._memory_key(p)
        for m in self._obstacle_memory:
            if m.get("key") == key and m.get("source") == source:
                old = np.asarray(m.get("point", p), dtype=float)
                m["point"] = 0.5 * old + 0.5 * p
                m["t"] = now
                m["last_seen"] = now
                m["miss_count"] = 0
                return

        self._obstacle_memory.append(
            {
                "t": now,
                "last_seen": now,
                "point": p,
                "source": source,
                "miss_count": 0,
                "key": key,
            }
        )

    def _obstacle_points(self) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for m in self._obstacle_memory:
            try:
                pts.append(np.asarray(m["point"], dtype=float).reshape(3))
            except Exception:
                pass
        return pts

    def _nearest_obstacle_for_polar(
        self,
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        obstacle_points: list[np.ndarray],
    ) -> tuple[np.ndarray | None, float]:
        if not obstacle_points:
            return None, float("inf")
        p = self._wrist_origin_base_m(theta_deg, r_mm, z_mm)
        d_min = float("inf")
        best = None
        for op in obstacle_points:
            d = float(np.linalg.norm(np.asarray(op, dtype=float).reshape(3) - p))
            if d < d_min:
                d_min = d
                best = np.asarray(op, dtype=float).reshape(3)
        return best, d_min * 1000.0

    def _predict_joint_config(
        self,
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        wrist_offset_deg: float,
        wrist_rot_deg: int,
        gripper_deg: int,
        fallback_joints: list[float],
    ) -> list[int]:
        x_mm, y_mm = polar_to_cartesian(theta_deg, r_mm)
        ik = solve_ik(x_mm, y_mm, z_mm)
        if ik is None:
            base = [int(round(v)) for v in fallback_joints[:6]]
            if len(base) < 6:
                base = [90, 90, 90, 90, 90, int(gripper_deg)]
            base[4] = int(wrist_rot_deg)
            base[5] = int(gripper_deg)
            return base

        wrist_v = apply_wrist_offset(float(ik.wrist_vert), float(wrist_offset_deg))
        return [
            int(ik.base),
            int(ik.shoulder),
            int(ik.elbow),
            int(wrist_v),
            int(wrist_rot_deg),
            int(gripper_deg),
        ]

    def _plan_future_trajectory(
        self,
        desired_theta: float,
        desired_r: float,
        desired_z: float,
        start_theta: float,
        start_r: float,
        start_z: float,
        wrist_offset: float,
        wrist_rot: int,
        gripper: int,
        base_obstacle_response: str,
        obstacle_threshold_mm: float,
        dt_first_s: float,
        endpoint_lock: bool,
    ) -> list[dict]:
        dt_preview = max(0.05, float(self._future_plan_dt_s))
        n_steps = max(2, int(round(max(0.2, float(self._future_plan_horizon_s)) / dt_preview)))

        obs_points = self._obstacle_points()
        cur_theta = float(start_theta)
        cur_r = float(start_r)
        cur_z = float(start_z)
        fallback_joints = [float(x) for x in self._state.joints]

        out: list[dict] = []
        for k in range(n_steps):
            dt_k = max(1e-3, float(dt_first_s) if k == 0 else dt_preview)
            obs_point, obs_dist_mm = self._nearest_obstacle_for_polar(cur_theta, cur_r, cur_z, obs_points)

            if base_obstacle_response == ObstacleResponse.BACK_AWAY:
                local_obs = ObstacleResponse.BACK_AWAY
                local_dist = 0.0
            elif obs_point is not None and np.isfinite(obs_dist_mm) and obs_dist_mm <= (1.6 * obstacle_threshold_mm):
                local_obs = ObstacleResponse.REPLAN
                local_dist = float(obs_dist_mm)
            else:
                local_obs = ObstacleResponse.CLEAR
                local_dist = -1.0
                obs_point = None

            theta_cmd, r_cmd, z_cmd, debug = self._replanner.step(
                desired_theta=desired_theta,
                desired_r=desired_r,
                desired_z=desired_z,
                prev_theta=cur_theta,
                prev_r=cur_r,
                prev_z=cur_z,
                obstacle_response=local_obs,
                obstacle_dist_mm=local_dist,
                obstacle_threshold_mm=obstacle_threshold_mm,
                dt_s=dt_k,
                r_bounds=(R_MIN, R_MAX),
                z_bounds=(Z_MIN, Z_MAX),
                obstacle_point_m=obs_point,
                endpoint_lock=bool(endpoint_lock and (k >= (n_steps - 2))),
            )

            if self._profile_side_active and self._profile_target_theta is not None:
                tgt = float(self._profile_target_theta)
                if tgt >= cur_theta:
                    theta_cmd = max(theta_cmd, cur_theta)
                    theta_cmd = min(theta_cmd, tgt)
                else:
                    theta_cmd = min(theta_cmd, cur_theta)
                    theta_cmd = max(theta_cmd, tgt)

            joints_pred = self._predict_joint_config(
                theta_deg=theta_cmd,
                r_mm=r_cmd,
                z_mm=z_cmd,
                wrist_offset_deg=wrist_offset,
                wrist_rot_deg=wrist_rot,
                gripper_deg=gripper,
                fallback_joints=fallback_joints,
            )
            fallback_joints = [float(v) for v in joints_pred]
            eef = self._tool_joint_point_base([float(v) for v in joints_pred], theta_cmd, r_cmd, z_cmd)

            out.append(
                {
                    "t_s": float((k + 1) * dt_preview),
                    "theta": float(theta_cmd),
                    "r_mm": float(r_cmd),
                    "z_mm": float(z_cmd),
                    "joints_deg": [int(v) for v in joints_pred],
                    "eef_m": [float(eef[0]), float(eef[1]), float(eef[2])],
                    "waypoint": bool((k % 3) == 0 or (k == n_steps - 1)),
                    "debug": debug,
                }
            )

            cur_theta, cur_r, cur_z = float(theta_cmd), float(r_cmd), float(z_cmd)

        return out

    def _memory_nearest(self, joints_deg: list[float]) -> tuple[float, str, np.ndarray | None]:
        now = time.monotonic()
        self._prune_obstacle_memory(now)
        if not self._obstacle_memory:
            return float("inf"), "", None

        with self._state._lock:
            theta = float(self._state.theta)
            r = float(self._state.r)
            z = float(self._state.z)

        tool = self._tool_joint_point_base(joints_deg, theta, r, z)
        d_min = float("inf")
        src = ""
        pt = None
        for m in self._obstacle_memory:
            d = float(np.linalg.norm(np.asarray(m["point"], dtype=float) - tool))
            if d < d_min:
                d_min = d
                src = m.get("source", "")
                pt = np.asarray(m["point"], dtype=float)
        return d_min * 1000.0, src, pt

    def _point_in_feasible_region(self, point_m: np.ndarray) -> bool:
        if self._feasible_region is None:
            return True
        try:
            return bool(self._feasible_region.contains(point_m))
        except Exception:
            return True

    def _update_obstacle_memory(self, now: float, tof_snap: dict) -> None:
        self._prune_obstacle_memory(now)

        with self._state._lock:
            theta = float(self._state.theta)
            r = float(self._state.r)
            z = float(self._state.z)
            joints = [float(x) for x in self._state.joints]

        threshold_mm = float(tof_snap.get("tof_threshold_mm", TOF_THRESHOLD_MM))
        wrist_origin, wrist_rot = self._wrist_pitch_pose_base(joints, theta, r, z)
        tool_point = self._tool_joint_point_base(joints, theta, r, z)

        grids = tof_snap.get("grids", [])
        validity = tof_snap.get("validity", [])
        last_rx = tof_snap.get("last_rx", [])

        channel_hits: dict[int, list[np.ndarray]] = {}
        processed_channels: set[int] = set()

        for ch in range(min(self._tof_channel_count, len(grids))):
            if ch >= len(last_rx):
                continue
            if (time.time() - float(last_rx[ch])) > TOF_FRESHNESS_S:
                continue

            processed_channels.add(ch)
            channel_hits[ch] = []

            grid = np.asarray(grids[ch], dtype=float)
            if grid.size != 64:
                continue
            val = np.asarray(validity[ch], dtype=int) if ch < len(validity) else np.ones((8, 8), dtype=int)
            if val.size != 64:
                continue
            if np.isnan(grid).all():
                continue

            seq_list = tof_snap.get("seq", [])
            mcu_list = tof_snap.get("mcu_ms", [])
            sid_list = tof_snap.get("sensor_id", [])
            st_list = tof_snap.get("status", [])

            frame = ToFFrameV1(
                seq=int(seq_list[ch]) if ch < len(seq_list) else -1,
                mcu_ms=int(mcu_list[ch]) if ch < len(mcu_list) else -1,
                sensor_id=sid_list[ch] if ch < len(sid_list) else f"ch{ch}",
                mux_channel=ch,
                joint_id="wrist_pitch",
                status=int(st_list[ch]) if ch < len(st_list) else 0,
                distances_mm=list(grid.reshape(-1)),
                validity=list(val.reshape(-1)),
            )

            pts = frame_to_points_base(
                frame,
                wrist_origin,
                wrist_rot_base=wrist_rot,
                base_theta_deg=theta,
                arm_pitch_deg=self._arm_pitch_deg(r, z),
            )
            if pts.size == 0:
                continue

            dists_m = np.linalg.norm(pts - tool_point[None, :], axis=1)
            valid_idx = np.where((dists_m * 1000.0) <= threshold_mm)[0]
            if valid_idx.size == 0:
                continue

            order = valid_idx[np.argsort(dists_m[valid_idx])]
            max_add = 6 if self._persistent_memory_active else 1
            for idx in order:
                p = np.asarray(pts[int(idx)], dtype=float)
                if not self._point_in_feasible_region(p):
                    continue
                self._upsert_memory_point(p, source=f"tof_ch{ch}", now=now)
                channel_hits[ch].append(p)
                if len(channel_hits[ch]) >= max_add:
                    break

        if self._persistent_memory_active:
            for m in self._obstacle_memory:
                ch = self._parse_memory_channel(str(m.get("source", "")))
                if ch < 0 or ch not in processed_channels:
                    continue

                hits = channel_hits.get(ch, [])
                if not hits:
                    m["miss_count"] = int(m.get("miss_count", 0)) + 1
                    continue

                p_mem = np.asarray(m.get("point", [0.0, 0.0, 0.0]), dtype=float)
                d_min = min(float(np.linalg.norm(np.asarray(h, dtype=float) - p_mem)) for h in hits)
                if d_min <= float(self._memory_match_radius_m):
                    m["miss_count"] = 0
                    m["last_seen"] = now
                else:
                    m["miss_count"] = int(m.get("miss_count", 0)) + 1

            self._obstacle_memory = [
                m for m in self._obstacle_memory if int(m.get("miss_count", 0)) < int(self._persistent_clear_required)
            ]

        if len(self._obstacle_memory) > 768:
            self._obstacle_memory = self._obstacle_memory[-768:]

    def _current_tuning_values(self) -> dict:
        return {
            "w_track_theta": float(self._weights.w_track_theta),
            "w_smooth_theta": float(self._weights.w_smooth_theta),
            "w_track_r": float(self._weights.w_track_r),
            "w_smooth_r": float(self._weights.w_smooth_r),
            "w_avoid_r": float(self._weights.w_avoid_r),
            "w_track_z": float(self._weights.w_track_z),
            "w_smooth_z": float(self._weights.w_smooth_z),
            "w_avoid_z": float(self._weights.w_avoid_z),
            "tof_threshold_mm": float(self._tof_state.tof_threshold_mm),
            "obstacle_decay_s": float(self._obstacle_decay_s),
            "persistent_memory_active": bool(self._persistent_memory_active),
            "imu_lpf_alpha": float(self._tof_state.imu_lpf_alpha),
        }

    def _apply_tuning_values(self, values: dict) -> None:
        self._weights = RelaxedWeights(
            w_track_theta=max(0.01, float(values["w_track_theta"])),
            w_smooth_theta=max(0.01, float(values["w_smooth_theta"])),
            w_track_r=max(0.01, float(values["w_track_r"])),
            w_smooth_r=max(0.01, float(values["w_smooth_r"])),
            w_avoid_r=max(0.01, float(values["w_avoid_r"])),
            w_track_z=max(0.01, float(values["w_track_z"])),
            w_smooth_z=max(0.01, float(values["w_smooth_z"])),
            w_avoid_z=max(0.01, float(values["w_avoid_z"])),
        )
        self._replanner = LinearRelaxedReplanner(weights=self._weights, limits=self._limits)

        with self._tof_state._lock:
            self._tof_state.tof_threshold_mm = max(40.0, min(3000.0, float(values["tof_threshold_mm"])))
            self._tof_state.set_imu_lpf_alpha(float(values.get("imu_lpf_alpha", self._tof_state.imu_lpf_alpha)))

        self._obstacle_decay_s = max(0.1, float(values["obstacle_decay_s"]))
        self._persistent_memory_active = bool(values.get("persistent_memory_active", self._persistent_memory_active))

        with self._state._lock:
            self._state.last_resp = "Tuning parameters updated"
            self._state.last_error = ""

    def _begin_imu_calibration(self) -> None:
        if not self._teensy_port or (not self._tof_bridge.is_open()):
            with self._state._lock:
                self._state.last_error = "Cannot calibrate IMU: Teensy link offline"
            return

        with self._imu_cal_lock:
            if self._imu_calibrating:
                with self._state._lock:
                    self._state.last_resp = "IMU calibration already running"
                return
            self._imu_calibrating = True

        self._disable_side_profile("imu calibration")
        with self._imu_cal_lock:
            self._imu_calibrated = False
            self._imu_cal_rms_deg = 0.0
        self._tof_state.set_imu_calibrated(False)
        with self._state._lock:
            self._state.last_resp = "IMU calibration started"
            self._state.last_error = ""

        th = threading.Thread(
            target=self._imu_calibration_worker,
            daemon=True,
            name="imu-calibration-worker",
        )
        with self._imu_cal_lock:
            self._imu_calib_thread = th
        th.start()

    def _imu_calibration_worker(self) -> None:
        ok = False
        msg = ""
        try:
            ok, msg = self._run_imu_calibration()
        except Exception as exc:
            ok = False
            msg = f"IMU calibration failed: {exc}"

        with self._imu_cal_lock:
            self._imu_calibrating = False

        if ok:
            with self._state._lock:
                self._state.last_resp = msg
                self._state.last_error = ""
        else:
            with self._state._lock:
                self._state.last_error = msg

    def _run_imu_calibration(self) -> tuple[bool, str]:
        with self._state._lock:
            restore_pose = (
                float(self._state.theta),
                float(self._state.r),
                float(self._state.z),
                float(self._state.wrist_offset),
                int(self._state.wrist_rot),
                int(self._state.gripper),
            )

        # Pose set chosen to excite wrist tilt in multiple directions while
        # staying in a conservative reachable envelope.
        poses = [
            (90.0, 112.0, 20.0, 0.0, 90, 73),
            (90.0, 140.0, 95.0, -20.0, 90, 73),
            (90.0, 100.0, -30.0, 20.0, 90, 73),
            (40.0, 118.0, 35.0, 0.0, 90, 73),
            (140.0, 118.0, 35.0, 0.0, 90, 73),
        ]

        src_accel: list[np.ndarray] = []
        dst_grav: list[np.ndarray] = []

        for theta, r_mm, z_mm, wv, wr, g in poses:
            self._send_calibration_pose(theta, r_mm, z_mm, wv, wr, g, settle_s=0.8)
            a_avg = self._collect_imu_accel_avg(duration_s=0.55, min_samples=12)
            if a_avg is None:
                continue

            with self._state._lock:
                joints = [float(x) for x in self._state.joints]
                stheta = float(self._state.theta)
                sr = float(self._state.r)
                sz = float(self._state.z)

            _, rot_fk = self._wrist_pitch_pose_base(joints, stheta, sr, sz, use_imu=False)
            if rot_fk is None:
                continue

            g_wrist = _normalize_vec(rot_fk.T @ np.array([0.0, 0.0, -1.0], dtype=float))
            a_norm = _normalize_vec(np.asarray(a_avg, dtype=float).reshape(3))
            if float(np.linalg.norm(g_wrist)) <= 1e-6 or float(np.linalg.norm(a_norm)) <= 1e-6:
                continue

            src_accel.append(a_norm)
            dst_grav.append(g_wrist)

        # Restore operator pose after calibration movement sequence.
        self._send_calibration_pose(*restore_pose, settle_s=0.6)

        if len(src_accel) < 3:
            with self._imu_cal_lock:
                self._imu_calibrated = False
                self._imu_cal_rms_deg = 0.0
            self._tof_state.set_imu_calibrated(False)
            return False, "IMU calibration failed: insufficient valid samples"

        rot_wrist_from_imu = self._solve_wahba_rotation(src_accel, dst_grav)
        errs = []
        for a_norm, g_ref in zip(src_accel, dst_grav):
            g_hat = _normalize_vec(rot_wrist_from_imu @ a_norm)
            c = float(np.dot(g_hat, g_ref))
            c = max(-1.0, min(1.0, c))
            errs.append(math.degrees(math.acos(c)))

        rms_deg = float(math.sqrt(sum(e * e for e in errs) / max(1, len(errs))))
        with self._imu_cal_lock:
            self._imu_rot_wrist_from_imu = rot_wrist_from_imu
            self._imu_calibrated = True
            self._imu_cal_rms_deg = rms_deg
        self._tof_state.set_imu_calibrated(True)

        return True, f"IMU calibration complete (rms={rms_deg:.1f} deg, samples={len(src_accel)})"

    def _send_calibration_pose(
        self,
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        wrist_offset_deg: float,
        wrist_rot_deg: int,
        gripper_deg: int,
        settle_s: float = 0.8,
    ) -> None:
        cmd = cmd_set_ik_polar(theta_deg, r_mm, z_mm, wrist_offset_deg, wrist_rot_deg, gripper_deg)
        with self._state._lock:
            self._state.theta = float(theta_deg)
            self._state.r = float(r_mm)
            self._state.z = float(z_mm)
            self._state.wrist_offset = float(wrist_offset_deg)
            self._state.wrist_rot = int(wrist_rot_deg)
            self._state.gripper = int(gripper_deg)
            self._state.last_cmd = cmd.strip()

        self._bridge.send_cmd(cmd)
        self._last_sent_theta = float(theta_deg)
        self._last_sent_r = float(r_mm)
        self._last_sent_z = float(z_mm)
        self._last_send_monotonic = time.monotonic()

        end_t = time.monotonic() + max(0.1, float(settle_s))
        while time.monotonic() < end_t:
            self._drain_responses()
            time.sleep(0.02)

    def _collect_imu_accel_avg(self, duration_s: float = 0.5, min_samples: int = 10) -> np.ndarray | None:
        deadline = time.monotonic() + max(0.1, float(duration_s))
        samples: list[np.ndarray] = []
        last_seq = -10**9

        while time.monotonic() < deadline:
            with self._tof_state._lock:
                online = bool(self._tof_state.imu_online)
                seq = int(self._tof_state.imu_seq)
                last_rx = float(self._tof_state.imu_last_rx)
                a = np.asarray(self._tof_state.imu_accel_g, dtype=float).copy()

            if online and (time.time() - last_rx) <= TOF_FRESHNESS_S and seq != last_seq:
                if float(np.linalg.norm(a)) > 1e-3:
                    samples.append(a)
                    last_seq = seq

            time.sleep(0.01)

        if len(samples) < int(min_samples):
            return None
        return np.mean(np.vstack(samples), axis=0)

    @staticmethod
    def _solve_wahba_rotation(sources: list[np.ndarray], targets: list[np.ndarray]) -> np.ndarray:
        h = np.zeros((3, 3), dtype=float)
        for src, dst in zip(sources, targets):
            s = _normalize_vec(np.asarray(src, dtype=float).reshape(3))
            d = _normalize_vec(np.asarray(dst, dtype=float).reshape(3))
            h += np.outer(d, s)

        u, _, vt = np.linalg.svd(h)
        r = u @ vt
        if float(np.linalg.det(r)) < 0.0:
            u[:, -1] *= -1.0
            r = u @ vt
        return r

    def _imu_correct_wrist_rotation(self, wrist_rot_fk: np.ndarray) -> np.ndarray:
        with self._imu_cal_lock:
            calibrated = bool(self._imu_calibrated)
            rot_wrist_from_imu = self._imu_rot_wrist_from_imu.copy()
            fuse_weight = max(0.0, min(1.0, float(self._imu_fuse_weight)))
            max_angle = math.radians(max(1.0, float(self._imu_max_tilt_correction_deg)))

        if not calibrated or fuse_weight <= 1e-6:
            return wrist_rot_fk

        with self._tof_state._lock:
            online = bool(self._tof_state.imu_online)
            last_rx = float(self._tof_state.imu_last_rx)
            a_imu = np.asarray(self._tof_state.imu_accel_g, dtype=float).copy()

        if (not online) or ((time.time() - last_rx) > TOF_FRESHNESS_S):
            return wrist_rot_fk

        a_imu = _normalize_vec(a_imu)
        if float(np.linalg.norm(a_imu)) <= 1e-6:
            return wrist_rot_fk

        g_meas_wrist = _normalize_vec(rot_wrist_from_imu @ a_imu)
        g_fk_wrist = _normalize_vec(wrist_rot_fk.T @ np.array([0.0, 0.0, -1.0], dtype=float))

        dot = float(np.dot(g_meas_wrist, g_fk_wrist))
        dot = max(-1.0, min(1.0, dot))
        axis = np.cross(g_meas_wrist, g_fk_wrist)
        axis_n = float(np.linalg.norm(axis))
        if axis_n <= 1e-9:
            return wrist_rot_fk

        angle = math.atan2(axis_n, dot)
        angle = min(max_angle, angle * fuse_weight)
        rot_delta = _rot_axis_angle(axis, angle)
        return wrist_rot_fk @ rot_delta

    def _send_wrist_rot(self) -> None:
        with self._state._lock:
            wr = self._state.wrist_rot
            self._state.joints[JOINT_WRIST_ROT] = wr
            cmd = cmd_set_joint(JOINT_WRIST_ROT, wr)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_gripper(self) -> None:
        with self._state._lock:
            g = self._state.gripper
            self._state.joints[JOINT_GRIPPER] = g
            cmd = cmd_set_joint(JOINT_GRIPPER, g)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_delta(self) -> None:
        with self._state._lock:
            d = self._state.delta
            cmd = cmd_set_delta(d)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_home(self) -> None:
        self._state.restore_equil()
        cmd = cmd_home()
        with self._state._lock:
            self._state.last_cmd = cmd.strip()
            self._state.last_error = ""
        self._bridge.send_cmd(cmd)
        self._bridge.send_cmd(cmd_get_pos())

    def _apply_state(self, state_dict: dict) -> None:
        with self._state._lock:
            self._state.joints = list(state_dict["joints"])
            self._state.theta = state_dict["theta"]
            self._state.r = state_dict["r"]
            self._state.z = state_dict["z"]
            self._state.wrist_offset = state_dict["wrist_offset"]
            self._state.wrist_rot = state_dict["wrist_rot"]
            self._state.gripper = state_dict["gripper"]

        cmd = cmd_set_all(state_dict["joints"])
        with self._state._lock:
            self._state.last_cmd = cmd.strip()
            self._state.last_error = ""
        self._bridge.send_cmd(cmd)

    def _run_named_state(self, name: str) -> None:
        st = self._state_lib.get_state(name)
        if st:
            self._apply_state(st)

    def _open_states_menu(self) -> None:
        from .states_menu import run_states_menu

        result = run_states_menu(self._stdscr, self._state_lib, self._state.snapshot())
        if result:
            self._apply_state(result)
        curses.curs_set(0)
        if self._kb is not None:
            self._kb.reset_after_overlay()

    def _open_seq_editor(self) -> None:
        from .sequence_editor import run_sequence_editor

        run_sequence_editor(self._stdscr, self._state_lib, self._run_named_state)
        curses.curs_set(0)
        if self._kb is not None:
            self._kb.reset_after_overlay()

    def _open_tuning_menu(self) -> None:
        from .tuning_menu import run_tuning_menu

        run_tuning_menu(self._stdscr, self._current_tuning_values(), self._apply_tuning_values)
        curses.curs_set(0)
        if self._kb is not None:
            self._kb.reset_after_overlay()






































