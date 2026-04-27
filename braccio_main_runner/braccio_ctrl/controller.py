"""Minimal Braccio host controller with deterministic ToF avoidance."""

from __future__ import annotations

import curses
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from planning.sector_planner import ObstacleMemory, SectorWaypointPlanner
from planning.sensor_config import SENSOR_CONFIG, sensor_urdf_path, set_active_channel_count
from planning.tof_projection import frame_cells_to_points_base
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
    PROFILE_SWEEP_DWELL_S,
    PROFILE_SWEEP_GRIPPER_DEG,
    PROFILE_SWEEP_R_MM,
    PROFILE_SWEEP_STEP_DEG,
    PROFILE_SWEEP_THETA_MAX_DEG,
    PROFILE_SWEEP_THETA_MIN_DEG,
    PROFILE_SWEEP_WRIST_OFFSET_DEG,
    PROFILE_SWEEP_WRIST_ROT_DEG,
    PROFILE_SWEEP_Z_MM,
    R_MAX,
    R_MIN,
    R_STEP,
    THETA_STEP,
    TOF_BAUD_RATE,
    TOF_CHANNELS_MAX,
    TOF_FRESHNESS_S,
    TOF_THRESHOLD_MM,
    WRIST_R_STEP,
    WRIST_V_STEP,
    Z_SCAN_MAX_STEPS,
    Z_SCAN_STEP_MM,
    Z_MAX,
    Z_MIN,
    Z_STEP,
)
from .display import CursesDisplay
from .ik_solver import apply_wrist_offset, polar_to_cartesian, solve_ik, wrist_level_angle
from .keyboard_handler import KeyboardHandler
from .protocol import cmd_get_pos, cmd_home, cmd_set_delta, cmd_set_ik_polar, cmd_set_joint
from .serial_bridge import SerialBridge
from .tof_sensor import ObstacleResponse, ToFBridge, ToFState


FSM_DETECT = "detect_sweep"
FSM_CONFIRM = "confirm_stop"
FSM_PROBE = "vertical_probe"
FSM_PLAN = "planning_hold"
FSM_AVOID = "avoid_execute"
FSM_TRACK = "tracking_sweep"
FSM_HOLD = "hold_no_path"

VERTICAL_PROBE_START_Z_MM = -100.0
VERTICAL_PROBE_MAX_Z_MM = 160.0
AUTO_WRIST_CLEARANCE_MIN_DEG = 35.0
AUTO_WRIST_CLEARANCE_MAX_OFFSET_DEG = 45.0


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return np.asarray(v, dtype=float).reshape(3) / n


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def _rot_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n <= 1e-12 or abs(angle_rad) <= 1e-12:
        return np.eye(3, dtype=float)
    u = np.asarray(axis, dtype=float).reshape(3) / n
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    k = _skew(u)
    return np.eye(3, dtype=float) + s * k + (1.0 - c) * (k @ k)


def _orthonormalize_rot(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=float).reshape(3, 3))
    out = u @ vt
    if float(np.linalg.det(out)) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def _rotmat_to_axis_angle(rot: np.ndarray) -> tuple[np.ndarray, float]:
    r = _orthonormalize_rot(rot)
    trace = float(np.trace(r))
    cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    angle = float(math.acos(cos_angle))
    if angle <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float), 0.0
    axis = np.array(
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
        dtype=float,
    )
    axis_n = float(np.linalg.norm(axis))
    if axis_n <= 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float), angle
    return axis / axis_n, angle


def _rot_zyx_deg(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    y = math.radians(float(yaw_deg))
    p = math.radians(float(pitch_deg))
    r = math.radians(float(roll_deg))
    cz, sz = math.cos(y), math.sin(y)
    cy, sy = math.cos(p), math.sin(p)
    cx, sx = math.cos(r), math.sin(r)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    return rz @ ry @ rx


@dataclass
class ChannelSample:
    channel: int
    min_mm: float
    median_near_mm: float
    points_m: list[list[float]]
    seq: int
    rx_wall: float


class BraccioController:
    """Top-level controller for the minimal 4-ToF avoidance runtime."""

    def __init__(
        self,
        port: str,
        baud: int = BAUD_RATE,
        teensy_port: str | None = None,
        teensy_baud: int = TOF_BAUD_RATE,
        tof_channels_enabled: int = TOF_CHANNELS_MAX,
        imu_lpf_alpha: float = 0.25,
        obstacle_class: str = "SECTOR",
        hyperion_mode: bool = True,
        detect_threshold_mm: float = TOF_THRESHOLD_MM,
        confirm_frames: int = 5,
        confirm_hits: int = 3,
    ) -> None:
        self._bridge = SerialBridge(port, baud)
        self._state = ArmState()
        self._stdscr = None
        self._kb = None
        self._plotter = None
        self._tof_plotter = None

        self._teensy_port = teensy_port
        self._teensy_baud = int(teensy_baud)
        self._tof_channel_count = max(1, min(int(tof_channels_enabled), TOF_CHANNELS_MAX))
        self._hyperion_mode = bool(hyperion_mode or self._tof_channel_count >= 4)
        self._detect_threshold_mm = max(40.0, min(3000.0, float(detect_threshold_mm)))
        self._confirm_frames = max(1, int(confirm_frames))
        self._confirm_hits = max(1, min(self._confirm_frames, int(confirm_hits)))
        self._authoritative_channels = {0, 1}
        self._tracking_channel_count = max(1, min(4, self._tof_channel_count))
        self._detect_channel_count = min(2, self._tracking_channel_count)

        set_active_channel_count(self._tracking_channel_count)
        self._tof_state = ToFState(num_channels=self._tracking_channel_count)
        self._tof_state.tof_threshold_mm = self._detect_threshold_mm
        for ch in range(self._tracking_channel_count):
            self._tof_state.set_channel_threshold(ch, self._detect_threshold_mm)
        self._tof_state.set_imu_lpf_alpha(float(imu_lpf_alpha))
        self._tof_bridge = ToFBridge(self._tof_state)
        self._tof_active_count = 0

        self._control_period_s = 1.0 / max(1.0, CONTROL_LOOP_HZ)
        self._display_period_s = 1.0 / max(1.0, DISPLAY_LOOP_HZ)

        self._fsm_mode = FSM_DETECT
        self._mode_reason = "startup"
        self._planner = SectorWaypointPlanner(
            safe_clearance_mm=25.0,
            r_limits_mm=(R_MIN, R_MAX),
            z_limits_mm=(Z_MIN, Z_MAX),
            theta_limits_deg=(0.0, 180.0),
        )
        self._next_obstacle_id = 1
        self._obstacle_memory: ObstacleMemory | None = None
        self._confirm_source_channel: int | None = None
        self._confirm_samples: deque[ChannelSample] = deque(maxlen=self._confirm_frames)
        self._committed_plan: list[dict] = []
        self._candidate_debug: list[dict] = []
        self._last_planner_debug: dict = {}
        self._record_pending_events: list[dict] = []
        self._vertical_probe_targets: deque[float] = deque()
        self._vertical_probe_origin: tuple[float, float, float] | None = None
        self._vertical_probe_target_z: float | None = None
        self._vertical_probe_wait_until = 0.0
        self._vertical_probe_debug: list[dict] = []
        self._vertical_opening_pose: tuple[float, float, float] | None = None
        self._vertical_probe_xy_lock: tuple[float, float, float, float] | None = None
        self._auto_wrist_base_offset = float(self._state.wrist_offset)

        self._profile_side_active = False
        self._profile_target_theta: float | None = None
        self._profile_setpoint_theta = float(self._state.theta)
        self._profile_hold_until = 0.0

        self._last_sent_theta = float(self._state.theta)
        self._last_sent_r = float(self._state.r)
        self._last_sent_z = float(self._state.z)
        self._last_send_monotonic = time.monotonic()
        self._last_cmd_theta_vel = 0.0
        self._last_cmd_r_vel = 0.0
        self._last_cmd_z_vel = 0.0
        self._last_cycle_ms = 0.0

        self._recording_active = False
        self._record_buffer: list[str] = []
        self._record_start_wall = 0.0
        self._save_thread: threading.Thread | None = None
        self._save_lock = threading.Lock()

        self._workspace_dir = Path(__file__).resolve().parents[1]
        self._repo_root = Path(__file__).resolve().parents[2]
        self._kinematics = None
        self._init_kinematics()

        self._imu_cal_lock = threading.RLock()
        self._imu_calibrating = False
        self._imu_calibrated = False
        self._imu_cal_rms_deg = 0.0
        self._imu_rot_wrist_from_imu = np.eye(3, dtype=float)
        self._imu_rot_base_from_ref = np.eye(3, dtype=float)
        self._imu_ref_anchor_valid = False
        self._imu_calib_thread: threading.Thread | None = None
        self._imu_fuse_weight = 0.8
        self._imu_max_tilt_correction_deg = 35.0

        with self._state._lock:
            self._state.obstacle_class = obstacle_class
            self._state.tof_channels_enabled = self._tracking_channel_count
            self._state.hyperion_mode = self._hyperion_mode
            self._state.imu_lpf_alpha = float(imu_lpf_alpha)
            self._state.control_mode = self._fsm_mode
            self._state.planner_active = False

    @property
    def tof_state(self) -> ToFState:
        return self._tof_state

    def attach_plotter(self, plotter) -> None:
        self._plotter = plotter

    def attach_tof_plotter(self, tof_plotter) -> None:
        self._tof_plotter = tof_plotter

    def run(self) -> None:
        curses.wrapper(self._curses_main)

    def _curses_main(self, stdscr) -> None:
        self._stdscr = stdscr
        self._kb = KeyboardHandler(stdscr)
        display = CursesDisplay(stdscr)

        self._connect()
        if self._plotter is not None:
            self._plotter.start()
        if self._tof_plotter is not None:
            self._tof_plotter.start()

        last_control = 0.0
        last_display = 0.0
        quit_requested = False
        try:
            while not quit_requested:
                now = time.monotonic()
                action = self._kb.get_action() if self._kb is not None else None
                if action == "quit":
                    quit_requested = True
                elif action is not None:
                    self._handle_action(action)

                if now - last_control >= self._control_period_s:
                    start = time.monotonic()
                    self._control_step(now)
                    self._last_cycle_ms = (time.monotonic() - start) * 1000.0
                    last_control = now

                if now - last_display >= self._display_period_s:
                    self._sync_state_runtime_fields()
                    display.render(self._state.snapshot())
                    if self._plotter is not None:
                        self._plotter.pump()
                    if self._tof_plotter is not None:
                        self._tof_plotter.pump()
                    last_display = now

                time.sleep(0.005)
        finally:
            if self._recording_active:
                self._finish_recording_and_home(async_save=False)
            self._wait_for_save_completion(timeout_s=2.0)
            self._tof_bridge.close()
            self._bridge.close()
            if self._plotter is not None:
                self._plotter.stop()
            if self._tof_plotter is not None:
                self._tof_plotter.stop()

    def _connect(self) -> None:
        if self._bridge.connect():
            ok = self._bridge.ping(timeout=1.5)
            with self._state._lock:
                self._state.connected = bool(ok)
                self._state.last_resp = "Mega connected" if ok else "Mega serial open, PING not acknowledged"
                self._state.last_error = "" if ok else "Mega did not respond to PING/PONG"
            self._bridge.send_cmd(cmd_get_pos())
        else:
            with self._state._lock:
                self._state.connected = False
                self._state.last_error = f"Mega connect failed: {self._bridge.connect_error}"

        if self._teensy_port:
            if self._tof_bridge.connect(self._teensy_port, self._teensy_baud):
                self._set_tof_active_count(self._detect_channel_count, reason="startup_detect_mode")
            else:
                with self._state._lock:
                    self._state.last_error = f"Teensy connect failed on {self._teensy_port}"

    def _control_step(self, now: float) -> None:
        self._drain_responses()
        tof_snap = self._tof_state.snapshot()
        self._update_state_from_tof_snapshot(tof_snap)

        if self._fsm_mode == FSM_DETECT:
            hit = self._detect_authoritative_hit(tof_snap)
            if hit is not None:
                self._begin_confirmation(hit)
            elif self._profile_side_active:
                self._run_profile_step(now, tracking=False)
        elif self._fsm_mode == FSM_CONFIRM:
            self._confirm_step(now, tof_snap)
        elif self._fsm_mode == FSM_PROBE:
            self._vertical_probe_step(now, tof_snap)
        elif self._fsm_mode == FSM_PLAN:
            self._planning_step(now)
        elif self._fsm_mode == FSM_AVOID:
            self._avoid_step(now)
        elif self._fsm_mode == FSM_TRACK:
            self._tracking_step(now)
        elif self._fsm_mode == FSM_HOLD:
            self._send_hold_pose()

        self._sync_state_runtime_fields()
        self._record_tick(now, tof_snap)

    def _detect_authoritative_hit(self, snap: dict) -> ChannelSample | None:
        best: ChannelSample | None = None
        for ch in sorted(self._authoritative_channels):
            if ch >= self._tracking_channel_count:
                continue
            sample = self._sample_channel(snap, ch, self._detect_threshold_mm)
            if sample is None:
                continue
            if sample.median_near_mm <= self._detect_threshold_mm:
                if best is None or sample.median_near_mm < best.median_near_mm:
                    best = sample
        return best

    def _begin_confirmation(self, hit: ChannelSample) -> None:
        self._confirm_samples.clear()
        self._confirm_source_channel = int(hit.channel)
        self._confirm_samples.append(hit)
        self._set_tof_active_count(self._tracking_channel_count, reason="confirm_obstacle")
        self._transition(FSM_CONFIRM, f"tof_ch{hit.channel} threshold {hit.median_near_mm:.0f}mm")
        self._send_hold_pose()
        with self._state._lock:
            self._state.obstacle_response = ObstacleResponse.REPLAN
            self._state.obstacle_source = f"tof_ch{hit.channel}"
            self._state.obstacle_dist_mm = float(hit.median_near_mm)

    def _confirm_step(self, now: float, snap: dict) -> None:
        self._send_hold_pose()
        source = 0 if self._confirm_source_channel is None else int(self._confirm_source_channel)
        sample = self._sample_channel(snap, source, self._detect_threshold_mm)
        if sample is not None:
            self._confirm_samples.append(sample)

        hits = [s for s in self._confirm_samples if s.median_near_mm <= self._detect_threshold_mm]
        if len(hits) >= self._confirm_hits:
            self._obstacle_memory = self._memory_from_samples(now, hits)
            self._start_vertical_probe(now)
            return

        if len(self._confirm_samples) >= self._confirm_frames:
            self._reset_obstacle_episode("confirmation_failed")

    def _memory_from_samples(self, now: float, samples: list[ChannelSample]) -> ObstacleMemory:
        points: list[np.ndarray] = []
        support_channels: set[int] = set()
        for sample in samples:
            support_channels.add(int(sample.channel))
            for p in sample.points_m:
                points.append(np.asarray(p, dtype=float).reshape(3))

        if points:
            arr = np.vstack(points)
            centroid = np.median(arr, axis=0)
            spreads = np.linalg.norm(arr - centroid.reshape(1, 3), axis=1) * 1000.0
            radius_mm = max(35.0, min(80.0, float(np.percentile(spreads, 70)) + 10.0))
        else:
            dist_m = float(np.median([s.median_near_mm for s in samples])) / 1000.0
            theta = math.radians(float(self._last_sent_theta))
            centroid = np.array([dist_m * math.cos(theta), dist_m * math.sin(theta), self._last_sent_z / 1000.0])
            radius_mm = 30.0

        theta_deg, r_mm, z_mm = self._point_to_polar_mm(centroid)
        source = int(samples[0].channel)
        memory = ObstacleMemory(
            obstacle_id=int(self._next_obstacle_id),
            source_channel=source,
            centroid_m=[float(v) for v in centroid],
            theta_deg=float(theta_deg),
            r_mm=float(r_mm),
            z_mm=float(z_mm),
            radius_mm=float(radius_mm),
            confidence=float(min(1.0, len(samples) / max(1, self._confirm_frames))),
            created_monotonic=float(now),
            sample_count=int(len(samples)),
            support_channels=sorted(support_channels),
        )
        self._next_obstacle_id += 1
        self._emit_event("obstacle_confirmed", {"obstacle": asdict(memory)})
        return memory

    def _start_vertical_probe(self, now: float) -> None:
        origin = (float(self._last_sent_theta), float(self._last_sent_r), float(self._last_sent_z))
        lock = self._vertical_xy_lock_from_pose(origin)
        with self._state._lock:
            self._auto_wrist_base_offset = float(self._state.wrist_offset)
        targets = self._build_vertical_probe_targets(origin[2])
        self._vertical_probe_origin = origin
        self._vertical_probe_xy_lock = lock
        self._vertical_probe_targets = deque(targets)
        self._vertical_probe_target_z = None
        self._vertical_probe_wait_until = float(now)
        self._vertical_probe_debug = []
        self._vertical_opening_pose = None
        source = self._obstacle_memory.source_channel if self._obstacle_memory is not None else self._confirm_source_channel
        self._transition(FSM_PROBE, "obstacle_confirmed_probe")
        self._emit_event(
            "vertical_probe_started",
            {
                "source_channel": None if source is None else int(source),
                "origin": {"theta": origin[0], "r_mm": origin[1], "z_mm": origin[2]},
                "xy_lock": {"x_mm": lock[0], "y_mm": lock[1], "theta": lock[2], "r_mm": lock[3]},
                "targets_z_mm": [float(v) for v in targets],
            },
        )

    def _vertical_xy_lock_from_pose(self, pose: tuple[float, float, float]) -> tuple[float, float, float, float]:
        theta, r_mm, _z = [float(v) for v in pose]
        x_mm, y_mm = polar_to_cartesian(theta, r_mm)
        r_lock = float(math.hypot(x_mm, y_mm))
        theta_lock = float(math.degrees(math.atan2(y_mm, x_mm)))
        if theta_lock < 0.0:
            theta_lock += 360.0
        theta_lock = max(0.0, min(180.0, theta_lock))
        return float(x_mm), float(y_mm), float(theta_lock), float(r_lock)

    def _vertical_probe_pose_for_z(self, z_mm: float) -> tuple[float, float, float]:
        if self._vertical_probe_xy_lock is None:
            if self._vertical_probe_origin is None:
                return float(self._last_sent_theta), float(self._last_sent_r), float(z_mm)
            lock = self._vertical_xy_lock_from_pose(self._vertical_probe_origin)
        else:
            lock = self._vertical_probe_xy_lock
        return float(lock[2]), float(lock[3]), float(z_mm)

    def _build_vertical_probe_targets(self, origin_z_mm: float) -> list[float]:
        origin = float(origin_z_mm)
        step = max(5.0, float(Z_SCAN_STEP_MM))
        max_steps = max(1, int(Z_SCAN_MAX_STEPS))
        start = max(Z_MIN, min(Z_MAX, float(VERTICAL_PROBE_START_Z_MM)))
        stop = max(start, min(float(VERTICAL_PROBE_MAX_Z_MM), float(Z_MAX)))
        targets: list[float] = []
        seen: set[float] = set()
        idx = 0
        while len(targets) < max_steps:
            z = start + idx * step
            idx += 1
            if z > stop + 1e-6:
                break
            key = round(float(z), 3)
            if key in seen or abs(float(z) - origin) <= 1e-6:
                continue
            seen.add(key)
            targets.append(float(z))
        return targets

    def _vertical_probe_step(self, now: float, snap: dict) -> None:
        self._set_tof_active_count(self._tracking_channel_count, reason="vertical_probe")
        if self._obstacle_memory is None:
            self._reset_obstacle_episode("vertical_probe_no_obstacle_memory")
            return

        if self._vertical_probe_origin is None:
            self._start_vertical_probe(now)
            return

        if self._vertical_probe_target_z is None:
            if not self._vertical_probe_targets:
                self._transition(FSM_HOLD, "vertical_probe_no_opening")
                self._emit_event("vertical_probe_exhausted", {"samples": list(self._vertical_probe_debug)})
                with self._state._lock:
                    self._state.last_error = "No vertical opening found with source ToF and CH2/CH3 support clear"
                return
            target_z = float(self._vertical_probe_targets.popleft())
            self._vertical_probe_target_z = target_z
            travel_mm = abs(target_z - float(self._last_sent_z))
            self._vertical_probe_wait_until = float(now + max(0.25, min(0.95, 0.18 + travel_mm / 80.0)))
            pose = self._vertical_probe_pose_for_z(target_z)
            self._send_pose_command(*pose)
            self._emit_event(
                "vertical_probe_target",
                {
                    "theta": pose[0],
                    "r_mm": pose[1],
                    "z_mm": pose[2],
                    "xy_lock": None
                    if self._vertical_probe_xy_lock is None
                    else {
                        "x_mm": self._vertical_probe_xy_lock[0],
                        "y_mm": self._vertical_probe_xy_lock[1],
                        "theta": self._vertical_probe_xy_lock[2],
                        "r_mm": self._vertical_probe_xy_lock[3],
                    },
                    "settle_until": self._vertical_probe_wait_until,
                },
            )
            return

        target_z = float(self._vertical_probe_target_z)
        pose = self._vertical_probe_pose_for_z(target_z)
        self._send_pose_command(*pose)
        if now < self._vertical_probe_wait_until:
            return

        source = int(self._obstacle_memory.source_channel)
        sample = self._sample_channel(snap, source, self._detect_threshold_mm)
        support = self._vertical_support_status(snap)
        source_clear = bool(sample is not None and sample.median_near_mm >= self._detect_threshold_mm + 10.0)
        row = {
            "z_mm": float(target_z),
            "source_channel": int(source),
            "source_clear": bool(source_clear),
            "source_min_mm": None if sample is None else float(sample.min_mm),
            "source_median_mm": None if sample is None else float(sample.median_near_mm),
            "support_ok": bool(support["ok"]),
            "support_clear_mm": float(support["clear_mm"]),
            "support": support["channels"],
        }
        self._vertical_probe_debug.append(row)
        self._vertical_probe_debug = self._vertical_probe_debug[-12:]
        self._emit_event("vertical_probe_sample", row)

        if source_clear and bool(support["ok"]):
            self._vertical_opening_pose = (float(pose[0]), float(pose[1]), float(pose[2]))
            self._vertical_probe_target_z = None
            self._emit_event("vertical_probe_opening", {"pose": {"theta": pose[0], "r_mm": pose[1], "z_mm": pose[2]}, "sample": row})
            self._transition(FSM_PLAN, "vertical_opening_found")
            return

        self._vertical_probe_target_z = None

    def _vertical_support_status(self, snap: dict) -> dict:
        _minima, fresh = self._tof_minima(snap)
        support_channels = [ch for ch in (2, 3) if ch < self._tracking_channel_count]
        clear_mm = max(110.0, min(180.0, 0.55 * float(self._detect_threshold_mm)))
        channels: dict[str, dict] = {}
        ok = True
        for ch in support_channels:
            key = f"tof_ch{ch}"
            sample = self._sample_channel(snap, ch, self._detect_threshold_mm)
            value = None if sample is None else float(sample.median_near_mm)
            channel_ok = bool(fresh.get(key, False) and value is not None and float(value) >= clear_mm)
            channels[key] = {
                "fresh": bool(fresh.get(key, False)),
                "min_mm": None if sample is None else float(sample.min_mm),
                "median_mm": None if value is None else float(value),
                "ok": bool(channel_ok),
            }
            ok = ok and channel_ok
        return {"ok": bool(ok), "clear_mm": float(clear_mm), "channels": channels}

    def _planning_step(self, now: float) -> None:
        self._send_hold_pose()
        if self._obstacle_memory is None:
            self._reset_obstacle_episode("no_obstacle_memory")
            return

        goal = self._current_goal_pose()
        current = (float(self._last_sent_theta), float(self._last_sent_r), float(self._last_sent_z))
        result = self._planner.select(
            current_pose=current,
            goal_pose=goal,
            obstacle=self._obstacle_memory,
            clearance_fn=self._clearance_for_pose,
            opening_pose=self._vertical_opening_pose,
        )
        self._candidate_debug = [
            {
                "name": c.name,
                "accepted": bool(c.accepted),
                "clearance_mm": float(c.clearance_mm),
                "start_clearance_mm": float(c.start_clearance_mm),
                "end_clearance_mm": float(c.end_clearance_mm),
                "egress_accepted": bool(c.egress_accepted),
                "path_cost": float(c.path_cost),
                "reject_reason": str(c.reject_reason),
                "waypoint_count": int(len(c.waypoints)),
            }
            for c in result.candidates
        ]
        if result.selected is None:
            self._transition(FSM_HOLD, result.reason)
            with self._state._lock:
                self._state.last_error = f"No safe avoidance candidate ({result.reason})"
            return

        self._committed_plan = self._plan_dicts(result.selected.waypoints, result.selected.name)
        self._transition(FSM_AVOID, f"selected {result.selected.name}")
        self._emit_event(
            "plan_committed",
            {
                "selected": result.selected.name,
                "clearance_mm": float(result.selected.clearance_mm),
                "egress_accepted": bool(result.selected.egress_accepted),
                "waypoint_count": int(len(self._committed_plan)),
            },
        )

    def _avoid_step(self, now: float) -> None:
        self._set_tof_active_count(self._tracking_channel_count, reason="avoid_tracking")
        if not self._committed_plan:
            if self._profile_side_active:
                self._finish_profile_after_avoidance(now)
            else:
                self._reset_obstacle_episode("avoidance_complete_no_profile")
            return

        target = self._committed_plan[0]
        pose = (
            float(target.get("theta", self._last_sent_theta)),
            float(target.get("r_mm", self._last_sent_r)),
            float(target.get("z_mm", self._last_sent_z)),
        )
        self._send_pose_command(*pose)
        if self._pose_reached(pose):
            self._committed_plan.pop(0)

    def _finish_profile_after_avoidance(self, now: float) -> None:
        self._profile_setpoint_theta = max(
            PROFILE_SWEEP_THETA_MIN_DEG,
            min(PROFILE_SWEEP_THETA_MAX_DEG, float(self._last_sent_theta)),
        )
        if self._profile_target_theta is None:
            self._transition(FSM_TRACK, "avoidance_complete")
            return

        target = float(self._profile_target_theta)
        if abs(float(self._last_sent_theta) - target) <= 1.0:
            self._profile_hold_until = float(now + PROFILE_SWEEP_DWELL_S)
            self._profile_target_theta = (
                PROFILE_SWEEP_THETA_MIN_DEG if target >= 90.0 else PROFILE_SWEEP_THETA_MAX_DEG
            )
            self._emit_event(
                "avoidance_completed_sweep_leg",
                {"theta": float(self._last_sent_theta), "next_target_theta": float(self._profile_target_theta)},
            )
            self._reset_obstacle_episode("sweep_leg_complete_after_avoidance")
            return

        self._transition(FSM_TRACK, "avoidance_complete")

    def _tracking_step(self, now: float) -> None:
        self._set_tof_active_count(self._tracking_channel_count, reason="track_confirm")
        if not self._profile_side_active:
            self._reset_obstacle_episode("tracking_done_no_profile")
            return
        reached = self._run_profile_step(now, tracking=True)
        if reached:
            self._reset_obstacle_episode("sweep_leg_complete")

    def _reset_obstacle_episode(self, reason: str) -> None:
        self._obstacle_memory = None
        self._confirm_source_channel = None
        self._confirm_samples.clear()
        self._committed_plan = []
        self._candidate_debug = []
        self._vertical_probe_targets.clear()
        self._vertical_probe_origin = None
        self._vertical_probe_target_z = None
        self._vertical_probe_wait_until = 0.0
        self._vertical_probe_debug = []
        self._vertical_opening_pose = None
        self._vertical_probe_xy_lock = None
        self._set_tof_active_count(self._detect_channel_count, reason=reason)
        self._transition(FSM_DETECT, reason)
        with self._state._lock:
            self._state.obstacle_response = ObstacleResponse.CLEAR
            self._state.obstacle_source = ""
            self._state.obstacle_dist_mm = -1.0
            if reason != "confirmation_failed":
                self._state.last_error = ""

    def _run_profile_step(self, now: float, tracking: bool) -> bool:
        if not self._profile_side_active:
            return False
        if now < self._profile_hold_until:
            self._send_pose_command(
                self._profile_setpoint_theta,
                PROFILE_SWEEP_R_MM,
                PROFILE_SWEEP_Z_MM,
                wrist_offset=PROFILE_SWEEP_WRIST_OFFSET_DEG,
                wrist_rot=PROFILE_SWEEP_WRIST_ROT_DEG,
                gripper=PROFILE_SWEEP_GRIPPER_DEG,
            )
            return False
        if self._profile_target_theta is None:
            return False

        target = float(self._profile_target_theta)
        delta = target - float(self._profile_setpoint_theta)
        if abs(delta) > 1e-9:
            step = math.copysign(min(abs(delta), PROFILE_SWEEP_STEP_DEG), delta)
            self._profile_setpoint_theta = max(
                PROFILE_SWEEP_THETA_MIN_DEG,
                min(PROFILE_SWEEP_THETA_MAX_DEG, self._profile_setpoint_theta + step),
            )

        reached = abs(float(self._profile_setpoint_theta) - target) <= 1e-6
        self._send_pose_command(
            self._profile_setpoint_theta,
            PROFILE_SWEEP_R_MM,
            PROFILE_SWEEP_Z_MM,
            wrist_offset=PROFILE_SWEEP_WRIST_OFFSET_DEG,
            wrist_rot=PROFILE_SWEEP_WRIST_ROT_DEG,
            gripper=PROFILE_SWEEP_GRIPPER_DEG,
        )

        if reached and abs(float(self._last_sent_theta) - target) <= 1.0:
            self._profile_hold_until = float(now + PROFILE_SWEEP_DWELL_S)
            self._profile_target_theta = (
                PROFILE_SWEEP_THETA_MIN_DEG if target >= 90.0 else PROFILE_SWEEP_THETA_MAX_DEG
            )
            return True
        return False

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
            self._state.theta = max(PROFILE_SWEEP_THETA_MIN_DEG, min(PROFILE_SWEEP_THETA_MAX_DEG, self._state.theta))
            start_theta = float(self._state.theta)
            self._state.last_resp = "Side profile: ON"
            self._state.last_error = ""

        self._profile_setpoint_theta = start_theta
        self._profile_target_theta = PROFILE_SWEEP_THETA_MAX_DEG if start_theta < 90.0 else PROFILE_SWEEP_THETA_MIN_DEG
        self._profile_side_active = True
        self._profile_hold_until = 0.0
        self._reset_obstacle_episode("profile_started")
        self._send_pose_command(
            self._profile_setpoint_theta,
            PROFILE_SWEEP_R_MM,
            PROFILE_SWEEP_Z_MM,
            wrist_offset=PROFILE_SWEEP_WRIST_OFFSET_DEG,
            wrist_rot=PROFILE_SWEEP_WRIST_ROT_DEG,
            gripper=PROFILE_SWEEP_GRIPPER_DEG,
        )

    def _disable_side_profile(self, reason: str) -> None:
        self._profile_side_active = False
        self._profile_target_theta = None
        self._profile_hold_until = 0.0
        self._reset_obstacle_episode(reason)
        with self._state._lock:
            self._state.last_resp = f"Side profile: OFF ({reason})"

    def _handle_action(self, action: str) -> None:
        if action == "profile_side_toggle":
            self._toggle_side_profile()
            return
        if action == "record_toggle":
            self._toggle_recording()
            return
        if action == "tof_threshold_inc":
            self._set_detect_threshold(self._detect_threshold_mm + 50.0)
            return
        if action == "tof_threshold_dec":
            self._set_detect_threshold(self._detect_threshold_mm - 50.0)
            return
        if action == "imu_calibrate":
            self._begin_imu_calibration()
            return
        if action == "tof_view_toggle" and self._tof_plotter is not None:
            self._tof_plotter.toggle_main()
            return
        if action == "tof_screenshot" and self._tof_plotter is not None:
            self._tof_plotter.save_screenshot()
            return

        manual_actions = {
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
            "delta_inc",
            "delta_dec",
            "go_home",
            "set_equil",
        }
        if action not in manual_actions:
            return

        if self._fsm_mode != FSM_DETECT or self._profile_side_active:
            self._disable_side_profile("manual override")

        send_fn = None
        ik_dirty = False
        with self._state._lock:
            st = self._state
            if action == "theta_inc":
                st.theta = min(180.0, st.theta + THETA_STEP)
                ik_dirty = True
            elif action == "theta_dec":
                st.theta = max(0.0, st.theta - THETA_STEP)
                ik_dirty = True
            elif action == "r_inc":
                st.r = min(R_MAX, st.r + R_STEP)
                ik_dirty = True
            elif action == "r_dec":
                st.r = max(R_MIN, st.r - R_STEP)
                ik_dirty = True
            elif action == "z_inc":
                st.z = min(Z_MAX, st.z + Z_STEP)
                ik_dirty = True
            elif action == "z_dec":
                st.z = max(Z_MIN, st.z - Z_STEP)
                ik_dirty = True
            elif action == "wv_inc":
                st.wrist_offset = min(90.0, st.wrist_offset + WRIST_V_STEP)
                ik_dirty = True
            elif action == "wv_dec":
                st.wrist_offset = max(-90.0, st.wrist_offset - WRIST_V_STEP)
                ik_dirty = True
            elif action == "wr_inc":
                st.wrist_rot = min(180, st.wrist_rot + int(WRIST_R_STEP))
                send_fn = self._send_wrist_rot
            elif action == "wr_dec":
                st.wrist_rot = max(0, st.wrist_rot - int(WRIST_R_STEP))
                send_fn = self._send_wrist_rot
            elif action == "grip_open":
                st.gripper = GRIPPER_OPEN
                send_fn = self._send_gripper
            elif action == "grip_close":
                st.gripper = GRIPPER_CLOSE
                send_fn = self._send_gripper
            elif action == "delta_inc":
                st.delta = min(DELTA_MAX, st.delta + 1)
                send_fn = self._send_delta
            elif action == "delta_dec":
                st.delta = max(DELTA_MIN, st.delta - 1)
                send_fn = self._send_delta
            elif action == "go_home":
                send_fn = self._send_home
            elif action == "set_equil":
                st.set_equil_from_current()
                st.last_resp = "Equilibrium saved"
                return

        if ik_dirty:
            with self._state._lock:
                self._send_pose_command(
                    self._state.theta,
                    self._state.r,
                    self._state.z,
                    wrist_offset=self._state.wrist_offset,
                    wrist_rot=self._state.wrist_rot,
                    gripper=self._state.gripper,
                )
        elif send_fn is not None:
            send_fn()

    def _sample_channel(self, snap: dict, ch: int, threshold_mm: float) -> ChannelSample | None:
        if not self._channel_fresh(snap, ch):
            return None
        grids = list(snap.get("grids", []))
        validity = list(snap.get("validity", []))
        if ch >= len(grids):
            return None
        grid = np.asarray(grids[ch], dtype=float)
        if grid.ndim != 2 or grid.size <= 0:
            return None
        val = np.asarray(validity[ch], dtype=int) if ch < len(validity) else np.ones_like(grid, dtype=int)
        if val.shape != grid.shape:
            val = np.ones_like(grid, dtype=int)
        valid = grid[val.astype(bool)]
        valid = valid[np.isfinite(valid)]
        valid = valid[valid > 0.0]
        if valid.size <= 0:
            return None
        nearest = np.sort(valid)[: min(5, valid.size)]
        min_mm = float(nearest[0])
        median_near = float(np.median(nearest))
        points = self._project_channel_points(snap, ch, threshold_mm)
        seq = int(snap.get("seq", [])[ch]) if ch < len(snap.get("seq", [])) else -1
        rx = float(snap.get("last_rx", [])[ch]) if ch < len(snap.get("last_rx", [])) else 0.0
        return ChannelSample(
            channel=int(ch),
            min_mm=float(min_mm),
            median_near_mm=float(median_near),
            points_m=[[float(x) for x in np.asarray(p, dtype=float).reshape(3)] for p in points],
            seq=seq,
            rx_wall=rx,
        )

    def _project_channel_points(self, snap: dict, ch: int, threshold_mm: float) -> list[np.ndarray]:
        grids = list(snap.get("grids", []))
        validity = list(snap.get("validity", []))
        if ch >= len(grids):
            return []
        grid = np.asarray(grids[ch], dtype=float)
        if grid.ndim != 2 or grid.size <= 0:
            return []
        val = np.asarray(validity[ch], dtype=int) if ch < len(validity) else np.ones_like(grid, dtype=int)
        if val.shape != grid.shape:
            val = np.ones_like(grid, dtype=int)

        with self._state._lock:
            joints = [float(x) for x in self._state.joints]
            theta = float(self._last_sent_theta)
            r_mm = float(self._last_sent_r)
            z_mm = float(self._last_sent_z)
        wrist_origin, wrist_rot = self._wrist_pose_base(joints, theta, r_mm, z_mm, use_imu=True, imu_snapshot=snap.get("imu", {}))
        frame = ToFFrameV1(
            seq=int(snap.get("seq", [])[ch]) if ch < len(snap.get("seq", [])) else -1,
            mcu_ms=int(snap.get("mcu_ms", [])[ch]) if ch < len(snap.get("mcu_ms", [])) else -1,
            sensor_id=str(snap.get("sensor_id", [])[ch]) if ch < len(snap.get("sensor_id", [])) else f"ch{ch}",
            mux_channel=int(ch),
            joint_id=str(SENSOR_CONFIG.get("channels", {}).get(int(ch), {}).get("parent_link", "link4")),
            status=int(snap.get("status", [])[ch]) if ch < len(snap.get("status", [])) else 0,
            distances_mm=list(grid.reshape(-1)),
            validity=list(val.reshape(-1)),
            rows=int(grid.shape[0]),
            cols=int(grid.shape[1]),
        )
        cells = frame_cells_to_points_base(frame, wrist_origin, wrist_rot_base=wrist_rot)
        if not cells:
            return []
        cutoff = max(float(threshold_mm) * 1.25, float(threshold_mm) + 60.0)
        pts: list[np.ndarray] = []
        for cell in sorted(cells, key=lambda item: float(item.get("distance_mm", float("inf")))):
            dist = float(cell.get("distance_mm", float("inf")))
            if np.isfinite(dist) and dist <= cutoff:
                pts.append(np.asarray(cell.get("point_base", [0.0, 0.0, 0.0]), dtype=float).reshape(3))
            if len(pts) >= 10:
                break
        return pts

    def _channel_fresh(self, snap: dict, ch: int) -> bool:
        last_rx = list(snap.get("last_rx", []))
        active = list(snap.get("active", []))
        if ch >= len(last_rx):
            return False
        if ch < len(active) and int(active[ch]) == 0:
            return False
        return bool((time.time() - float(last_rx[ch])) <= TOF_FRESHNESS_S)

    def _tof_minima(self, snap: dict) -> tuple[list[float | None], dict[str, bool]]:
        minima: list[float | None] = []
        fresh: dict[str, bool] = {}
        grids = list(snap.get("grids", []))
        validity = list(snap.get("validity", []))
        for ch, grid_raw in enumerate(grids):
            fresh[f"tof_ch{ch}"] = self._channel_fresh(snap, ch)
            if not fresh[f"tof_ch{ch}"]:
                minima.append(None)
                continue
            try:
                grid = np.asarray(grid_raw, dtype=float)
                val = np.asarray(validity[ch], dtype=int) if ch < len(validity) else np.ones_like(grid, dtype=int)
                if val.shape != grid.shape:
                    val = np.ones_like(grid, dtype=int)
                vals = grid[val.astype(bool)]
                vals = vals[np.isfinite(vals)]
                vals = vals[vals > 0.0]
                minima.append(None if vals.size <= 0 else float(np.min(vals)))
            except Exception:
                minima.append(None)
        return minima, fresh

    def _current_goal_pose(self) -> tuple[float, float, float]:
        if self._profile_side_active and self._profile_target_theta is not None:
            return float(self._profile_target_theta), float(PROFILE_SWEEP_R_MM), float(PROFILE_SWEEP_Z_MM)
        with self._state._lock:
            return float(self._state.theta), float(self._state.r), float(self._state.z)

    def _plan_dicts(self, waypoints: list[tuple[float, float, float]], name: str) -> list[dict]:
        out: list[dict] = []
        for idx, (theta, r_mm, z_mm) in enumerate(waypoints):
            wrist_offset = self._auto_wrist_offset_for_pose(theta, r_mm, z_mm)
            joints = self._predict_joint_config(theta, r_mm, z_mm, wrist_offset_deg=wrist_offset)
            eef = self._tool_point_for_pose(theta, r_mm, z_mm)
            out.append(
                {
                    "idx": int(idx),
                    "type": str(name),
                    "theta": float(theta),
                    "r_mm": float(r_mm),
                    "z_mm": float(z_mm),
                    "wrist_offset_deg": float(wrist_offset),
                    "joints_deg": [int(v) for v in joints],
                    "eef_m": [float(v) for v in eef],
                    "waypoint": True,
                }
            )
        return out

    def _clearance_for_pose(self, pose: tuple[float, float, float], obstacle: ObstacleMemory) -> float:
        theta, r_mm, z_mm = [float(v) for v in pose]
        joints = self._predict_joint_config(theta, r_mm, z_mm)
        chain = self._chain_points_for_pose(joints, theta, r_mm, z_mm)
        point = np.asarray(obstacle.centroid_m, dtype=float).reshape(3)
        min_dist_m = float("inf")
        for idx in range(max(0, chain.shape[0] - 1)):
            min_dist_m = min(min_dist_m, self._point_to_segment_distance_m(point, chain[idx], chain[idx + 1]))
        if not np.isfinite(min_dist_m):
            min_dist_m = float(np.linalg.norm(point - self._tool_point_for_pose(theta, r_mm, z_mm)))
        link_radius_mm = 18.0
        return float(min_dist_m * 1000.0 - float(obstacle.radius_mm) - link_radius_mm)

    @staticmethod
    def _point_to_segment_distance_m(point_m: np.ndarray, a_m: np.ndarray, b_m: np.ndarray) -> float:
        p = np.asarray(point_m, dtype=float).reshape(3)
        a = np.asarray(a_m, dtype=float).reshape(3)
        b = np.asarray(b_m, dtype=float).reshape(3)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return float(np.linalg.norm(p - a))
        t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
        return float(np.linalg.norm(p - (a + t * ab)))

    def _pose_reached(self, pose: tuple[float, float, float]) -> bool:
        return (
            abs(float(self._last_sent_theta) - float(pose[0])) <= 0.75
            and abs(float(self._last_sent_r) - float(pose[1])) <= 3.0
            and abs(float(self._last_sent_z) - float(pose[2])) <= 3.0
        )

    def _send_hold_pose(self) -> None:
        with self._state._lock:
            wrist_offset = float(self._state.wrist_offset)
            wrist_rot = int(self._state.wrist_rot)
            gripper = int(self._state.gripper)
        self._send_pose_command(self._last_sent_theta, self._last_sent_r, self._last_sent_z, wrist_offset, wrist_rot, gripper)

    def _send_pose_command(
        self,
        theta: float,
        r_mm: float,
        z_mm: float,
        wrist_offset: float | None = None,
        wrist_rot: int | None = None,
        gripper: int | None = None,
    ) -> None:
        theta = max(0.0, min(180.0, float(theta)))
        r_mm = max(R_MIN, min(R_MAX, float(r_mm)))
        z_mm = max(Z_MIN, min(Z_MAX, float(z_mm)))
        with self._state._lock:
            if wrist_offset is None and self._fsm_mode in {FSM_PROBE, FSM_AVOID}:
                wrist_offset = self._auto_wrist_offset_for_pose(theta, r_mm, z_mm)
            else:
                wrist_offset = float(self._state.wrist_offset if wrist_offset is None else wrist_offset)
            wrist_rot = int(self._state.wrist_rot if wrist_rot is None else wrist_rot)
            gripper = int(self._state.gripper if gripper is None else gripper)

        now = time.monotonic()
        dt = max(1e-3, now - self._last_send_monotonic)
        prev_theta, prev_r, prev_z = self._last_sent_theta, self._last_sent_r, self._last_sent_z
        joints = self._predict_joint_config(theta, r_mm, z_mm, wrist_offset, wrist_rot, gripper)
        cmd = cmd_set_ik_polar(theta, r_mm, z_mm, wrist_offset, wrist_rot, gripper)

        with self._state._lock:
            self._state.theta = float(theta)
            self._state.r = float(r_mm)
            self._state.z = float(z_mm)
            self._state.wrist_offset = float(wrist_offset)
            self._state.wrist_rot = int(wrist_rot)
            self._state.gripper = int(gripper)
            self._state.joints = [int(v) for v in joints]
            self._state.last_cmd = cmd.strip()

        self._bridge.send_cmd(cmd)
        self._last_sent_theta = float(theta)
        self._last_sent_r = float(r_mm)
        self._last_sent_z = float(z_mm)
        self._last_cmd_theta_vel = (self._last_sent_theta - prev_theta) / dt
        self._last_cmd_r_vel = (self._last_sent_r - prev_r) / dt
        self._last_cmd_z_vel = (self._last_sent_z - prev_z) / dt
        self._last_send_monotonic = now

    def _auto_wrist_offset_for_pose(self, theta_deg: float, r_mm: float, z_mm: float) -> float:
        base_offset = float(self._auto_wrist_base_offset)
        x_mm, y_mm = polar_to_cartesian(float(theta_deg), float(r_mm))
        ik = solve_ik(x_mm, y_mm, float(z_mm))
        if ik is None:
            return base_offset
        level = float(wrist_level_angle(float(ik.shoulder), float(ik.elbow)))
        needed = max(0.0, float(AUTO_WRIST_CLEARANCE_MIN_DEG) - level)
        if needed <= 1e-6:
            return base_offset
        return float(base_offset + min(float(AUTO_WRIST_CLEARANCE_MAX_OFFSET_DEG), needed))

    def _predict_joint_config(
        self,
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        wrist_offset_deg: float | None = None,
        wrist_rot_deg: int | None = None,
        gripper_deg: int | None = None,
    ) -> list[int]:
        with self._state._lock:
            fallback = [int(x) for x in self._state.joints]
            wrist_offset = float(self._state.wrist_offset if wrist_offset_deg is None else wrist_offset_deg)
            wrist_rot = int(self._state.wrist_rot if wrist_rot_deg is None else wrist_rot_deg)
            gripper = int(self._state.gripper if gripper_deg is None else gripper_deg)
        x_mm, y_mm = polar_to_cartesian(float(theta_deg), float(r_mm))
        ik = solve_ik(x_mm, y_mm, float(z_mm))
        if ik is None:
            return fallback
        level = wrist_level_angle(float(ik.shoulder), float(ik.elbow))
        return [
            int(ik.base),
            int(ik.shoulder),
            int(ik.elbow),
            int(apply_wrist_offset(level, wrist_offset)),
            int(wrist_rot),
            int(gripper),
        ]

    def _send_wrist_rot(self) -> None:
        with self._state._lock:
            wr = int(self._state.wrist_rot)
            self._state.joints[JOINT_WRIST_ROT] = wr
            cmd = cmd_set_joint(JOINT_WRIST_ROT, wr)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_gripper(self) -> None:
        with self._state._lock:
            g = int(self._state.gripper)
            self._state.joints[JOINT_GRIPPER] = g
            cmd = cmd_set_joint(JOINT_GRIPPER, g)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_delta(self) -> None:
        with self._state._lock:
            cmd = cmd_set_delta(self._state.delta)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_home(self) -> None:
        self._state.restore_equil()
        with self._state._lock:
            theta = float(self._state.theta)
            r_mm = float(self._state.r)
            z_mm = float(self._state.z)
            wrist_offset = float(self._state.wrist_offset)
            wrist_rot = int(self._state.wrist_rot)
            gripper = int(self._state.gripper)
            self._state.last_cmd = "HOME"
            self._state.last_error = ""
        self._bridge.send_cmd(cmd_home())
        self._send_pose_command(theta, r_mm, z_mm, wrist_offset, wrist_rot, gripper)
        self._bridge.send_cmd(cmd_get_pos())

    def _drain_responses(self) -> None:
        while True:
            try:
                resp = self._bridge.response_queue.get_nowait()
            except Exception:
                break
            typ = resp.get("type")
            with self._state._lock:
                self._state.last_resp = str(resp.get("raw", typ))
                if typ == "error":
                    self._state.last_error = str(resp.get("message", resp.get("raw", "")))
                elif typ in {"pos", "ok_all", "ok_ik"} and "positions" in resp:
                    vals = [int(v) for v in resp.get("positions", [])]
                    if len(vals) >= 6:
                        self._state.joints = vals[:6]
                elif typ == "ok_joint":
                    j = int(resp.get("joint", -1))
                    if 0 <= j < len(self._state.joints):
                        self._state.joints[j] = int(resp.get("deg", self._state.joints[j]))

    def _set_detect_threshold(self, value: float) -> None:
        self._detect_threshold_mm = max(40.0, min(3000.0, float(value)))
        self._tof_state.tof_threshold_mm = self._detect_threshold_mm
        for ch in range(self._tracking_channel_count):
            self._tof_state.set_channel_threshold(ch, self._detect_threshold_mm)
        with self._state._lock:
            self._state.last_resp = f"ToF detect threshold: {self._detect_threshold_mm:.0f} mm"

    def _set_tof_active_count(self, count: int, reason: str) -> None:
        count = max(1, min(int(count), self._tracking_channel_count))
        if count == self._tof_active_count:
            return
        self._tof_active_count = count
        set_active_channel_count(count)
        with self._tof_state._lock:
            self._tof_state.configured_channels = count
        self._tof_bridge.set_active_channels(count)
        self._emit_event("tof_active_count", {"count": int(count), "reason": str(reason)})

    def _transition(self, mode: str, reason: str) -> None:
        if mode == self._fsm_mode and reason == self._mode_reason:
            return
        old = self._fsm_mode
        self._fsm_mode = str(mode)
        self._mode_reason = str(reason)
        self._emit_event("mode_transition", {"from": old, "to": self._fsm_mode, "reason": self._mode_reason})

    def _emit_event(self, event_type: str, payload: dict | None = None) -> None:
        self._record_pending_events.append(
            {
                "type": str(event_type),
                "host_monotonic": round(time.monotonic(), 6),
                "payload": {} if payload is None else payload,
            }
        )

    def _sync_state_runtime_fields(self) -> None:
        tof_snap = self._tof_state.snapshot()
        minima, fresh = self._tof_minima(tof_snap)
        memory_count = 1 if self._obstacle_memory is not None else 0
        planner_active = self._fsm_mode in {FSM_CONFIRM, FSM_PROBE, FSM_PLAN, FSM_AVOID, FSM_TRACK, FSM_HOLD}
        debug = {
            "schema": "braccio_session_v2",
            "mode": self._fsm_mode,
            "reason": self._mode_reason,
            "goal": {
                "theta": self._current_goal_pose()[0],
                "r_mm": self._current_goal_pose()[1],
                "z_mm": self._current_goal_pose()[2],
            },
            "command": {
                "theta": float(self._last_sent_theta),
                "r_mm": float(self._last_sent_r),
                "z_mm": float(self._last_sent_z),
            },
            "debug": {
                "severity": 1.0 if planner_active else 0.0,
                "speed_scale": 0.55 if self._fsm_mode in {FSM_CONFIRM, FSM_PROBE, FSM_PLAN, FSM_HOLD} else (0.70 if self._fsm_mode == FSM_AVOID else 1.0),
                "min_clearance_mm": self._candidate_debug[0]["clearance_mm"] if self._candidate_debug else -1.0,
            },
            "candidate_plans": list(self._candidate_debug),
            "vertical_probe": {
                "target_z_mm": self._vertical_probe_target_z,
                "opening_pose": None
                if self._vertical_opening_pose is None
                else {
                    "theta": float(self._vertical_opening_pose[0]),
                    "r_mm": float(self._vertical_opening_pose[1]),
                    "z_mm": float(self._vertical_opening_pose[2]),
                },
                "samples": list(self._vertical_probe_debug),
            },
            "sensor_frame_freshness_by_channel": dict(fresh),
            "sensor_threshold_by_channel_mm": {f"tof_ch{ch}": float(self._detect_threshold_mm) for ch in range(self._tracking_channel_count)},
        }
        self._last_planner_debug = debug
        with self._state._lock:
            self._state.tof_snapshot = tof_snap
            self._state.control_mode = self._fsm_mode
            self._state.planner_active = bool(planner_active)
            self._state.fallback_active = bool(self._fsm_mode == FSM_HOLD)
            self._state.qp_status = ""
            self._state.qp_solve_ms = 0.0
            self._state.predicted_clearance_m = -1.0
            self._state.current_speed_scale = float(debug["debug"]["speed_scale"])
            self._state.profile_active = self._profile_side_active
            self._state.recording_active = self._recording_active
            self._state.obstacle_memory_count = int(memory_count)
            self._state.obstacle_decay_s = 0.0
            self._state.tof_channels_enabled = int(self._tof_active_count or self._tracking_channel_count)
            self._state.future_plan_points = int(len(self._committed_plan))
            self._state.imu_lpf_alpha = float(self._tof_state.imu_lpf_alpha)
            self._state.last_latency_ms = float(self._last_cycle_ms)
            if self._obstacle_memory is not None:
                self._state.obstacle_response = ObstacleResponse.REPLAN
                self._state.obstacle_source = f"tof_ch{self._obstacle_memory.source_channel}"
                self._state.obstacle_dist_mm = float(np.linalg.norm(np.asarray(self._obstacle_memory.centroid_m) - self._tool_point_for_pose(self._last_sent_theta, self._last_sent_r, self._last_sent_z)) * 1000.0)
            elif any(v is not None and float(v) <= self._detect_threshold_mm for v in minima[:2]):
                self._state.obstacle_response = ObstacleResponse.CAUTION
                self._state.obstacle_source = ""
                self._state.obstacle_dist_mm = min(float(v) for v in minima[:2] if v is not None)

    def _update_state_from_tof_snapshot(self, snap: dict) -> None:
        with self._state._lock:
            self._state.tof_snapshot = snap
            self._state.imu_lpf_alpha = float(snap.get("imu", {}).get("lpf_alpha", self._state.imu_lpf_alpha))

    def _init_kinematics(self) -> None:
        try:
            path = sensor_urdf_path()
            self._kinematics = BraccioURDFKinematics(path) if path.exists() else None
        except Exception:
            self._kinematics = None

    def _wrist_pose_base(
        self,
        joints_deg: list[float],
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        use_imu: bool = True,
        imu_snapshot: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._kinematics is not None:
            try:
                tf = self._kinematics.link_transform([float(v) for v in joints_deg], "link4")
                origin = np.asarray(tf[:3, 3], dtype=float).reshape(3)
                rot = np.asarray(tf[:3, :3], dtype=float).reshape(3, 3)
                if use_imu:
                    rot = self._imu_correct_wrist_rotation(rot, imu_snapshot=imu_snapshot)
                return origin, rot
            except Exception:
                pass
        origin = self._tool_point_for_pose(theta_deg, r_mm, z_mm)
        rot = _rot_zyx_deg(float(theta_deg), self._arm_pitch_deg(r_mm, z_mm), 0.0)
        if use_imu:
            rot = self._imu_correct_wrist_rotation(rot, imu_snapshot=imu_snapshot)
        return origin, rot

    def _chain_points_for_pose(self, joints_deg: list[int], theta_deg: float, r_mm: float, z_mm: float) -> np.ndarray:
        if self._kinematics is not None:
            try:
                return self._kinematics.chain_positions_m([float(v) for v in joints_deg])
            except Exception:
                pass
        base = np.array([0.0, 0.0, 0.0], dtype=float)
        elbow = self._tool_point_for_pose(theta_deg, max(R_MIN, r_mm * 0.5), z_mm * 0.5)
        wrist = self._tool_point_for_pose(theta_deg, r_mm, z_mm)
        return np.vstack([base, elbow, wrist])

    @staticmethod
    def _tool_point_for_pose(theta_deg: float, r_mm: float, z_mm: float) -> np.ndarray:
        th = math.radians(float(theta_deg))
        return np.array(
            [float(r_mm) * math.cos(th), float(r_mm) * math.sin(th), float(z_mm)],
            dtype=float,
        ) / 1000.0

    @staticmethod
    def _point_to_polar_mm(point_m: np.ndarray) -> tuple[float, float, float]:
        p = np.asarray(point_m, dtype=float).reshape(3)
        theta = math.degrees(math.atan2(float(p[1]), float(p[0])))
        if theta < 0.0:
            theta += 360.0
        theta = max(0.0, min(180.0, theta))
        r_mm = math.hypot(float(p[0]), float(p[1])) * 1000.0
        z_mm = float(p[2]) * 1000.0
        return float(theta), float(r_mm), float(z_mm)

    @staticmethod
    def _arm_pitch_deg(r_mm: float, z_mm: float) -> float:
        return math.degrees(math.atan2(float(z_mm), max(abs(float(r_mm)), 1e-3)))

    def _imu_rotation_from_snapshot(self, imu_snapshot: dict | None = None) -> np.ndarray | None:
        snap = imu_snapshot
        if snap is None:
            snap = self._tof_state.snapshot().get("imu", {})
        if not isinstance(snap, dict) or not bool(snap.get("online", False)):
            return None
        if not bool(snap.get("orientation_ready", False)):
            return None
        if (time.time() - float(snap.get("last_rx", 0.0))) > TOF_FRESHNESS_S:
            return None
        mat = np.asarray(snap.get("rot_ref_from_imu", []), dtype=float)
        if mat.size != 9:
            return None
        try:
            return _orthonormalize_rot(mat.reshape(3, 3))
        except Exception:
            return None

    def _update_imu_reference_anchor(self, wrist_rot_fk: np.ndarray, imu_snapshot: dict | None = None, force: bool = False) -> bool:
        with self._imu_cal_lock:
            if self._imu_ref_anchor_valid and not force:
                return True
            rot_wrist_from_imu = self._imu_rot_wrist_from_imu.copy()
        rot_ref_from_imu = self._imu_rotation_from_snapshot(imu_snapshot=imu_snapshot)
        if rot_ref_from_imu is None:
            return False
        rot_ref_from_wrist = rot_ref_from_imu @ rot_wrist_from_imu.T
        rot_base_from_ref = np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3) @ rot_ref_from_wrist.T
        with self._imu_cal_lock:
            self._imu_rot_base_from_ref = _orthonormalize_rot(rot_base_from_ref)
            self._imu_ref_anchor_valid = True
        return True

    def _imu_correct_wrist_rotation(self, wrist_rot_fk: np.ndarray, imu_snapshot: dict | None = None) -> np.ndarray:
        with self._imu_cal_lock:
            calibrated = bool(self._imu_calibrated)
            rot_wrist_from_imu = self._imu_rot_wrist_from_imu.copy()
            rot_base_from_ref = self._imu_rot_base_from_ref.copy()
            anchor_valid = bool(self._imu_ref_anchor_valid)
            fuse_weight = max(0.0, min(1.0, float(self._imu_fuse_weight)))
            max_angle = math.radians(max(1.0, float(self._imu_max_tilt_correction_deg)))
        if not calibrated or fuse_weight <= 1e-6:
            return np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3)

        rot_ref_from_imu = self._imu_rotation_from_snapshot(imu_snapshot=imu_snapshot)
        if rot_ref_from_imu is not None and (anchor_valid or self._update_imu_reference_anchor(wrist_rot_fk, imu_snapshot=imu_snapshot)):
            with self._imu_cal_lock:
                rot_base_from_ref = self._imu_rot_base_from_ref.copy()
            rot_ref_from_wrist = rot_ref_from_imu @ rot_wrist_from_imu.T
            wrist_rot_est = _orthonormalize_rot(rot_base_from_ref @ rot_ref_from_wrist)
            rot_err = wrist_rot_est @ np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3).T
            axis_ref, angle = _rotmat_to_axis_angle(rot_err)
            angle = min(max_angle, max(0.0, angle * fuse_weight))
            return _orthonormalize_rot(_rot_axis_angle(axis_ref, angle) @ wrist_rot_fk)

        snap = imu_snapshot if isinstance(imu_snapshot, dict) else self._tof_state.snapshot().get("imu", {})
        if (not bool(snap.get("online", False))) or ((time.time() - float(snap.get("last_rx", 0.0))) > TOF_FRESHNESS_S):
            return np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3)
        a_imu = _normalize_vec(
            np.array(
                [float(snap.get("ax_g", 0.0)), float(snap.get("ay_g", 0.0)), float(snap.get("az_g", 0.0))],
                dtype=float,
            )
        )
        if float(np.linalg.norm(a_imu)) <= 1e-6:
            return np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3)
        g_meas_wrist = _normalize_vec(rot_wrist_from_imu @ a_imu)
        g_fk_wrist = _normalize_vec(np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3).T @ np.array([0.0, 0.0, -1.0], dtype=float))
        dot = max(-1.0, min(1.0, float(np.dot(g_meas_wrist, g_fk_wrist))))
        axis = np.cross(g_meas_wrist, g_fk_wrist)
        axis_n = float(np.linalg.norm(axis))
        if axis_n <= 1e-9:
            return np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3)
        angle = min(max_angle, math.atan2(axis_n, dot) * fuse_weight)
        return np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3) @ _rot_axis_angle(axis, angle)

    def _begin_imu_calibration(self) -> None:
        if not self._teensy_port or not self._tof_bridge.is_open():
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
            self._imu_ref_anchor_valid = False
        self._tof_state.set_imu_calibrated(False)
        with self._state._lock:
            self._state.last_resp = "IMU calibration started"
            self._state.last_error = ""
        th = threading.Thread(target=self._imu_calibration_worker, daemon=True, name="imu-calibration-worker")
        with self._imu_cal_lock:
            self._imu_calib_thread = th
        th.start()

    def _imu_calibration_worker(self) -> None:
        ok = False
        msg = ""
        try:
            ok, msg = self._run_imu_calibration()
        except Exception as exc:
            msg = f"IMU calibration failed: {exc}"
        with self._imu_cal_lock:
            self._imu_calibrating = False
        with self._state._lock:
            if ok:
                self._state.last_resp = msg
                self._state.last_error = ""
            else:
                self._state.last_error = msg

    def _run_imu_calibration(self) -> tuple[bool, str]:
        with self._state._lock:
            restore = (
                float(self._state.theta),
                float(self._state.r),
                float(self._state.z),
                float(self._state.wrist_offset),
                int(self._state.wrist_rot),
                int(self._state.gripper),
            )
        poses = [
            (90.0, 112.0, 20.0, 0.0, 90, 73),
            (90.0, 140.0, 95.0, -20.0, 90, 73),
            (90.0, 100.0, 5.0, 20.0, 90, 73),
            (40.0, 118.0, 35.0, 0.0, 90, 73),
            (140.0, 118.0, 35.0, 0.0, 90, 73),
        ]
        src_accel: list[np.ndarray] = []
        dst_grav: list[np.ndarray] = []
        for pose in poses:
            self._send_calibration_pose(*pose, settle_s=0.8)
            a_avg = self._collect_imu_accel_avg(duration_s=0.55, min_samples=12)
            if a_avg is None:
                continue
            joints = self._predict_joint_config(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
            _, rot_fk = self._wrist_pose_base(joints, pose[0], pose[1], pose[2], use_imu=False)
            g_wrist = _normalize_vec(rot_fk.T @ np.array([0.0, 0.0, -1.0], dtype=float))
            a_norm = _normalize_vec(a_avg)
            if float(np.linalg.norm(g_wrist)) > 1e-6 and float(np.linalg.norm(a_norm)) > 1e-6:
                src_accel.append(a_norm)
                dst_grav.append(g_wrist)
        self._send_calibration_pose(*restore, settle_s=0.6)
        if len(src_accel) < 3:
            with self._imu_cal_lock:
                self._imu_calibrated = False
                self._imu_ref_anchor_valid = False
                self._imu_cal_rms_deg = 0.0
            self._tof_state.set_imu_calibrated(False)
            return False, "IMU calibration failed: insufficient valid samples"
        rot_wrist_from_imu = self._solve_wahba_rotation(src_accel, dst_grav)
        errs = []
        for a_norm, g_ref in zip(src_accel, dst_grav):
            g_hat = _normalize_vec(rot_wrist_from_imu @ a_norm)
            c = max(-1.0, min(1.0, float(np.dot(g_hat, g_ref))))
            errs.append(math.degrees(math.acos(c)))
        rms = float(math.sqrt(sum(e * e for e in errs) / max(1, len(errs))))
        with self._imu_cal_lock:
            self._imu_rot_wrist_from_imu = rot_wrist_from_imu
            self._imu_calibrated = True
            self._imu_cal_rms_deg = rms
            self._imu_ref_anchor_valid = False
        self._tof_state.set_imu_calibrated(True)
        with self._state._lock:
            joints = [float(x) for x in self._state.joints]
            theta = float(self._state.theta)
            r_mm = float(self._state.r)
            z_mm = float(self._state.z)
        _, rot_fk = self._wrist_pose_base(joints, theta, r_mm, z_mm, use_imu=False)
        self._update_imu_reference_anchor(rot_fk, imu_snapshot=self._tof_state.snapshot().get("imu", {}), force=True)
        return True, f"IMU calibration complete (rms={rms:.1f} deg, samples={len(src_accel)})"

    def _send_calibration_pose(
        self,
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        wrist_offset_deg: float,
        wrist_rot_deg: int,
        gripper_deg: int,
        settle_s: float,
    ) -> None:
        self._send_pose_command(theta_deg, r_mm, z_mm, wrist_offset_deg, wrist_rot_deg, gripper_deg)
        end_t = time.monotonic() + max(0.1, float(settle_s))
        while time.monotonic() < end_t:
            self._drain_responses()
            time.sleep(0.02)

    def _collect_imu_accel_avg(self, duration_s: float, min_samples: int) -> np.ndarray | None:
        deadline = time.monotonic() + max(0.1, float(duration_s))
        samples: list[np.ndarray] = []
        last_seq = -10**9
        while time.monotonic() < deadline:
            snap = self._tof_state.snapshot().get("imu", {})
            seq = int(snap.get("seq", -1))
            if bool(snap.get("online", False)) and (time.time() - float(snap.get("last_rx", 0.0))) <= TOF_FRESHNESS_S and seq != last_seq:
                a = np.array([float(snap.get("ax_g", 0.0)), float(snap.get("ay_g", 0.0)), float(snap.get("az_g", 0.0))])
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
            h += np.outer(_normalize_vec(dst), _normalize_vec(src))
        u, _, vt = np.linalg.svd(h)
        r = u @ vt
        if float(np.linalg.det(r)) < 0.0:
            u[:, -1] *= -1.0
            r = u @ vt
        return r

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
        self._record_pending_events.clear()
        self._record_start_wall = time.time()
        self._recording_active = True
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

    def _wait_for_save_completion(self, timeout_s: float = 0.0) -> None:
        with self._save_lock:
            th = self._save_thread
        if th is not None and th.is_alive():
            th.join(timeout=max(0.0, float(timeout_s)))

    def _save_recording_lines(self, lines: list[str], start_wall: float, end_wall: float) -> str | None:
        if not lines:
            return None
        end_stamp = datetime.fromtimestamp(end_wall).strftime("%Y%m%d_%H%M%S")
        session_dir = Path("logs") / "sessions" / f"session_{end_stamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        data_path = session_dir / "session.jsonl"
        with data_path.open("w", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line)
                fh.write("\n")
        meta = {
            "schema": "braccio_session_v2",
            "start_wall": float(start_wall),
            "end_wall": float(end_wall),
            "duration_s": max(0.0, float(end_wall - start_wall)),
            "samples": int(len(lines)),
            "tof_channels_enabled": int(self._tracking_channel_count),
            "detect_channel_count": int(self._detect_channel_count),
            "tracking_channel_count": int(self._tracking_channel_count),
            "detect_threshold_mm": float(self._detect_threshold_mm),
            "confirm_frames": int(self._confirm_frames),
            "confirm_hits": int(self._confirm_hits),
            "planner_model": "vertical_probe_sector_waypoints_v2",
            "planner_clearance_mm": float(self._planner.safe_clearance_mm),
            "sweep_profile": {
                "theta_min_deg": float(PROFILE_SWEEP_THETA_MIN_DEG),
                "theta_max_deg": float(PROFILE_SWEEP_THETA_MAX_DEG),
                "r_mm": float(PROFILE_SWEEP_R_MM),
                "z_mm": float(PROFILE_SWEEP_Z_MM),
                "step_deg": float(PROFILE_SWEEP_STEP_DEG),
                "dwell_s": float(PROFILE_SWEEP_DWELL_S),
            },
            "imu_lpf_alpha": float(self._tof_state.imu_lpf_alpha),
            "imu_calibrated": bool(self._imu_calibrated),
            "imu_calibration_rms_deg": float(self._imu_cal_rms_deg),
            "imu_rot_wrist_from_imu": np.asarray(self._imu_rot_wrist_from_imu, dtype=float).reshape(3, 3).tolist(),
            "urdf_path": str(sensor_urdf_path()),
        }
        (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return str(data_path)

    def _record_tick(self, now: float, tof_snap: dict) -> None:
        if not self._recording_active:
            return
        minima, fresh = self._tof_minima(tof_snap)
        events = list(self._record_pending_events)
        self._record_pending_events.clear()
        with self._state._lock:
            joints = [int(x) for x in self._state.joints]
            eef = self._tool_point_for_pose(self._state.theta, self._state.r, self._state.z)
            obstacle_rows = []
            if self._obstacle_memory is not None:
                obstacle_rows.append(
                    {
                        "x_m": float(self._obstacle_memory.centroid_m[0]),
                        "y_m": float(self._obstacle_memory.centroid_m[1]),
                        "z_m": float(self._obstacle_memory.centroid_m[2]),
                        "radius_mm": float(self._obstacle_memory.radius_mm),
                        "source": f"tof_ch{self._obstacle_memory.source_channel}",
                        "confidence": float(self._obstacle_memory.confidence),
                        "hits": int(self._obstacle_memory.sample_count),
                        "opacity": 0.85,
                    }
                )
            row = {
                "schema": "braccio_session_v2",
                "host_monotonic": round(now, 6),
                "wall_time": time.time(),
                "theta": round(float(self._state.theta), 3),
                "r_mm": round(float(self._state.r), 3),
                "z_mm": round(float(self._state.z), 3),
                "wrist_offset_deg": round(float(self._state.wrist_offset), 3),
                "wrist_rot_deg": int(self._state.wrist_rot),
                "gripper_deg": int(self._state.gripper),
                "joints_deg": joints,
                "eef_m": [float(eef[0]), float(eef[1]), float(eef[2])],
                "last_sent": {
                    "theta": round(float(self._last_sent_theta), 3),
                    "r_mm": round(float(self._last_sent_r), 3),
                    "z_mm": round(float(self._last_sent_z), 3),
                },
                "cmd_velocity": {
                    "theta_deg_s": float(self._last_cmd_theta_vel),
                    "r_mm_s": float(self._last_cmd_r_vel),
                    "z_mm_s": float(self._last_cmd_z_vel),
                },
                "mode": self._fsm_mode,
                "planner_mode": self._fsm_mode,
                "planner_active": bool(self._fsm_mode != FSM_DETECT),
                "obstacle": {
                    "response": self._state.obstacle_response,
                    "source": self._state.obstacle_source,
                    "distance_mm": float(self._state.obstacle_dist_mm),
                },
                "obstacles": obstacle_rows,
                "future_plan": list(self._committed_plan),
                "tof_min_by_ch_mm": minima,
                "sensor_frame_freshness_by_channel": dict(fresh),
                "threshold_by_ch_mm": [float(self._detect_threshold_mm)] * self._tracking_channel_count,
                "tof_active_count": int(self._tof_active_count),
                "events": events,
                "planner_debug": dict(self._last_planner_debug),
                "imu": {
                    "online": bool(tof_snap.get("imu", {}).get("online", False)),
                    "calibrated": bool(self._imu_calibrated),
                    "lpf_alpha": float(tof_snap.get("imu", {}).get("lpf_alpha", 0.25)),
                    "ax_g": float(tof_snap.get("imu", {}).get("ax_g", 0.0)),
                    "ay_g": float(tof_snap.get("imu", {}).get("ay_g", 0.0)),
                    "az_g": float(tof_snap.get("imu", {}).get("az_g", 0.0)),
                    "gx_dps": float(tof_snap.get("imu", {}).get("gx_dps", 0.0)),
                    "gy_dps": float(tof_snap.get("imu", {}).get("gy_dps", 0.0)),
                    "gz_dps": float(tof_snap.get("imu", {}).get("gz_dps", 0.0)),
                    "orientation_ready": bool(tof_snap.get("imu", {}).get("orientation_ready", False)),
                    "rms_deg": float(self._imu_cal_rms_deg),
                },
                "timing": {"control_cycle_ms": float(self._last_cycle_ms)},
                "profile_active": bool(self._profile_side_active),
            }
            self._state.record_samples += 1
        self._record_buffer.append(json.dumps(row, separators=(",", ":")))
