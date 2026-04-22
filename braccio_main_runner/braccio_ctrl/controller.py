"""
controller.py — Main control loop.  Integrates all modules.

BraccioController.run() is the entry point.  It wraps the curses session
and drives the main loop:

  1. Drain Arduino response queue → update ArmState
  2. Read keyboard → get action string
  3. Dispatch action → mutate state → send serial command
  4. Render display

The IK path (theta/r/z/wrist changes) recomputes all joint angles on
every relevant keypress and sends a single SET ALL command.
Independent axes (wrist rotation, gripper, slew rate) send targeted
single-joint or DELTA commands.

ToF / IR Integration:
  - ToF sensors (VL53L5CX * 4) on separate Teensy via tof_sensor.py
  - IR proximity (OUT1D, 2-bit) for last-resort obstacle detection
  - Obstacle detected by ToF → replan trajectory
  - Obstacle detected by IR (ToF missed) → back away immediately
"""

import atexit
import curses
import logging
import queue
import signal
import sys
import time

# ── Display adapters ──────────────────────────────────────────────────────
#
# The legacy CursesDisplay.render() signature expects `sweep_snap` and
# `obs_snap` dicts with specific keys produced by AutoSweeper.get_status()
# and ObstacleMap.snapshot(). We keep the display contract stable by
# repackaging the SafetyAPI snapshot into the same shape.


def _legacy_sweep_snap(safety_snap: dict) -> dict:
    """Repackage SafetyAPI snapshot into the legacy sweep-snap shape."""
    mode = safety_snap.get("mode", "idle")
    bt_state = safety_snap.get("bt_state", "idle")
    return {
        "running": mode == "sweep",
        "state": bt_state,
        "direction": int(safety_snap.get("sweep_direction", +1)),
        "current_theta": float(safety_snap.get("sweep_target_deg", 90.0)),
        "sweep_r": 152.0,
        "sweep_z": 35.0,
        "bypass_theta": None,
        "last_strategy": safety_snap.get("last_strategy", ""),
        "last_failure": safety_snap.get("last_failure"),
    }


def _legacy_obs_snap(safety_snap: dict) -> dict:
    """Repackage SafetyAPI world snapshot into the legacy obstacle-snap shape."""
    world = safety_snap.get("world", {}) or {}
    return {
        "num_points": int(world.get("num_points", 0)),
        "centroid": None,
        "kalman_estimate": None,
        "kalman_uncertainty_mm": -1.0,
        "has_active": world.get("num_points", 0) > 0,
        "last_obs_age_s": float(world.get("oldest_age_s", 0.0)),
        "memory": {
            "num_cells": int(world.get("grid_cells_occ", 0) +
                             world.get("grid_cells_free", 0)),
            "num_occupied_cells": int(world.get("grid_cells_occ", 0)),
            "cell_mm": 30.0,
            "occupied_threshold": 0.4,
        },
    }

from .arm_state import ArmState
from .backends import BraccioBackend, HardwareBackend
from .ik_solver import solve_ik, polar_to_cartesian, fk_polar
from .keyboard_handler import KeyboardHandler
from .display import CursesDisplay
from .state_library import StateLibrary
from .tof_sensor import ToFState, ToFBridge, ObstacleResponse
from .imu_state import IMUState
from .data_publisher import DataPublisher
from .session_logger import SessionLogger
from .rl_recorder import RLRecorder
from .safety import (
    RefusalDialog,
    SafetyAPI,
    Waypoint,
    WaypointType,
    render_to_curses,
)
from .protocol import (
    cmd_set_all,
    cmd_set_joint,
    cmd_set_delta,
    cmd_home,
    cmd_get_pos,
)
from .constants import (
    THETA_STEP,
    R_STEP,
    Z_STEP,
    WRIST_V_STEP,
    WRIST_R_STEP,
    GRIPPER_OPEN,
    GRIPPER_CLOSE,
    DELTA_MIN,
    DELTA_MAX,
    JOINT_WRIST_ROT,
    JOINT_GRIPPER,
    R_MIN,
    R_MAX,
    Z_MIN,
    Z_MAX,
    BAUD_RATE,
    TOF_NUM_CHANNELS,
    TOF_BAUD_RATE,
    TOF_THRESHOLD_MM,
    TOF_THRESHOLDS_MM,
    TOF_DEFAULT_PORT,
    HOME_POS,
    STRICT_MODE_DEFAULT,
    KEY_BINDINGS,
)


class BraccioController:
    """
    Top-level controller.

    Parameters
    ----------
    port         : serial port device path for Braccio arm, e.g. '/dev/ttyACM0'
                   (ignored if `arm_backend` is provided)
    baud         : baud rate (default 115200, must match Arduino sketch)
    teensy_port  : serial port for ToF/IR Teensy (None = no ToF)
    teensy_baud  : baud rate for Teensy (default 115200)
    arm_backend  : pre-built BraccioBackend. When None, a HardwareBackend
                   (pyserial) is built from `port`/`baud`. The digital twin
                   passes a SimBackend here so the same control loop drives
                   PyBullet instead of a physical Arduino.
    sensor_backend: pre-built ToF/IR/IMU reader. When None, a hardware
                   ToFBridge is built. The sim passes a reader that pushes
                   ray-cast results into ToFState/IMUState directly.
    tof_state    : optional pre-built ToFState. Defaults to a fresh one.
    imu_state    : optional pre-built IMUState. Defaults to a fresh one.
    """

    def __init__(
        self,
        port: str,
        baud: int = BAUD_RATE,
        teensy_port: str = TOF_DEFAULT_PORT,
        teensy_baud: int = TOF_BAUD_RATE,
        enable_ir: bool = True,
        arm_backend: BraccioBackend | None = None,
        sensor_backend=None,
        tof_state: "ToFState | None" = None,
        imu_state: "IMUState | None" = None,
        rl_policy_path: str | None = None,
    ):
        # Arm backend — either injected (sim) or built from port (hardware).
        self._bridge: BraccioBackend = arm_backend or HardwareBackend(port, baud)
        self._state = ArmState()
        self._state_lib = StateLibrary()
        self._publisher = DataPublisher()
        self._stdscr = None  # set inside _curses_main

        # ── ToF / IR subsystem ────────────────────────────────────────────
        self._tof_state = tof_state or ToFState(num_channels=TOF_NUM_CHANNELS)
        self._tof_state.tof_threshold_mm = TOF_THRESHOLDS_MM[0]
        self._tof_state.tof_thresholds_mm = list(TOF_THRESHOLDS_MM)
        # IR integration can be disabled at launch with --no-ir. The ToF
        # reader still receives IR lines but forces the debounced bits to 0,
        # so neither obstacle_response nor the display path ever reacts.
        self._tof_state.ir_enabled = bool(enable_ir)
        self._imu_state = imu_state or IMUState()
        self._tof_bridge = sensor_backend or ToFBridge(self._tof_state, self._imu_state)
        self._teensy_port = teensy_port
        self._teensy_baud = teensy_baud

        # ── Safety stack (replaces legacy MotionGuard/AutoSweeper) ───────
        # Single integration surface for point-cloud + KDTree + capsule
        # collision + BiRRT + cascading replanner + py_trees orchestration.
        self._safety = SafetyAPI(send_cmd=self._send_safety_command)
        self._safety.set_strict_mode(STRICT_MODE_DEFAULT)
        self._refusal_dialog = RefusalDialog()
        self._shutdown_done = False
        # Install an atexit + SIGINT/SIGTERM trap so the arm parks at
        # HOME (slowest slew) whenever the process exits for ANY reason —
        # KeyboardInterrupt, terminal close, kill, uncaught exception.
        # The original jitter-on-restart symptom came from the servos
        # keeping old PWM targets through an unclean exit.
        atexit.register(self._graceful_shutdown)
        try:
            signal.signal(signal.SIGTERM, self._signal_shutdown)
            signal.signal(signal.SIGHUP,  self._signal_shutdown)
        except (AttributeError, ValueError):
            # SIGHUP is POSIX-only and signal.signal only works on the
            # main thread; either failure is non-fatal.
            pass

        # ── Session telemetry logger (JSONL for offline review) ─────────
        self._logger = SessionLogger()
        self._logger.set_sources(
            arm_state=self._state,
            tof_state=self._tof_state,
            safety=self._safety,
        )
        # ── RL transition recorder (writes logs/rl_transitions_*.npz) ───
        # Stage 0 data collection — see Data_Collection_Guide.pdf. No
        # obstacle_map arg since the legacy ObstacleMap was removed in
        # the safety-stack refactor; rl_recorder tolerates None.
        self._rl_recorder = RLRecorder(
            arm_state=self._state,
            tof_state=self._tof_state,
            imu_state=self._imu_state,
            obstacle_map=None,
        )
        # RL deployment (optional — only active when --rl-policy is passed)
        self._rl_policy_path = rl_policy_path
        self._rl_sweeper = None          # built lazily in _curses_main

        # Track BT state transitions for edge-triggered events
        self._prev_bt_mode = "idle"
        self._prev_obs_response = "clear"

    # ── Entry point ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the curses session and start the main loop."""
        curses.wrapper(self._curses_main)

    # ── curses main ───────────────────────────────────────────────────────

    def _curses_main(self, stdscr) -> None:
        self._stdscr = stdscr
        kb = KeyboardHandler(stdscr)
        display = CursesDisplay(stdscr)

        # Connect to Arduino (Braccio arm)
        self._connect()

        # Connect to Teensy (ToF / IR sensors)
        if self._teensy_port:
            ok = self._tof_bridge.connect(self._teensy_port, self._teensy_baud)
            if not ok:
                detail = self._tof_bridge._last_open_error or (
                    f"cannot open {self._teensy_port} "
                    f"(try: python tof_serial_diagnose.py {self._teensy_port})"
                )
                with self._state._lock:
                    self._state.last_error = f"ToF Teensy: {detail}"

        # Start session telemetry logger (writes session_logs/session_*.jsonl)
        try:
            log_path = self._logger.start()
            with self._state._lock:
                self._state.last_resp = f"log: {log_path}"
        except Exception as exc:
            with self._state._lock:
                self._state.last_error = f"logger: {exc}"

        # Wait for the Arduino's first POS response so we never plan from
        # the stale software default while the real arm sits somewhere
        # else — that mismatch is what causes the "jerks on restart"
        # behaviour the user reported.
        self._await_pos_sync(timeout_s=1.5)

        # Give the Teensy a brief window to stream its first ToF frames so
        # the recorder's initial ticks see real grids instead of the
        # ``grids=[None]*4 → np.ones`` sentinel. Best-effort only — if no
        # Teensy is configured this just sleeps 0.3 s and moves on.
        if self._teensy_port:
            import time as _t
            _deadline = _t.monotonic() + 0.8
            while _t.monotonic() < _deadline:
                snap = self._tof_state.snapshot()
                if any(snap.get("frame_cnt", [])):
                    break
                _t.sleep(0.05)

        # Start RL transition recorder AFTER polar + ToF are warm. Otherwise
        # the first ~15 transitions would record stale defaults
        # (theta=90, r=152, z=-50) and ones-valued ToF grids — polluting
        # calibrate_noise's early-sample statistics.
        try:
            self._rl_recorder.start()
        except Exception as exc:
            with self._state._lock:
                self._state.last_error = f"rl_recorder: {exc}"

        # Start the safety stack's behavior tree daemon.
        self._safety.start()

        # If a trained RL policy was passed via --rl-policy, build an RLSweeper
        # so the Z key drives the RL policy instead of the safety BT sweep.
        if self._rl_policy_path:
            try:
                from stable_baselines3 import SAC
                from .rl_sweeper import AtomicPolicyRef, RLSweeper
                _policy = SAC.load(self._rl_policy_path)
                self._rl_policy_ref = AtomicPolicyRef(_policy)
                self._rl_sweeper = RLSweeper(
                    arm_state=self._state,
                    bridge=self._bridge,
                    tof_state=self._tof_state,
                    obstacle_map=None,
                    policy_ref=self._rl_policy_ref,
                )
                with self._state._lock:
                    self._state.last_resp = f"RL policy ready: {self._rl_policy_path}"
            except Exception as exc:
                with self._state._lock:
                    self._state.last_error = f"RL policy load failed: {exc}"

        # Auto-calibrate the IMU before the main loop can ingest any ToF
        # frames. Without this, the cached rotation matrix defaults to an
        # uncalibrated identity and world-frame obstacle projections drift
        # — which, among other things, makes HOME-via-sequence commands
        # look like they "succeed" at the Arduino (ok_all) without the
        # arm actually reaching a clean 90° pose.
        if self._imu_state.wait_for_frame(timeout_s=2.0):
            self._run_imu_calibration()
        else:
            with self._state._lock:
                self._state.last_error = (
                    "IMU: no frames within 2 s — running uncalibrated "
                    "(world-frame obstacle projections may drift)"
                )

        # Main loop (~5 Hz, paced by keyboard halfdelay)
        while True:
            self._drain_responses()
            self._check_obstacle()
            self._push_sensor_to_safety()
            self._log_edge_events()

            # Only surface the refusal dialog when strict mode is on;
            # otherwise the manual branch auto-replans silently.
            if (
                self._safety.strict_mode_enabled()
                and not self._refusal_dialog.is_open
            ):
                refusal = self._safety.consume_refusal()
                if refusal is not None:
                    self._refusal_dialog.open(refusal)

            if self._refusal_dialog.is_open:
                # Consume F1–F5 / ESC directly for the dialog, but do NOT
                # block the rest of the controller — the dialog is purely
                # informational when strict-mode is enabled and the user
                # should still be able to toggle strict off ('t'), drive
                # the arm, or quit. Non-dialog keys fall through to the
                # normal action path below.
                raw_key = kb.get_raw_key()
                if raw_key is not None:
                    if int(raw_key) in self._DIALOG_RAW_KEY_MAP:
                        self._dispatch_refusal_raw_key(raw_key)
                    else:
                        # Translate the raw key back to an action so it can
                        # flow through the usual dispatch (including 't'
                        # which will dismiss the dialog by toggling strict
                        # mode off — see _handle_action).
                        action = KEY_BINDINGS.get(int(raw_key))
                        if action == "quit":
                            break
                        if action == "strict_toggle":
                            self._refusal_dialog.close()
                        if action is not None:
                            self._logger.log_event(
                                "action", {"action": action},
                            )
                            self._handle_action(action)
            else:
                action = kb.get_action()
                if action == "quit":
                    break
                if action is not None:
                    self._logger.log_event("action", {"action": action})
                    self._handle_action(action)

            # Merge ToF + IMU + safety snapshots into arm state for display
            tof_snap = self._tof_state.snapshot()
            imu_snap = self._imu_state.snapshot()
            safety_snap = self._safety.snapshot()
            with self._state._lock:
                self._state.tof_snapshot = tof_snap

            # Surface recorder health in the status line so the user can
            # tell at a glance whether transitions are still accumulating
            # during a walk-away collection session.
            rec_st = self._rl_recorder.status()
            with self._state._lock:
                if rec_st["tick_fatal"]:
                    self._state.last_error = f"REC FATAL: {rec_st['tick_fatal']}"
                elif rec_st["last_save_error"]:
                    self._state.last_error = f"REC SAVE ERR: {rec_st['last_save_error']}"
                elif rec_st["running"]:
                    # Use last_resp so a session_logger "log: …" line from
                    # startup doesn't get stuck on-screen for 75 minutes.
                    self._state.last_resp = f"rec {rec_st['buffered']} samp"

            arm_snap = self._state.snapshot()
            display.render(
                arm_snap, imu_snap=imu_snap, sweep_snap=_legacy_sweep_snap(safety_snap),
                obs_snap=_legacy_obs_snap(safety_snap),
            )
            # If the refusal dialog is active, paint it on top.
            render_to_curses(stdscr, self._refusal_dialog)

            # Stream data to standalone plotter apps (fire-and-forget UDP)
            self._publisher.send_arm(arm_snap)
            self._publisher.send_tof(tof_snap)

        self._graceful_shutdown()

    # ── Connection ────────────────────────────────────────────────────────

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
                    # Peek at what the Arduino actually sent for a helpful hint
                    peek = ""
                    try:
                        r = self._bridge.response_queue.get_nowait()
                        raw = r.get("raw", r.get("type", ""))
                        if raw:
                            peek = f" (Arduino sent: {raw!r})"
                    except queue.Empty:
                        pass
                    self._state.last_error = (
                        f"No PONG from {self._bridge._port} — "
                        f"re-upload the new sketch?{peek}"
                    )
            if pong:
                # Sync position shadow from Arduino
                self._bridge.send_cmd(cmd_get_pos())
        else:
            err_msg = self._bridge.connect_error
            if not err_msg:
                hint = "pyserial not installed — pip install pyserial"
            elif "ermission denied" in err_msg or "Errno 13" in err_msg:
                hint = (
                    f"{err_msg} — "
                    "add user to dialout group: "
                    "sudo usermod -aG dialout $USER  "
                    "(Arch: uucp), then log out/in"
                )
            elif "busy" in err_msg.lower() or "Errno 16" in err_msg:
                hint = f"{err_msg} — close Arduino IDE / other serial monitors"
            else:
                hint = err_msg
            with self._state._lock:
                self._state.connected = False
                self._state.last_error = f"Cannot open {self._bridge._port} — {hint}"

    # ── Telemetry edge events ─────────────────────────────────────────────

    def _log_edge_events(self) -> None:
        """Emit edge-triggered events as safety state transitions."""
        if not self._logger.is_running:
            return

        safety_snap = self._safety.snapshot()
        mode = safety_snap.get("mode", "idle")
        if mode != self._prev_bt_mode:
            self._logger.log_event(
                "bt_mode",
                {"from": self._prev_bt_mode, "to": mode,
                 "bt_state": safety_snap.get("bt_state"),
                 "last_strategy": safety_snap.get("last_strategy")},
            )
            self._prev_bt_mode = mode

        with self._state._lock:
            response = self._state.obstacle_response
            source = self._state.obstacle_source
            dist = self._state.obstacle_dist_mm
        if response != self._prev_obs_response:
            self._logger.log_event(
                "obstacle_response",
                {
                    "from": self._prev_obs_response,
                    "to": response,
                    "source": source,
                    "dist_mm": dist,
                },
            )
            self._prev_obs_response = response

    # ── Obstacle detection ────────────────────────────────────────────────

    def _check_obstacle(self) -> None:
        """
        Check ToF/IR obstacle state and update arm state with warnings.

        Decision logic:
          IR   → obstacle detected (ToF missed!) → BACK AWAY immediately
          ToF  → obstacle within threshold → WARN (advisory only)
          BT   → trajectory actually replanned → REPLAN (red banner)

        The ToF per-channel threshold no longer drives REPLAN directly —
        that was producing false red banners for CH3 wiring reflections
        the BT correctly ignored. REPLAN is now published by the BT via
        ``safety_snap["last_strategy"]`` when it genuinely reroutes.

        The ToF warning is routed through `last_error` for display. This
        function must never overwrite an error set by a higher layer (a
        guard/refusal message) — those always take precedence. Only
        previously self-authored ToF/IR/BT warnings (or empty errors)
        are replaced on a new tick.
        """
        snap = self._tof_state.snapshot()
        response = snap["obstacle_response"]
        source = snap["obstacle_source"]
        dist = snap["obstacle_dist_mm"]

        # Pull the BT's latest replan strategy so we can surface a red
        # "REPLAN" banner only when the planner is genuinely rerouting.
        safety_snap = self._safety.snapshot() if self._safety else {}
        bt_strategy = str(safety_snap.get("last_strategy", "") or "")
        bt_is_replanning = any(
            bt_strategy.startswith(prefix)
            for prefix in (
                "sweep_replan",
                "sweep_z_ladder",
                "sweep_birrt_over",
                "sweep_edge_reverse",
                "escape_collision",
            )
        )

        with self._state._lock:
            self._state.obstacle_response = response
            self._state.obstacle_source = source
            self._state.obstacle_dist_mm = dist

            current_err = self._state.last_error or ""
            # Only error strings we previously wrote ourselves (or blanks)
            # are safe to replace. Guard rejection errors ("Obstacle
            # blocks", "Replanned around obstacle", …) must survive so
            # the user sees why their command was refused.
            overwritable = (
                not current_err
                or current_err.startswith("ToF:")
                or current_err.startswith("IR DANGER")
                or current_err.startswith("BT:")
            )

            ch_idx = -1
            try:
                ch_idx = int(source.replace("tof_ch", ""))
            except (ValueError, AttributeError):
                pass
            thresholds = snap.get("tof_thresholds_mm", TOF_THRESHOLDS_MM)
            ch_thresh = (
                thresholds[ch_idx]
                if 0 <= ch_idx < len(thresholds)
                else None
            )

            if response == ObstacleResponse.BACK_AWAY:
                if overwritable:
                    self._state.last_error = (
                        f"IR DANGER — BACK AWAY! (src={source}) "
                        f"ToF failed to detect, IR is second line of defense"
                    )
            elif bt_is_replanning and overwritable:
                # BT is actually rerouting — red "REPLAN" banner is honest here.
                self._state.last_error = (
                    f"BT: replanning ({bt_strategy})"
                )
            elif response == ObstacleResponse.WARN:
                if overwritable:
                    thr_text = (
                        f"{ch_thresh:.0f} mm"
                        if isinstance(ch_thresh, (int, float))
                        else "?"
                    )
                    self._state.last_error = (
                        f"ToF: ch{ch_idx} warning — {dist:.0f} mm "
                        f"< {thr_text} (advisory, BT unaffected)"
                    )
            elif response == ObstacleResponse.REPLAN:
                # Legacy path — kept so any future code that sets REPLAN
                # directly still renders. Not used by the current pipeline.
                if overwritable:
                    thr_text = (
                        f"{ch_thresh:.0f} mm"
                        if isinstance(ch_thresh, (int, float))
                        else "?"
                    )
                    self._state.last_error = (
                        f"ToF: obstacle at {dist:.0f} mm "
                        f"(ch{ch_idx} threshold={thr_text}, "
                        f"src={source}) — REPLAN TRAJECTORY"
                    )
            else:
                # Obstacle cleared: drop only our own warnings so we
                # don't clobber unrelated error strings from other handlers.
                if (
                    current_err.startswith("ToF:")
                    or current_err.startswith("IR DANGER")
                    or current_err.startswith("BT:")
                ):
                    self._state.last_error = ""

    # ── Sensor → SafetyAPI pump ──────────────────────────────────────────

    def _push_sensor_to_safety(self) -> None:
        """Feed the latest ToF / IR / IMU / arm state into the safety stack.

        Called every main-loop tick. The safety stack's WorldModel ingests
        the projected point cloud; the BT reads current_q / ir_severity /
        tof_filtered on its next tick.
        """
        tof_snap = self._tof_state.snapshot()
        arm_snap = self._state.snapshot()
        joints = list(arm_snap.get("joints", HOME_POS))
        thresholds = tof_snap.get("tof_thresholds_mm", TOF_THRESHOLDS_MM)

        # Point cloud ingest — pass imu_R=None so points land in the arm's
        # base frame, which is the same frame FK / capsule collision / the
        # self-filter / the sweep's polar query all use. Rotating ToF
        # points into a gravity-aligned world frame (via imu_R) put the
        # cloud in a frame none of the downstream safety checks knew
        # about, so the BT's is_blocked() always said "clear" even when
        # CH2/CH3 were reporting fingers flush against the sensor. The
        # IMU calibration stays useful for display / telemetry; it's just
        # not applied to the obstacle point cloud.
        self._safety.ingest_tof_frame(
            tof_snap["grids"], joints, None, thresholds,
        )
        # Hysteresis feed: per-channel minimum distance within threshold
        dists: list[float] = []
        for ch, grid in enumerate(tof_snap.get("grids", [])):
            if grid is None:
                dists.append(float("inf"))
                continue
            import numpy as _np
            try:
                if _np.isnan(grid).all():
                    dists.append(float("inf"))
                else:
                    dists.append(float(_np.nanmin(grid)))
            except Exception:
                dists.append(float("inf"))
        self._safety.update_tof_distances(dists)
        # IR severity from ObstacleResponse (0=clear, 2=close, 3=danger).
        response = tof_snap.get("obstacle_response", ObstacleResponse.CLEAR)
        severity = {
            ObstacleResponse.CLEAR: 0,
            ObstacleResponse.REPLAN: 1,
            ObstacleResponse.BACK_AWAY: 3,
        }.get(response, 0)
        self._safety.update_ir_severity(severity)
        # Sync current joints so the BT's manual/sweep branches plan from
        # the real arm position.
        self._safety.update_current_q(joints)

    # ── IMU calibration ────────────────────────────────────────────────────

    def _run_imu_calibration(self) -> None:
        """Record the current IMU yaw as the calibration reference.

        After calibration all yaw values (and therefore world-frame obstacle
        projections) are expressed relative to this pose.  Call this with the
        arm at a known orientation (e.g. theta=90, arm pointing forward).
        """
        self._imu_state.record_calibration()
        # World frame changed — drop the accumulated point cloud / grid.
        self._safety.world.clear()
        self._safety.polar.clear()
        with self._state._lock:
            self._state.last_resp = (
                f"IMU calibrated: yaw_ref="
                f"{self._imu_state.yaw_calibration_offset:.1f}°"
                f" (world model cleared)"
            )
            self._state.last_error = ""

    # ── Response draining ─────────────────────────────────────────────────

    def _drain_responses(self) -> None:
        """Process all pending Arduino responses, update state."""
        while True:
            try:
                resp = self._bridge.response_queue.get_nowait()
            except queue.Empty:
                break

            rtype = resp.get("type", "unknown")
            with self._state._lock:
                if rtype == "pos":
                    # Always shadow joints from Arduino. Polar (theta/r/z) is
                    # synced exactly once — the first POS after startup — so
                    # the first keypress doesn't compute IK from the stale
                    # software defaults (r=152, z=-50). After that the polar
                    # state is user-authoritative; re-deriving it from every
                    # POS would leak IK integer-rounding drift back into the
                    # commanded values (right-arrow slowly dropping z, etc.).
                    positions = list(resp["positions"])
                    self._state.joints = positions
                    if not self._state.polar_synced:
                        theta, r, z = fk_polar(positions)
                        self._state.theta = theta
                        self._state.r = r
                        self._state.z = z
                        self._state.polar_synced = True
                        self._state.last_resp = "POS synced"
                    else:
                        self._state.last_resp = "POS"
                    self._state.last_error = ""
                elif rtype == "error":
                    self._state.last_error = resp.get("message", "")
                    self._state.last_resp = resp.get("raw", "ERR")
                elif rtype == "ready":
                    self._state.connected = True
                    self._state.last_resp = "READY"
                    self._state.last_error = ""
                elif rtype in ("ok_all", "ok_joint", "ok_home", "ok_delta", "pong"):
                    self._state.last_error = ""
                    self._state.last_resp = resp.get("raw", rtype)
                elif rtype == "unknown":
                    self._state.last_resp = resp.get("raw", "")

    # ── Action dispatch ───────────────────────────────────────────────────

    def _handle_action(self, action: str) -> None:
        """
        Mutate ArmState for the given action, then send the appropriate
        serial command.  IK-affecting actions always send SET ALL.
        Independent axes send targeted single commands.
        """
        # ── Overlay menus (take over the screen synchronously) ────────────
        if action == "states_menu":
            self._open_states_menu()
            return
        if action == "seq_editor":
            self._open_seq_editor()
            return
        # ── Autonomous sweep ──────────────────────────────────────────────
        if action == "sweep_toggle":
            if self._rl_sweeper is not None:
                # RL policy mode: Z key drives RLSweeper, not the safety BT.
                if self._rl_sweeper.running():
                    self._rl_sweeper.stop()
                    with self._state._lock:
                        self._state.last_resp = "RL sweep stopped"
                else:
                    self._rl_sweeper.start()
                    with self._state._lock:
                        self._state.last_resp = "RL sweep started"
            else:
                safety_snap = self._safety.snapshot()
                if safety_snap.get("mode") == "sweep":
                    self._safety.stop_sweep()
                    with self._state._lock:
                        self._state.last_resp = "Sweep stopped"
                else:
                    self._safety.start_sweep()
                    with self._state._lock:
                        self._state.last_resp = "Sweep started"
            return
        # ── IMU calibration ───────────────────────────────────────────────
        if action == "imu_calibrate":
            self._run_imu_calibration()
            return
        # ── Strict-mode toggle ────────────────────────────────────────────
        if action == "strict_toggle":
            new_state = not self._safety.strict_mode_enabled()
            self._safety.set_strict_mode(new_state)
            # Turning strict OFF should also close any open refusal dialog
            # and clear the pending refusal so the auto-replan path takes
            # over without the user needing to press anything else.
            if not new_state:
                if self._refusal_dialog.is_open:
                    self._refusal_dialog.close()
                self._safety.consume_refusal()
            with self._state._lock:
                self._state.last_resp = (
                    "Strict mode: ON (refusal dialog enabled)"
                    if new_state
                    else "Strict mode: OFF (silent auto-replan)"
                )
            return
        # ── No-op placeholder (numpad 5 etc.) ─────────────────────────────
        if action == "ik_noop":
            return
        # ── Dialog actions leaked through KEY_BINDINGS are ignored here;
        #    they are only meaningful while the refusal dialog is open
        #    and are handled by _dispatch_refusal_raw_key instead.
        if action in ("dialog_retreat", "dialog_axis", "dialog_step",
                      "dialog_override", "dialog_cancel"):
            return

        state = self._state
        ik_dirty = False  # needs full IK recompute + SET ALL
        send_fn = None  # callable → sends the command after state update

        with state._lock:
            # ── IK polar axes ─────────────────────────────────────────────
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
            # ── Wrist vertical offset (still needs IK recompute) ──────────
            elif action == "wv_inc":
                state.wrist_offset = min(90.0, state.wrist_offset + WRIST_V_STEP)
                ik_dirty = True
            elif action == "wv_dec":
                state.wrist_offset = max(-90.0, state.wrist_offset - WRIST_V_STEP)
                ik_dirty = True
            # ── Independent axes ──────────────────────────────────────────
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
            # ── Slew rate ─────────────────────────────────────────────────
            elif action == "delta_inc":
                state.delta = min(DELTA_MAX, state.delta + 1)
                send_fn = self._send_delta
            elif action == "delta_dec":
                state.delta = max(DELTA_MIN, state.delta - 1)
                send_fn = self._send_delta
            # ── Equilibrium ───────────────────────────────────────────────
            elif action == "go_home":
                send_fn = self._send_home
            elif action == "set_equil":
                state.set_equil_from_current()
                # No command to send — just save local state
                with state._lock:
                    state.last_cmd = "(equil saved)"
                    state.last_resp = ""
                return

        if ik_dirty:
            self._send_ik_move()
        elif send_fn is not None:
            send_fn()

    # ── Command senders ───────────────────────────────────────────────────

    def _send_ik_move(self) -> None:
        """Recompute IK for the current polar state and queue the target
        through the SafetyAPI as a manual intent.

        The BT's ManualBranch validates the target against the full-arm
        capsule collision checker and either emits the command or opens a
        refusal dialog. The IK shadow is rolled back here if IK itself is
        unreachable — otherwise the BT will either execute or refuse.
        """
        state = self._state
        with state._lock:
            theta = state.theta
            r = state.r
            z = state.z
            wrist_offset = state.wrist_offset
            wrist_rot = state.wrist_rot
            gripper = state.gripper
            cur_joints = list(state.joints)

        x, y = polar_to_cartesian(theta, r)
        result = solve_ik(x, y, z)
        if result is None:
            with state._lock:
                # IK impossible — roll back the polar shadow to the current
                # arm pose so the display doesn't lie.
                state.theta, state.r, state.z = fk_polar(cur_joints)
                state.last_error = (
                    f"IK unreachable: theta={theta:.0f}°  "
                    f"r={r:.0f} mm  z={z:.0f} mm"
                )
            return

        state.update_joints_from_ik(result, wrist_offset, wrist_rot, gripper)
        target = list(state.snapshot()["joints"])
        with state._lock:
            state.last_cmd = f"manual intent → {target}"
            state.last_error = ""
        self._safety.queue_manual_intent(target)

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
        """Route to the equilibrium pose through the safety stack.

        The cascading replanner handles obstacle avoidance (direct plan →
        via-HOME detour → midpoint perturbations → wait-and-retry →
        bailout). On success the arm walks the returned path one step at a
        time, driven by the BT's sequence branch.
        """
        state = self._state
        with state._lock:
            cur_joints = list(state.joints)
            eq_theta = state.equil_theta
            eq_r = state.equil_r
            eq_z = state.equil_z
            eq_wrist_offset = state.equil_wrist_offset
            eq_wrist_rot = state.equil_wrist_rot
            eq_gripper = state.equil_gripper

        # Build the goal joint config from the equilibrium polar pose.
        x, y = polar_to_cartesian(eq_theta, eq_r)
        result = solve_ik(x, y, eq_z)
        if result is None:
            with state._lock:
                state.last_error = (
                    f"HOME IK unreachable: θ={eq_theta:.0f}° r={eq_r:.0f} z={eq_z:.0f}"
                )
            return
        goal_q = [
            int(result.base),
            int(result.shoulder),
            int(result.elbow),
            int(max(0, min(180, result.wrist_vert + eq_wrist_offset))),
            int(eq_wrist_rot),
            int(eq_gripper),
        ]
        self._safety.queue_sequence([
            Waypoint(q=goal_q, kind=WaypointType.ESSENTIAL, name="HOME",
                     gripper=int(eq_gripper)),
        ])
        with state._lock:
            state.last_cmd = f"HOME via sequence → {goal_q}"
            state.last_error = ""

    # ── State library helpers ─────────────────────────────────────────────

    def _apply_state(self, state_dict: dict) -> None:
        """Apply a saved state through the cascading replanner.

        The safety stack's CascadingReplanner owns the strategies for going
        around obstacles. Saved states are essential waypoints — the
        replanner does NOT skip them.
        """
        state = self._state
        joints = list(state_dict["joints"])
        with state._lock:
            state.theta = float(state_dict["theta"])
            state.r = float(state_dict["r"])
            state.z = float(state_dict["z"])
            state.wrist_offset = state_dict["wrist_offset"]
            state.wrist_rot = state_dict["wrist_rot"]
            state.gripper = state_dict["gripper"]
            state.last_cmd = f"saved state → {joints}"
            state.last_error = ""
        self._safety.queue_sequence([
            Waypoint(q=joints, kind=WaypointType.ESSENTIAL,
                     name=state_dict.get("name", "saved_state"),
                     gripper=int(state_dict.get("gripper", 73))),
        ])

    def _run_named_state(self, name: str) -> None:
        """Look up a state by name and apply it.  Called from background thread."""
        st = self._state_lib.get_state(name)
        if st:
            self._apply_state(st)

    def _open_states_menu(self) -> None:
        """Open the states-management overlay, then restore curses settings."""
        from .states_menu import run_states_menu

        result = run_states_menu(
            self._stdscr,
            self._state_lib,
            self._state.snapshot(),
        )
        if result:
            self._apply_state(result)
        curses.halfdelay(2)
        curses.curs_set(0)

    def _open_seq_editor(self) -> None:
        """Open the sequence editor overlay, then restore curses settings."""
        from .sequence_editor import run_sequence_editor

        run_sequence_editor(
            self._stdscr,
            self._state_lib,
            self._run_named_state,
        )
        curses.halfdelay(2)
        curses.curs_set(0)

    # ── SafetyAPI plumbing ────────────────────────────────────────────────

    def _send_safety_command(self, joints: list, delta: int) -> None:
        """SafetyAPI callback — send a post-EMA joint target to the arm.

        The BT has already validated the move (planner says the full-arm
        path is collision-free). We just translate the joint list into the
        Arduino protocol and push it onto the bridge.
        """
        cmd = cmd_set_all(joints)
        with self._state._lock:
            self._state.joints = list(joints)
            # Keep polar shadow in sync with the actual command so display
            # reads are coherent — the existing fk_polar handles this on
            # next GET POS, but at high tick rates the display can race.
            try:
                theta, r, z = fk_polar(joints)
                self._state.theta = theta
                self._state.r = r
                self._state.z = z
            except Exception:
                pass
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    # ── Raw keypress routing for refusal dialog ──────────────────────────
    #
    # Dialog keys live on F1–F5 / ESC rather than letter keys (R/A/S/O/C)
    # because the letter keys double as manual-IK bindings. When the
    # dialog is open, the main loop bypasses KEY_BINDINGS and calls this
    # with the raw curses int.

    _DIALOG_RAW_KEY_MAP = {
        curses.KEY_F1: "retreat",
        curses.KEY_F2: "try_axis",
        curses.KEY_F3: "reduce_step",
        curses.KEY_F4: "override",
        curses.KEY_F5: "cancel",
        27:            "cancel",   # ESC
    }

    def _dispatch_refusal_raw_key(self, key: int) -> None:
        from .safety import DialogChoice

        choice_name = self._DIALOG_RAW_KEY_MAP.get(int(key))
        if choice_name is None:
            return
        choice = DialogChoice(choice_name)
        self._refusal_dialog.close()

        with self._state._lock:
            self._state.last_error = f"refusal: chose {choice.value}"
        if choice in (DialogChoice.CANCEL, DialogChoice.RETREAT):
            self._safety.bb.manual_intent = None
        elif choice == DialogChoice.OVERRIDE:
            info = self._refusal_dialog.info
            if info is not None and getattr(info, "attempted_q", None):
                # Narrow escape hatch: re-queue the refused intent once,
                # reset the EMA so it lands in full.
                self._safety.joint_ema.reset()
                self._safety.queue_manual_intent(list(info.attempted_q))

    def _signal_shutdown(self, signum, frame) -> None:
        """Signal handler — parks the arm and exits."""
        self._graceful_shutdown()
        # Re-raise the default handler so the process actually exits.
        import sys
        sys.exit(0)

    # ── Graceful shutdown ────────────────────────────────────────────────

    def _graceful_shutdown(self) -> None:
        """Stop the BT, send the arm to HOME at the slowest slew, and close
        every I/O handle. Idempotent — safe to call from the main loop's
        normal exit, an atexit handler, or a SIGINT / SIGTERM trap.

        The HOME + slow-delta pair matters: when the program restarts
        without first parking the arm, servo PWM from the old session can
        still be latched and new commands from a different joint shadow
        cause the arm to whip violently. Parking at HOME at DELTA=1 leaves
        the arm in a known, safe pose for the next session.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        try:
            if getattr(self, "_rl_sweeper", None) is not None:
                self._rl_sweeper.stop()
        except Exception:
            pass
        try:
            self._safety.stop()
        except Exception:
            pass
        try:
            # Slow slew + HOME — minimises servo current on restart.
            self._bridge.send_cmd(cmd_set_delta(1))
            self._bridge.send_cmd(cmd_home())
        except Exception:
            pass
        try:
            self._logger.stop()
        except Exception:
            pass
        # RL recorder — this is the one-shot data collection artifact, so
        # surface success + failure loudly. curses is torn down by the time
        # _graceful_shutdown runs (wrapper has restored the terminal), so
        # plain print() is safe and visible. Logging also lands in
        # logs/controller.log via the __main__ basicConfig.
        try:
            self._rl_recorder.stop()
        except Exception as exc:
            print(f"\nERROR: rl_recorder flush failed: {exc}", file=sys.stderr)
            logging.getLogger(__name__).exception(
                "rl_recorder.stop() failed during shutdown"
            )
        else:
            st = self._rl_recorder.status()
            if st["last_save_path"]:
                print(
                    f"\n✓ RL transitions saved: {st['last_save_count']} "
                    f"→ {st['last_save_path']}"
                )
            elif st["tick_fatal"]:
                print(
                    f"\nERROR: rl_recorder crashed before saving: {st['tick_fatal']}",
                    file=sys.stderr,
                )
            else:
                print(
                    "\nWARNING: rl_recorder stopped with 0 transitions buffered — "
                    "nothing was written. Check logs/controller.log.",
                    file=sys.stderr,
                )
        try:
            self._bridge.close()
        except Exception:
            pass
        try:
            self._tof_bridge.close()
        except Exception:
            pass
        try:
            self._publisher.close()
        except Exception:
            pass

    # ── Startup POS wait ─────────────────────────────────────────────────

    def _await_pos_sync(self, timeout_s: float = 1.5) -> None:
        """Block until ``_drain_responses`` processes the first POS
        response from the Arduino, or the timeout expires.

        Without this wait, the behavior tree can start planning from the
        software-default HOME_POS before the real arm position is known,
        which causes a violent snap on the first keypress if the arm was
        left somewhere else between runs.
        """
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            self._drain_responses()
            with self._state._lock:
                # ``_drain_responses`` sets last_resp to "POS synced" when
                # the Arduino replies to the GET POS issued in _connect.
                if self._state.last_resp == "POS synced":
                    return
            time.sleep(0.02)
