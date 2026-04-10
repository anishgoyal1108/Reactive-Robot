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

import curses
import queue

from .arm_state import ArmState
from .backends import BraccioBackend, HardwareBackend
from .ik_solver import solve_ik, polar_to_cartesian, fk_polar
from .keyboard_handler import KeyboardHandler
from .display import CursesDisplay
from .state_library import StateLibrary
from .tof_sensor import ToFState, ToFBridge, ObstacleResponse
from .imu_state import IMUState
from .obstacle_map import ObstacleMap
from .motion_guard import MotionGuard
from .auto_sweep import AutoSweeper
from .data_publisher import DataPublisher
from .session_logger import SessionLogger
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
    SENSOR_IGNORE_CHANNELS,
    TOF_DEFAULT_PORT,
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

        # ── Obstacle map + motion guard + autonomous sweep ───────────────
        self._obstacle_map = ObstacleMap(thresholds_mm=TOF_THRESHOLDS_MM)
        # Wire tof_state into the guard so manual commands are gated on the
        # live ToF snapshot, not just the voxel memory. The voxel memory
        # lags the sensor FOV by exactly the commands that would push the
        # arm into an obstacle — see the "Live-sensor gate" section of
        # motion_guard.py for the full reasoning.
        self._guard = MotionGuard(self._obstacle_map, tof_state=self._tof_state)
        self._sweeper = AutoSweeper(
            arm_state=self._state,
            bridge=self._bridge,
            tof_state=self._tof_state,
            obstacle_map=self._obstacle_map,
            guard=self._guard,
        )

        # ── Session telemetry logger (JSONL for offline review) ─────────
        self._logger = SessionLogger()
        self._logger.set_sources(
            arm_state=self._state,
            tof_state=self._tof_state,
            sweeper=self._sweeper,
        )
        # Track sweep state for edge-triggered events
        self._prev_sweep_running = False
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

        # Main loop (~5 Hz, paced by keyboard halfdelay)
        while True:
            self._drain_responses()
            self._check_obstacle()
            self._update_obstacle_map()
            self._log_edge_events()

            action = kb.get_action()

            if action == "quit":
                break

            if action is not None:
                # Tag every user action in the telemetry stream
                self._logger.log_event("action", {"action": action})
                self._handle_action(action)

            # Merge ToF + IMU + sweep snapshots into arm state for display
            tof_snap = self._tof_state.snapshot()
            imu_snap = self._imu_state.snapshot()
            sweep_snap = self._sweeper.get_status()
            obs_snap = self._obstacle_map.snapshot()
            with self._state._lock:
                self._state.tof_snapshot = tof_snap

            arm_snap = self._state.snapshot()
            display.render(
                arm_snap, imu_snap=imu_snap, sweep_snap=sweep_snap, obs_snap=obs_snap
            )

            # Stream data to standalone plotter apps (fire-and-forget UDP)
            self._publisher.send_arm(arm_snap)
            self._publisher.send_tof(tof_snap)

        if self._sweeper.is_running():
            self._sweeper.stop()
        try:
            self._logger.stop()
        except Exception:
            pass
        self._bridge.close()
        self._tof_bridge.close()
        self._publisher.close()

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
        """
        Emit one-off telemetry events when observable state transitions.
        Called once per main-loop tick, BEFORE handling new keyboard input,
        so the stream reads as a causal history.
        """
        if not self._logger.is_running:
            return

        # Sweep start/stop edges
        sweep_running = self._sweeper.is_running()
        if sweep_running != self._prev_sweep_running:
            self._logger.log_event(
                "sweep_running",
                {"running": sweep_running},
            )
            self._prev_sweep_running = sweep_running

        # Obstacle response edges (clear → replan → back_away)
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
          ToF  → obstacle within threshold → REPLAN trajectory
          IR   → obstacle detected (ToF missed!) → BACK AWAY immediately

        The actual trajectory replanning / retreat is signaled through
        ArmState fields so the display shows the alert and higher-level
        autonomy code can act on it.

        The ToF warning is routed through `last_error` for display. This
        function must never overwrite an error set by MotionGuard (e.g.
        "Obstacle blocks θ=…") — guard errors always take precedence so
        the user sees why a specific command was refused. Only previously
        self-authored ToF/IR warnings (or empty errors) are replaced on
        a new tick.
        """
        snap = self._tof_state.snapshot()
        response = snap["obstacle_response"]
        source = snap["obstacle_source"]
        dist = snap["obstacle_dist_mm"]

        with self._state._lock:
            self._state.obstacle_response = response
            self._state.obstacle_source = source
            self._state.obstacle_dist_mm = dist

            current_err = self._state.last_error or ""
            # Only error strings we previously wrote ourselves (or blanks)
            # are safe to replace. Guard rejection errors ("Obstacle
            # blocks", "Replanned around obstacle", "HOME blocked",
            # "Saved state blocked", …) must survive so the user sees why
            # their command was refused.
            overwritable = (
                not current_err
                or current_err.startswith("ToF:")
                or current_err.startswith("IR DANGER")
            )

            if response == ObstacleResponse.BACK_AWAY:
                if overwritable:
                    self._state.last_error = (
                        f"IR DANGER — BACK AWAY! (src={source}) "
                        f"ToF failed to detect, IR is second line of defense"
                    )
            elif response == ObstacleResponse.REPLAN:
                if overwritable:
                    ch_idx = -1
                    try:
                        ch_idx = int(source.replace("tof_ch", ""))
                    except (ValueError, AttributeError):
                        pass
                    thresholds = snap.get("tof_thresholds_mm", TOF_THRESHOLDS_MM)
                    ch_thresh = (
                        thresholds[ch_idx]
                        if 0 <= ch_idx < len(thresholds)
                        else "?"
                    )
                    self._state.last_error = (
                        f"ToF: obstacle at {dist:.0f} mm "
                        f"(ch{ch_idx} threshold={ch_thresh:.0f} mm, "
                        f"src={source}) — REPLAN TRAJECTORY"
                    )
            else:
                # Obstacle cleared: drop only our own ToF/IR warnings so we
                # don't clobber unrelated error strings from other handlers.
                if (
                    current_err.startswith("ToF:")
                    or current_err.startswith("IR DANGER")
                ):
                    self._state.last_error = ""

    # ── Obstacle map update ───────────────────────────────────────────────

    def _update_obstacle_map(self) -> None:
        """Project current ToF grids into world frame and update obstacle map."""
        tof_snap = self._tof_state.snapshot()
        arm_snap = self._state.snapshot()
        imu_snap = self._imu_state.snapshot()
        imu_R = self._imu_state.rotation_matrix()
        self._obstacle_map.update(
            tof_snap["grids"],
            arm_snap,
            imu_snap,
            imu_R,
            tof_thresholds_mm=tof_snap.get("tof_thresholds_mm"),
        )

    # ── IMU calibration ────────────────────────────────────────────────────

    def _run_imu_calibration(self) -> None:
        """
        Record the current IMU yaw as the calibration reference.

        After calibration all yaw values (and therefore world-frame obstacle
        projections) are expressed relative to this pose.  Call this with the
        arm at a known orientation (e.g. theta=90, arm pointing forward).
        """
        self._imu_state.record_calibration()
        # World frame changed — clear persistent obstacle memory
        self._obstacle_map.clear_memory()
        with self._state._lock:
            self._state.last_resp = (
                f"IMU calibrated: yaw_ref="
                f"{self._imu_state.yaw_calibration_offset:.1f}°"
                f" (obstacle memory cleared)"
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
                    # Sync joints AND IK polar state from Arduino's actual angles.
                    # Without this, the first keypress after startup computes IK
                    # from the stale software default (r=152, z=-50) rather than
                    # the arm's real position, causing a violent unexpected move.
                    positions = list(resp["positions"])
                    theta, r, z = fk_polar(positions)
                    self._state.joints = positions
                    self._state.theta = theta
                    self._state.r = r
                    self._state.z = z
                    self._state.last_resp = "POS synced"
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
        if action == "plot_main_toggle":
            if self._plotter is not None:
                self._plotter.toggle_main()
            return
        if action.startswith("plot_joint_") and action.endswith("_toggle"):
            if self._plotter is not None:
                idx = int(action[len("plot_joint_"):]) - 1
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
        if action == "plot_log_toggle":
            if self._plotter is not None:
                self._plotter.toggle_logging()
            return
        # ── ToF / IR actions ──────────────────────────────────────────────
        if action == "tof_view_toggle":
            if self._tof_plotter is not None:
                self._tof_plotter.toggle_main()
            return
        if action == "tof_export_csv":
            if self._tof_plotter is not None:
                path = self._tof_plotter.export_csv_snapshot()
                with self._state._lock:
                    self._state.last_resp = f"ToF CSV → {path}"
            return
        if action == "tof_screenshot":
            if self._tof_plotter is not None:
                self._tof_plotter.save_screenshot()
            return
        if action == "tof_log_toggle":
            if self._tof_plotter is not None:
                self._tof_plotter.toggle_logging()
            return
        if action == "tof_threshold_inc":
            # Adjust primary (side) channel thresholds only
            with self._tof_state._lock:
                for ch in range(len(self._tof_state.tof_thresholds_mm)):
                    if ch not in SENSOR_IGNORE_CHANNELS:
                        self._tof_state.tof_thresholds_mm[ch] = min(
                            3000.0, self._tof_state.tof_thresholds_mm[ch] + 50.0)
            return
        if action == "tof_threshold_dec":
            with self._tof_state._lock:
                for ch in range(len(self._tof_state.tof_thresholds_mm)):
                    if ch not in SENSOR_IGNORE_CHANNELS:
                        self._tof_state.tof_thresholds_mm[ch] = max(
                            50.0, self._tof_state.tof_thresholds_mm[ch] - 50.0)
            return
        # ── Autonomous sweep ──────────────────────────────────────────────
        if action == "sweep_toggle":
            if self._sweeper.is_running():
                self._sweeper.stop()
                with self._state._lock:
                    self._state.last_resp = "Sweep stopped"
            else:
                self._sweeper.start()
                with self._state._lock:
                    self._state.last_resp = "Sweep started"
            return
        # ── IMU calibration ───────────────────────────────────────────────
        if action == "imu_calibrate":
            self._run_imu_calibration()
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
        """Recompute IK and send SET ALL for the current polar state.

        Runs the desired pose through MotionGuard first. If the target is
        blocked the guard may return a replanned pose; if nothing clears,
        the command is rejected and the IK shadow is rolled back so the
        display matches what the arm actually does.
        """
        state = self._state
        with state._lock:
            theta = state.theta
            r = state.r
            z = state.z
            wrist_offset = state.wrist_offset
            wrist_rot = state.wrist_rot
            gripper = state.gripper
            # Where the arm currently is, for the swept-volume path check.
            cur_theta_r_z = fk_polar(list(state.joints))

        planned = self._guard.plan_clear_pose(
            desired=(theta, r, z),
            current=cur_theta_r_z,
        )

        if planned is None:
            with state._lock:
                # Roll back the IK shadow to current so the display doesn't
                # show a pose the arm never entered.
                state.theta, state.r, state.z = cur_theta_r_z
                state.last_error = (
                    f"Obstacle blocks θ={theta:.0f}° r={r:.0f} z={z:.0f} "
                    f"— retract (s) or change z (q/e) to clear"
                )
            return

        if planned != (theta, r, z):
            with state._lock:
                state.theta, state.r, state.z = planned
                state.last_error = (
                    f"Replanned around obstacle → "
                    f"θ={planned[0]:.0f}° r={planned[1]:.0f} z={planned[2]:.0f}"
                )
            theta, r, z = planned
        else:
            with state._lock:
                state.last_error = ""

        x, y = polar_to_cartesian(theta, r)
        result = solve_ik(x, y, z)

        if result is None:
            with state._lock:
                state.last_error = (
                    f"IK unreachable: theta={theta:.0f}°  "
                    f"r={r:.0f} mm  z={z:.0f} mm"
                )
            return

        state.update_joints_from_ik(result, wrist_offset, wrist_rot, gripper)
        positions = state.snapshot()["joints"]
        cmd = cmd_set_all(positions)
        with state._lock:
            state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

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
        """Restore IK state to equilibrium and send HOME command.

        The firmware HOME is a one-shot interpolation we can't abort
        mid-move, so we gate it through MotionGuard first. If the
        equilibrium pose is clear we send the native HOME; otherwise we
        fall back to SET ALL at a replanned pose.
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

        cur_theta_r_z = fk_polar(cur_joints)
        planned = self._guard.plan_clear_pose(
            desired=(eq_theta, eq_r, eq_z),
            current=cur_theta_r_z,
        )

        if planned is None:
            with state._lock:
                state.last_error = "HOME blocked by obstacle — no clear route"
            return

        if planned == (eq_theta, eq_r, eq_z):
            # Clear path to the real equilibrium — fast path via firmware HOME.
            state.restore_equil()
            cmd = cmd_home()
            with state._lock:
                state.last_cmd = cmd.strip()
                state.last_error = ""
            self._bridge.send_cmd(cmd)
            self._bridge.send_cmd(cmd_get_pos())
            return

        # Equilibrium blocked — build a SET ALL move to the replanned pose.
        theta, r, z = planned
        x, y = polar_to_cartesian(theta, r)
        result = solve_ik(x, y, z)
        if result is None:
            with state._lock:
                state.last_error = (
                    f"HOME replan unreachable: θ={theta:.0f}° r={r:.0f} z={z:.0f}"
                )
            return

        with state._lock:
            state.theta = theta
            state.r = r
            state.z = z
            state.wrist_offset = eq_wrist_offset
            state.wrist_rot = eq_wrist_rot
            state.gripper = eq_gripper
        state.update_joints_from_ik(
            result, eq_wrist_offset, eq_wrist_rot, eq_gripper
        )
        positions = state.snapshot()["joints"]
        cmd = cmd_set_all(positions)
        with state._lock:
            state.last_cmd = cmd.strip()
            state.last_error = (
                f"HOME replanned around obstacle → "
                f"θ={theta:.0f}° r={r:.0f} z={z:.0f}"
            )
        self._bridge.send_cmd(cmd)

    # ── State library helpers ─────────────────────────────────────────────

    def _apply_state(self, state_dict: dict) -> None:
        """
        Apply a saved state dict to the arm.

        Runs the saved pose through MotionGuard first. If the pose is
        blocked, either replans to a nearby clear pose (preserving wrist
        offset, wrist rotation, and gripper) or rejects the command and
        leaves the arm where it is.
        """
        state = self._state
        with state._lock:
            cur_joints = list(state.joints)
        cur_theta_r_z = fk_polar(cur_joints)

        desired_theta = float(state_dict["theta"])
        desired_r     = float(state_dict["r"])
        desired_z     = float(state_dict["z"])
        wrist_offset  = state_dict["wrist_offset"]
        wrist_rot     = state_dict["wrist_rot"]
        gripper       = state_dict["gripper"]

        planned = self._guard.plan_clear_pose(
            desired=(desired_theta, desired_r, desired_z),
            current=cur_theta_r_z,
        )

        if planned is None:
            with state._lock:
                state.last_error = (
                    f"Saved state blocked by obstacle "
                    f"(θ={desired_theta:.0f}° r={desired_r:.0f} z={desired_z:.0f})"
                )
            return

        theta, r, z = planned
        was_replanned = planned != (desired_theta, desired_r, desired_z)

        # Happy path: the saved joint array is still valid because nothing
        # was replanned. Reuse the raw joints verbatim so we preserve any
        # off-level wrist pose the user captured.
        if not was_replanned:
            with state._lock:
                state.joints = list(state_dict["joints"])
                state.theta = theta
                state.r = r
                state.z = z
                state.wrist_offset = wrist_offset
                state.wrist_rot = wrist_rot
                state.gripper = gripper
            cmd = cmd_set_all(state_dict["joints"])
            with state._lock:
                state.last_cmd = cmd.strip()
                state.last_error = ""
            self._bridge.send_cmd(cmd)
            return

        # Replanned: recompute joints via IK at the new pose, preserving
        # wrist_offset / wrist_rot / gripper from the saved state.
        x, y = polar_to_cartesian(theta, r)
        result = solve_ik(x, y, z)
        if result is None:
            with state._lock:
                state.last_error = (
                    f"Replan unreachable: θ={theta:.0f}° r={r:.0f} z={z:.0f}"
                )
            return

        with state._lock:
            state.theta = theta
            state.r = r
            state.z = z
            state.wrist_offset = wrist_offset
            state.wrist_rot = wrist_rot
            state.gripper = gripper
        state.update_joints_from_ik(result, wrist_offset, wrist_rot, gripper)
        positions = state.snapshot()["joints"]
        cmd = cmd_set_all(positions)
        with state._lock:
            state.last_cmd = cmd.strip()
            state.last_error = (
                f"Saved state replanned around obstacle → "
                f"θ={theta:.0f}° r={r:.0f} z={z:.0f}"
            )
        self._bridge.send_cmd(cmd)

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
