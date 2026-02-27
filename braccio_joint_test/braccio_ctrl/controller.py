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
"""

import curses
import queue

from .arm_state      import ArmState
from .serial_bridge  import SerialBridge
from .ik_solver      import solve_ik, polar_to_cartesian
from .keyboard_handler import KeyboardHandler
from .display        import CursesDisplay
from .protocol       import (
    cmd_set_all, cmd_set_joint, cmd_set_delta, cmd_home, cmd_get_pos,
)
from .constants import (
    THETA_STEP, R_STEP, Z_STEP, WRIST_V_STEP, WRIST_R_STEP,
    GRIPPER_OPEN, GRIPPER_CLOSE,
    DELTA_MIN, DELTA_MAX,
    JOINT_WRIST_ROT, JOINT_GRIPPER,
    R_MIN, R_MAX, Z_MIN, Z_MAX,
    BAUD_RATE,
)


class BraccioController:
    """
    Top-level controller.

    Parameters
    ----------
    port : serial port device path, e.g. '/dev/ttyACM0'
    baud : baud rate (default 115200, must match Arduino sketch)
    """

    def __init__(self, port: str, baud: int = BAUD_RATE):
        self._bridge  = SerialBridge(port, baud)
        self._state   = ArmState()
        self._plotter = None   # set via attach_plotter() before run()

    def attach_plotter(self, plotter) -> None:
        """Attach an ArmPlotter instance before calling run()."""
        self._plotter = plotter

    # ── Entry point ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Open the curses session and start the main loop."""
        curses.wrapper(self._curses_main)

    # ── curses main ───────────────────────────────────────────────────────

    def _curses_main(self, stdscr) -> None:
        kb      = KeyboardHandler(stdscr)
        display = CursesDisplay(stdscr)

        # Connect to Arduino
        self._connect()

        # Start plotter window (non-blocking — runs in its own thread)
        if self._plotter is not None:
            self._plotter.start()

        # Main loop (~5 Hz, paced by keyboard halfdelay)
        while True:
            self._drain_responses()

            action = kb.get_action()

            if action == 'quit':
                break

            if action is not None:
                self._handle_action(action)

            display.render(self._state.snapshot())

        self._bridge.close()
        if self._plotter is not None:
            self._plotter.stop()

    # ── Connection ────────────────────────────────────────────────────────

    def _connect(self) -> None:
        ok = self._bridge.connect()
        if ok:
            pong = self._bridge.ping()
            with self._state._lock:
                self._state.connected = pong
                if pong:
                    self._state.last_resp  = "PONG"
                    self._state.last_error = ""
                else:
                    # Peek at what the Arduino actually sent for a helpful hint
                    peek = ""
                    try:
                        r = self._bridge.response_queue.get_nowait()
                        raw = r.get('raw', r.get('type', ''))
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
            with self._state._lock:
                self._state.connected  = False
                self._state.last_error = (
                    f"Cannot open {self._bridge._port} — "
                    f"{self._bridge.connect_error}"
                )

    # ── Response draining ─────────────────────────────────────────────────

    def _drain_responses(self) -> None:
        """Process all pending Arduino responses, update state."""
        while True:
            try:
                resp = self._bridge.response_queue.get_nowait()
            except queue.Empty:
                break

            rtype = resp.get('type', 'unknown')
            with self._state._lock:
                if rtype == 'pos':
                    # Sync joint shadow from Arduino's actual commanded angles
                    self._state.joints = list(resp['positions'])
                    self._state.last_resp = "POS synced"
                    self._state.last_error = ""
                elif rtype == 'error':
                    self._state.last_error = resp.get('message', '')
                    self._state.last_resp  = resp.get('raw', 'ERR')
                elif rtype == 'ready':
                    self._state.connected  = True
                    self._state.last_resp  = "READY"
                    self._state.last_error = ""
                elif rtype in ('ok_all', 'ok_joint', 'ok_home',
                               'ok_delta', 'pong'):
                    self._state.last_error = ""
                    self._state.last_resp  = resp.get('raw', rtype)
                elif rtype == 'unknown':
                    self._state.last_resp = resp.get('raw', '')

    # ── Action dispatch ───────────────────────────────────────────────────

    def _handle_action(self, action: str) -> None:
        """
        Mutate ArmState for the given action, then send the appropriate
        serial command.  IK-affecting actions always send SET ALL.
        Independent axes send targeted single commands.
        """
        state = self._state
        ik_dirty   = False   # needs full IK recompute + SET ALL
        send_fn    = None    # callable → sends the command after state update

        with state._lock:
            # ── IK polar axes ─────────────────────────────────────────────
            if action == 'theta_inc':
                state.theta = min(180.0, state.theta + THETA_STEP)
                ik_dirty = True
            elif action == 'theta_dec':
                state.theta = max(0.0, state.theta - THETA_STEP)
                ik_dirty = True
            elif action == 'r_inc':
                state.r = min(R_MAX, state.r + R_STEP)
                ik_dirty = True
            elif action == 'r_dec':
                state.r = max(R_MIN, state.r - R_STEP)
                ik_dirty = True
            elif action == 'z_inc':
                state.z = min(Z_MAX, state.z + Z_STEP)
                ik_dirty = True
            elif action == 'z_dec':
                state.z = max(Z_MIN, state.z - Z_STEP)
                ik_dirty = True
            # ── Wrist vertical offset (still needs IK recompute) ──────────
            elif action == 'wv_inc':
                state.wrist_offset = min(90.0, state.wrist_offset + WRIST_V_STEP)
                ik_dirty = True
            elif action == 'wv_dec':
                state.wrist_offset = max(-90.0, state.wrist_offset - WRIST_V_STEP)
                ik_dirty = True
            # ── Independent axes ──────────────────────────────────────────
            elif action == 'wr_inc':
                state.wrist_rot = min(180, state.wrist_rot + int(WRIST_R_STEP))
                send_fn = self._send_wrist_rot
            elif action == 'wr_dec':
                state.wrist_rot = max(0, state.wrist_rot - int(WRIST_R_STEP))
                send_fn = self._send_wrist_rot
            elif action == 'grip_open':
                state.gripper = GRIPPER_OPEN
                send_fn = self._send_gripper
            elif action == 'grip_close':
                state.gripper = GRIPPER_CLOSE
                send_fn = self._send_gripper
            # ── Slew rate ─────────────────────────────────────────────────
            elif action == 'delta_inc':
                state.delta = min(DELTA_MAX, state.delta + 1)
                send_fn = self._send_delta
            elif action == 'delta_dec':
                state.delta = max(DELTA_MIN, state.delta - 1)
                send_fn = self._send_delta
            # ── Equilibrium ───────────────────────────────────────────────
            elif action == 'go_home':
                send_fn = self._send_home
            elif action == 'set_equil':
                state.set_equil_from_current()
                # No command to send — just save local state
                with state._lock:
                    state.last_cmd  = "(equil saved)"
                    state.last_resp = ""
                return

        if ik_dirty:
            self._send_ik_move()
        elif send_fn is not None:
            send_fn()

    # ── Command senders ───────────────────────────────────────────────────

    def _send_ik_move(self) -> None:
        """Recompute IK and send SET ALL for the current polar state."""
        state = self._state
        with state._lock:
            theta        = state.theta
            r            = state.r
            z            = state.z
            wrist_offset = state.wrist_offset
            wrist_rot    = state.wrist_rot
            gripper      = state.gripper

        x, y   = polar_to_cartesian(theta, r)
        result = solve_ik(x, y, z)

        if result is None:
            with state._lock:
                state.last_error = (
                    f"IK unreachable: theta={theta:.0f}°  "
                    f"r={r:.0f} mm  z={z:.0f} mm"
                )
            return

        state.update_joints_from_ik(result, wrist_offset, wrist_rot, gripper)
        positions = state.snapshot()['joints']
        cmd = cmd_set_all(positions)
        with state._lock:
            state.last_cmd   = cmd.strip()
            state.last_error = ""
        self._bridge.send_cmd(cmd)

    def _send_wrist_rot(self) -> None:
        with self._state._lock:
            wr  = self._state.wrist_rot
            self._state.joints[JOINT_WRIST_ROT] = wr
            cmd = cmd_set_joint(JOINT_WRIST_ROT, wr)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_gripper(self) -> None:
        with self._state._lock:
            g   = self._state.gripper
            self._state.joints[JOINT_GRIPPER] = g
            cmd = cmd_set_joint(JOINT_GRIPPER, g)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_delta(self) -> None:
        with self._state._lock:
            d   = self._state.delta
            cmd = cmd_set_delta(d)
            self._state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)

    def _send_home(self) -> None:
        """Restore IK state to equilibrium and send HOME command."""
        self._state.restore_equil()
        cmd = cmd_home()
        with self._state._lock:
            self._state.last_cmd   = cmd.strip()
            self._state.last_error = ""
        self._bridge.send_cmd(cmd)
        # Resync joint shadow after home completes
        self._bridge.send_cmd(cmd_get_pos())
