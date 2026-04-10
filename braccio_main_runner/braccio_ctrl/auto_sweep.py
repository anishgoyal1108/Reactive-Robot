"""
auto_sweep.py — Autonomous theta 0→180 sweep with obstacle-aware replanning.

The AutoSweeper runs a daemon thread that advances the arm's base angle
(theta) from SWEEP_THETA_MIN to SWEEP_THETA_MAX and back at a configurable
rate, while checking the ToF/IR obstacle state on every tick.

State machine
-------------
  IDLE               — not running
  RUNNING            — normal sweep in progress
  PAUSED_OBSTACLE    — obstacle detected; computing bypass theta
  BACK_AWAY          — IR DANGER/CLOSE; retreating before replanning
  DONE               — (unused — sweep loops indefinitely until stop())

Replanning strategy
-------------------
When a REPLAN response is received the sweeper:
  1. Slows the arm (increases SET DELTA) for smoother deceleration.
  2. Reverses sweep direction and retreats by SWEEP_RETREAT_DEG. The retreat
     must exceed the ToF half-FOV so the on-arm sensor actually rotates off
     the obstacle; jumping forward past the obstacle would just move the
     sensor cone closer to it.
  3. Commits the reversal and returns. The clear-branch in _sweep_loop then
     resets the commit on the first tick the obstacle clears, and normal
     sweeping resumes in the new direction.
  4. Tertiary fallback: if the arm is pinned against a sweep limit and the
     retreat delta is too small to matter, try a Z-axis hop (go over or
     under) after a short hold. If that also fails, hold position.

Kalman estimation
-----------------
When the obstacle leaves sensor FOV after the arm passes it,
ObstacleMap.estimate_lost_obstacle() provides a predicted world position.
The sweeper uses this to expand the forbidden zone conservatively so it
does not swing back into a recently observed but temporarily invisible obstacle.
"""

import enum
import math
import threading
import time
from typing import Optional

from .arm_state    import ArmState
from .backends     import BraccioBackend
from .ik_solver    import solve_ik, polar_to_cartesian
from .obstacle_map import ObstacleMap
from .motion_guard import MotionGuard
from .tof_sensor   import ToFState, ObstacleResponse
from .protocol     import cmd_set_all, cmd_set_delta

from .constants    import (
    SWEEP_THETA_MIN, SWEEP_THETA_MAX,
    SWEEP_R_DEFAULT, SWEEP_Z_DEFAULT,
    SWEEP_STEP_DEG, SWEEP_TICK_HZ,
    SWEEP_DELTA_NORMAL, SWEEP_DELTA_REPLAN,
    SWEEP_BACK_STEPS, SWEEP_OBSTACLE_MARGIN_DEG,
    SWEEP_RETREAT_DEG,
    SWEEP_MIN_CMD_INTERVAL_S,
    SWEEP_MIN_DELTA_DEG,
    DELTA_MIN, DELTA_MAX,
)


class SweepState(enum.Enum):
    IDLE             = 'idle'
    RUNNING          = 'running'
    PAUSED_OBSTACLE  = 'paused_obstacle'
    BACK_AWAY        = 'back_away'


class AutoSweeper:
    """
    Daemon-thread sweep controller.  Must be constructed after the
    backend and ArmState are ready.

    Parameters
    ----------
    arm_state   : shared ArmState (read + written under its lock)
    bridge      : BraccioBackend for sending SET ALL / SET DELTA commands
                  (hardware SerialBridge or SimBackend — same interface)
    tof_state   : ToFState for obstacle response polling
    obstacle_map: ObstacleMap for world-frame forbidden zone queries
    """

    def __init__(self,
                 arm_state:    ArmState,
                 bridge:       BraccioBackend,
                 tof_state:    ToFState,
                 obstacle_map: ObstacleMap,
                 guard:        MotionGuard):
        self._arm_state    = arm_state
        self._bridge       = bridge
        self._tof_state    = tof_state
        self._obstacle_map = obstacle_map
        self._guard        = guard

        # ── Configurable sweep parameters ────────────────────────────────────
        self.sweep_r:           float = SWEEP_R_DEFAULT
        self.sweep_z:           float = SWEEP_Z_DEFAULT
        self.sweep_theta_min:   float = SWEEP_THETA_MIN
        self.sweep_theta_max:   float = SWEEP_THETA_MAX
        self.step_deg:          float = SWEEP_STEP_DEG
        self.tick_hz:           float = SWEEP_TICK_HZ
        self.delta_normal:      int   = SWEEP_DELTA_NORMAL
        self.delta_replan:      int   = SWEEP_DELTA_REPLAN

        # ── Runtime state ─────────────────────────────────────────────────────
        self._sweep_state: SweepState = SweepState.IDLE
        self._direction:   int        = 1      # +1 → forward (theta increasing)
        self._current_theta: float    = SWEEP_THETA_MIN
        self._bypass_theta: Optional[float] = None
        self._obstacle_wait_start: float = 0.0
        self._obstacle_wait_timeout_s: float = 10.0  # max wait before skipping

        # Replan commit state: once we pick a plan (bypass theta OR Z hop OR
        # "hold position"), we stick with it until the obstacle actually
        # clears. Without this, _do_replan re-ran every 100 ms and produced
        # a fresh plan each tick — the arm oscillated between z=55 and z=80
        # at 10 Hz, which is what caused the physical jitter.
        self._replan_committed: bool = False
        self._replan_enter_time: float = 0.0
        # How long to hold position before attempting a Z-axis bypass.
        # Theta bypass is still tried immediately on entry.
        self._replan_hold_before_z_s: float = 2.0

        # ── Thread control ────────────────────────────────────────────────────
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()   # protects sweep runtime state

        # ── Command rate-limiting state ──────────────────────────────────────
        # Track when the last SET ALL was actually flushed and what joint
        # positions it asked for. _send_move skips commands that arrive too
        # soon after the previous one unless they move a joint by more than
        # SWEEP_MIN_DELTA_DEG. Keeps the serial link from being flooded when
        # a sensor reading flaps every tick.
        self._last_cmd_time: float = 0.0
        self._last_cmd_positions: Optional[list] = None

    # ── Public control interface ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the sweep from the current arm theta position."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Sync starting theta from arm state
        with self._arm_state._lock:
            self._current_theta = self._arm_state.theta
        with self._state_lock:
            self._sweep_state = SweepState.RUNNING
            self._direction   = 1

        self._send_delta(self.delta_normal)
        self._thread = threading.Thread(
            target=self._sweep_loop,
            daemon=True,
            name='auto-sweeper',
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the sweep and restore normal delta."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._state_lock:
            self._sweep_state = SweepState.IDLE
        self._send_delta(self.delta_normal)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        """Thread-safe status dict for the display."""
        with self._state_lock:
            return {
                'sweep_state':    self._sweep_state.value,
                'current_theta':  round(self._current_theta, 1),
                'direction':      self._direction,
                'sweep_r':        self.sweep_r,
                'sweep_z':        self.sweep_z,
                'bypass_theta':   self._bypass_theta,
                'running':        self.is_running(),
            }

    # ── Sweep loop (daemon thread) ────────────────────────────────────────────

    def _sweep_loop(self) -> None:
        tick = 1.0 / self.tick_hz
        while not self._stop.is_set():
            t0 = time.monotonic()

            obs_response = self._tof_state.snapshot()['obstacle_response']

            if obs_response == ObstacleResponse.BACK_AWAY:
                self._do_back_away()
            elif obs_response == ObstacleResponse.REPLAN:
                self._do_replan()
            else:
                # Obstacle cleared — resume normal sweep and throw away any
                # committed replan so a fresh obstacle triggers a fresh plan.
                resumed = False
                with self._state_lock:
                    if self._sweep_state in (SweepState.PAUSED_OBSTACLE,
                                             SweepState.BACK_AWAY):
                        self._sweep_state = SweepState.RUNNING
                        self._replan_committed = False
                        self._bypass_theta = None
                        resumed = True
                if resumed:
                    self._send_delta(self.delta_normal)
                self._do_step()

            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, tick - elapsed)
            self._stop.wait(sleep_t)

    # ── Step (normal motion) ──────────────────────────────────────────────────

    def _do_step(self) -> None:
        with self._state_lock:
            state = self._sweep_state
        if state != SweepState.RUNNING:
            return

        with self._state_lock:
            self._current_theta += self._direction * self.step_deg
            # Bounce at limits
            if self._current_theta >= self.sweep_theta_max:
                self._current_theta = self.sweep_theta_max
                self._direction = -1
            elif self._current_theta <= self.sweep_theta_min:
                self._current_theta = self.sweep_theta_min
                self._direction = 1
            theta = self._current_theta

        self._send_move(theta)

    # ── Replan (obstacle in primary FOV) ─────────────────────────────────────

    def _do_replan(self) -> None:
        """
        On a fresh REPLAN entry: reverse sweep direction and retreat by
        SWEEP_RETREAT_DEG. Then commit — the tick loop resumes normal
        stepping in the new direction, which naturally carries the arm
        (and the on-arm ToF sensor) AWAY from the obstacle.

        Why reverse-and-retreat instead of "jump past the obstacle":
        the primary ToF sensors are mounted on the end-effector and
        rotate with the arm. Jumping 15° forward past a detected hand
        moves the sensor *closer* to that hand, not farther — the arm
        ends up staring at the obstacle from a slightly worse angle.
        Retreating > FOV/2 = 22.5° rotates the sensor cone off the
        obstacle entirely. As soon as the sensor clears, the clear
        branch in _sweep_loop resets committed and the arm continues
        sweeping in the new direction.

        Tertiary fallback (Z hop) is kept for the edge case where the
        arm is jammed up against a sweep limit and cannot retreat.
        """
        with self._state_lock:
            is_fresh_entry = self._sweep_state != SweepState.PAUSED_OBSTACLE
            self._sweep_state = SweepState.PAUSED_OBSTACLE
            if is_fresh_entry:
                self._replan_enter_time = time.time()
                self._replan_committed = False
                self._bypass_theta = None
                self._obstacle_wait_start = time.time()
            committed = self._replan_committed
            cur_theta = self._current_theta
            cur_direction = self._direction
            cur_z = self.sweep_z

        # Slow the arm ONCE on entry — sending SET DELTA every tick was
        # also contributing to serial-bus noise.
        if is_fresh_entry:
            self._send_delta(self.delta_replan)

        # Plan already committed and we're waiting for the obstacle to
        # clear (or for the Z-hop grace period to expire, handled below).
        if committed:
            return

        # ── Step 1: reverse-and-retreat (primary strategy) ───────────────────
        new_direction = -cur_direction
        retreat_target = cur_theta + new_direction * SWEEP_RETREAT_DEG
        retreat_target = max(
            self.sweep_theta_min,
            min(self.sweep_theta_max, retreat_target),
        )
        retreat_delta = abs(retreat_target - cur_theta)

        # If we have headroom to actually move, commit the reversal now.
        # The retreat delta must be larger than one normal step to count
        # as progress; otherwise we're pinned against a sweep limit and
        # need to fall through to the Z-hop fallback.
        if retreat_delta >= max(2.0, self.step_deg):
            with self._state_lock:
                self._direction       = new_direction
                self._current_theta   = retreat_target
                self._bypass_theta    = retreat_target
                self._replan_committed = True
            self._send_move(retreat_target, z=cur_z)
            return

        # ── Step 2: Z-axis hop (tertiary fallback) ───────────────────────────
        # Reached here only when cur_theta is already at (or within one
        # step of) the sweep limit on the retreat side — reversing can't
        # create meaningful separation. Try going over/under the obstacle
        # after a short hold so a transient reading doesn't trigger a hop.
        if (time.time() - self._replan_enter_time) < self._replan_hold_before_z_s:
            return   # hold position, try again on a later tick

        new_z = self._guard.find_clear_z(cur_theta, self.sweep_r, cur_z)
        if new_z is not None:
            with self._state_lock:
                self.sweep_z           = new_z
                self._bypass_theta     = None
                self._replan_committed = True
            self._send_move(cur_theta, z=new_z)
            return

        # ── Step 3: no plan exists — commit to holding position ──────────────
        # Pinned against a limit, no Z hop available. Hold until the clear
        # branch in _sweep_loop resets committed and resumes the sweep.
        with self._state_lock:
            self._replan_committed = True

    # ── Back away (IR DANGER / CLOSE) ────────────────────────────────────────

    def _do_back_away(self) -> None:
        with self._state_lock:
            if self._sweep_state == SweepState.BACK_AWAY:
                return   # already retreating
            self._sweep_state = SweepState.BACK_AWAY

        self._send_delta(self.delta_replan)

        # Step backwards away from the obstacle
        with self._state_lock:
            retreat = -self._direction * self.step_deg * SWEEP_BACK_STEPS
            self._current_theta = max(
                self.sweep_theta_min,
                min(self.sweep_theta_max, self._current_theta + retreat)
            )
            theta = self._current_theta

        self._send_move(theta)

    # ── Bypass computation ────────────────────────────────────────────────────

    def _compute_bypass_theta(self, forbidden: list) -> Optional[float]:
        """
        Given a list of occupied theta values, find the nearest clear theta
        on the far side of the obstacle in the current sweep direction.

        Returns None if no bypass exists within the sweep range.
        """
        if not forbidden:
            return None

        margin = SWEEP_OBSTACLE_MARGIN_DEG
        with self._state_lock:
            direction = self._direction

        # Find the extent of the obstacle band
        obs_min = min(forbidden) - margin
        obs_max = max(forbidden) + margin

        if direction > 0:
            # Sweeping forward (theta increasing) — bypass past far edge
            candidate = obs_max + margin
        else:
            # Sweeping backward (theta decreasing) — bypass past near edge
            candidate = obs_min - margin

        candidate = max(self.sweep_theta_min,
                        min(self.sweep_theta_max, candidate))

        # Sanity: if candidate is still inside forbidden, bail
        if obs_min <= candidate <= obs_max:
            return None

        return round(candidate, 1)

    # ── Command helpers ───────────────────────────────────────────────────────

    def _send_move(self, theta: float, z: float = None) -> bool:
        """
        Compute IK for (theta, sweep_r, z) and send SET ALL.
        Returns True if command was sent, False if blocked by collision check
        or suppressed by the rate limiter.
        z defaults to self.sweep_z if omitted.
        """
        z = z if z is not None else self.sweep_z

        if not self._guard.is_clear(theta, self.sweep_r, z):
            return False

        x, y   = polar_to_cartesian(theta, self.sweep_r)
        result = solve_ik(x, y, z)
        if result is None:
            return False   # kinematically unreachable — skip this step

        with self._arm_state._lock:
            wrist_offset = self._arm_state.wrist_offset
            wrist_rot    = self._arm_state.wrist_rot
            gripper      = self._arm_state.gripper
            self._arm_state.theta = theta
            self._arm_state.r     = self.sweep_r
            self._arm_state.z     = z
        self._arm_state.update_joints_from_ik(result, wrist_offset,
                                               wrist_rot, gripper)
        positions = self._arm_state.snapshot()['joints']

        # Rate-limit: if this command is both too recent AND numerically
        # close to the previous one, skip sending it. We apply the limiter
        # AFTER updating local state so the shadow tracks what the sweeper
        # *intended*; the serial bus just doesn't see a fresh packet.
        now = time.monotonic()
        if self._last_cmd_positions is not None:
            dt = now - self._last_cmd_time
            max_delta = max(
                abs(a - b)
                for a, b in zip(positions, self._last_cmd_positions)
            )
            if dt < SWEEP_MIN_CMD_INTERVAL_S and max_delta < SWEEP_MIN_DELTA_DEG:
                return False

        cmd = cmd_set_all(positions)
        with self._arm_state._lock:
            self._arm_state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)
        self._last_cmd_time = now
        self._last_cmd_positions = list(positions)
        return True

    def _send_delta(self, delta: int) -> None:
        """Send SET DELTA command and update arm state."""
        delta = max(DELTA_MIN, min(DELTA_MAX, delta))
        with self._arm_state._lock:
            self._arm_state.delta = delta
        self._bridge.send_cmd(cmd_set_delta(delta))
