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
  2. Queries ObstacleMap.get_obstacle_thetas() to find the occupied theta band.
  3. Computes a bypass theta just past the far edge of the obstacle + margin.
  4. Jumps the sweep target to the bypass theta so the arm arcs around the
     obstacle, then restores normal speed.
  5. If no bypass is possible (obstacle spans the whole sweep range) the arm
     waits in place until the obstacle clears.

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
from .serial_bridge import SerialBridge
from .ik_solver    import solve_ik, polar_to_cartesian, reachability
from .obstacle_map import ObstacleMap
from .tof_sensor   import ToFState, ObstacleResponse
from .protocol     import cmd_set_all, cmd_set_delta

from .constants    import (
    SWEEP_THETA_MIN, SWEEP_THETA_MAX,
    SWEEP_R_DEFAULT, SWEEP_Z_DEFAULT,
    SWEEP_STEP_DEG, SWEEP_TICK_HZ,
    SWEEP_DELTA_NORMAL, SWEEP_DELTA_REPLAN,
    SWEEP_BACK_STEPS, SWEEP_OBSTACLE_MARGIN_DEG,
    SWEEP_Z_CANDIDATES,
    SWEEP_COLLISION_RADIUS_MM,
    DELTA_MIN, DELTA_MAX,
    R_MIN, R_MAX, Z_MIN, Z_MAX,
)


class SweepState(enum.Enum):
    IDLE             = 'idle'
    RUNNING          = 'running'
    PAUSED_OBSTACLE  = 'paused_obstacle'
    BACK_AWAY        = 'back_away'


class AutoSweeper:
    """
    Daemon-thread sweep controller.  Must be constructed after the
    SerialBridge and ArmState are ready.

    Parameters
    ----------
    arm_state   : shared ArmState (read + written under its lock)
    bridge      : SerialBridge for sending SET ALL / SET DELTA commands
    tof_state   : ToFState for obstacle response polling
    obstacle_map: ObstacleMap for world-frame forbidden zone queries
    """

    def __init__(self,
                 arm_state:    ArmState,
                 bridge:       SerialBridge,
                 tof_state:    ToFState,
                 obstacle_map: ObstacleMap):
        self._arm_state    = arm_state
        self._bridge       = bridge
        self._tof_state    = tof_state
        self._obstacle_map = obstacle_map

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

        # ── Thread control ────────────────────────────────────────────────────
        self._stop  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()   # protects sweep runtime state

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
                with self._state_lock:
                    if self._sweep_state in (SweepState.PAUSED_OBSTACLE,
                                             SweepState.BACK_AWAY):
                        # Obstacle cleared — resume normal sweep
                        self._sweep_state = SweepState.RUNNING
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
        # Always attempt bypass on every tick — no 10-second early-return.
        # The old code blocked here for _obstacle_wait_timeout_s (10 s),
        # which prevented Z-axis and theta bypass from being retried.
        with self._state_lock:
            if self._sweep_state != SweepState.PAUSED_OBSTACLE:
                self._obstacle_wait_start = time.time()
            self._sweep_state = SweepState.PAUSED_OBSTACLE

        # Slow down for smooth deceleration / replanning
        self._send_delta(self.delta_replan)

        with self._state_lock:
            cur_theta = self._current_theta

        # ── Step 1: theta bypass (horizontal avoidance) ───────────────────────
        # Merge rolling cloud + persistent memory for a complete obstacle picture
        forbidden = self._obstacle_map.get_obstacle_thetas(
            self.sweep_r, self.sweep_z
        )
        mem_thetas = self._obstacle_map.memory_occupied_thetas(
            self.sweep_r, self.sweep_z
        )
        forbidden.extend(mem_thetas)

        # Include Kalman estimate of obstacle that left the sensor FOV
        est = self._obstacle_map.estimate_lost_obstacle()
        if est is not None:
            est_theta = math.degrees(math.atan2(est[1], est[0]))
            est_theta = max(0.0, min(180.0, est_theta))
            forbidden.append(est_theta)

        bypass_theta = self._compute_bypass_theta(forbidden)

        if bypass_theta is not None:
            with self._state_lock:
                self._bypass_theta  = bypass_theta
                self._current_theta = bypass_theta
                self._sweep_state   = SweepState.RUNNING
            self._send_move(bypass_theta)
            self._send_delta(self.delta_normal)
            return

        # ── Step 2: Z-axis avoidance (go over or under the obstacle) ─────────
        # Retried every tick so the arm reacts as soon as a Z level clears.
        new_z = self._find_clear_z(cur_theta)
        if new_z is not None:
            with self._state_lock:
                self.sweep_z       = new_z
                self._bypass_theta = None
                self._sweep_state  = SweepState.RUNNING
            self._send_move(cur_theta, z=new_z)
            self._send_delta(self.delta_normal)
            return

        # ── Step 3: hold position — obstacle spans all options ────────────────
        # Will keep retrying steps 1 + 2 on every tick until clear.

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
        Returns True if command was sent, False if blocked by collision check.
        z defaults to self.sweep_z if omitted.
        """
        z = z if z is not None else self.sweep_z

        if not self._is_position_clear(theta, z):
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
        cmd = cmd_set_all(positions)
        with self._arm_state._lock:
            self._arm_state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)
        return True

    def _send_delta(self, delta: int) -> None:
        """Send SET DELTA command and update arm state."""
        delta = max(DELTA_MIN, min(DELTA_MAX, delta))
        with self._arm_state._lock:
            self._arm_state.delta = delta
        self._bridge.send_cmd(cmd_set_delta(delta))
