"""
rl_sequence_runner.py — Goal-conditioned SequenceRunner that delegates motion
to an RLSweeper instead of running direct SET ALL commands.

Replaces the standard `SequenceRunner` when the controller is launched with
`--rl-policy`.  For each sequence step it:

  1. Looks up the named state in the StateLibrary.
  2. Calls `sweeper.set_goal({theta, r, z})`.
  3. Waits until `sweeper.goal_reached()` returns True (or times out).
  4. Sleeps for the step's `wait_ms`.
  5. Clears the goal so the sweeper falls back to sweep-mode defaults.

The RL policy's three goal-delta features (obs[67:70]) automatically encode
which waypoint to move toward — no changes to the sequence file format or the
StateLibrary are needed.

This runner exposes the same public attributes the curses editor reads:
  running, current_step, total_steps, last_error, current_pass, total_passes.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional, Iterable, Any

log = logging.getLogger(__name__)

# Tolerances for goal_reached() — balanced vs. policy jitter at the waypoint.
DEFAULT_TOL_THETA_DEG = 5.0
DEFAULT_TOL_R_MM      = 10.0
DEFAULT_TOL_Z_MM      = 10.0

DEFAULT_STEP_TIMEOUT_S = 15.0   # give up waiting for goal after this many seconds
POLL_INTERVAL_S        = 0.05


class SequenceRLRunner:
    """
    Goal-conditioned sequence runner.

    Parameters
    ----------
    steps           : iterable of {'name': str, 'wait_ms': int} dicts.
                      `name` is a key in `state_library`.
    state_library   : StateLibrary — looks up absolute (theta, r, z) per name.
    sweeper         : RLSweeper with set_goal / goal_reached / get_goal.
    passes          : total repetition count (default 1).
    tol_theta_deg, tol_r_mm, tol_z_mm : goal-reached tolerances.
    step_timeout_s  : max seconds to wait for a single goal.
    """

    def __init__(
        self,
        steps:         Iterable[dict],
        state_library: Any,
        sweeper:       Any,
        passes:         int   = 1,
        tol_theta_deg:  float = DEFAULT_TOL_THETA_DEG,
        tol_r_mm:       float = DEFAULT_TOL_R_MM,
        tol_z_mm:       float = DEFAULT_TOL_Z_MM,
        step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
    ):
        self._steps         = list(steps)
        self._state_library = state_library
        self._sweeper       = sweeper
        self._tol_theta_deg = float(tol_theta_deg)
        self._tol_r_mm      = float(tol_r_mm)
        self._tol_z_mm      = float(tol_z_mm)
        self._step_timeout  = float(step_timeout_s)

        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Legacy status fields the curses editor reads directly
        self.running:        bool = False
        self.current_step:   int  = 0
        self.current_pass:   int  = 0
        self.total_steps:    int  = len(self._steps)
        self.total_passes:   int  = int(max(1, passes))
        self.current_line:   int  = 0
        self.current_kind:   str  = ""
        self.last_error:     str  = ""

    # ── Public control interface ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.running      = True
        self.current_step = 0
        self.current_pass = 0
        self.last_error   = ""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rl-sequence-runner")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._sweeper.set_goal(None)   # clear any pending goal
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def stopped(self) -> bool:
        return self._stop.is_set()

    # ── Run loop ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            for p in range(self.total_passes):
                if self._stop.is_set():
                    break
                self.current_pass = p + 1
                self.current_step = 0

                for i, step in enumerate(self._steps):
                    if self._stop.is_set():
                        break
                    self.current_step = i + 1
                    self.current_line = i
                    self.current_kind = "MoveStmt"

                    self._execute_step(step)

                    wait_ms = int(step.get('wait_ms', 0) or 0)
                    if wait_ms > 0:
                        self._stop.wait(wait_ms / 1000.0)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.exception("SequenceRLRunner crashed")
        finally:
            self._sweeper.set_goal(None)
            self.running = False

    def _execute_step(self, step: dict) -> None:
        name = step.get('name') or ""
        if not name:
            self.last_error = "step missing 'name'"
            return

        target = self._state_library.get_state(name)
        if target is None:
            self.last_error = f"state '{name}' not found"
            log.warning("SequenceRLRunner: unknown state %r — skipping", name)
            return

        goal = {
            'theta': float(target.get('theta', 90.0)),
            'r':     float(target.get('r',    152.0)),
            'z':     float(target.get('z',     60.0)),
        }
        self._sweeper.set_goal(goal)

        # Wait until the sweeper reports goal_reached or timeout fires
        deadline = time.monotonic() + self._step_timeout
        while not self._stop.is_set():
            if self._sweeper.goal_reached(
                    self._tol_theta_deg, self._tol_r_mm, self._tol_z_mm):
                return
            if time.monotonic() > deadline:
                self.last_error = f"timeout reaching state '{name}'"
                log.warning("SequenceRLRunner: timeout on %r (%.1fs)",
                            name, self._step_timeout)
                return
            time.sleep(POLL_INTERVAL_S)
