"""
rl_sweeper.py — Drop-in replacement for AutoSweeper backed by a learned RL policy.

Uses the same constructor shape as AutoSweeper (arm_state, bridge, tof_state,
obstacle_map) plus a policy reference.  The policy reference is a hot-swappable
container so the online trainer can replace weights without pausing the arm.

Loop structure (SWEEP_TICK_HZ):
  1. Check IR emergency stop (fast path, halts before any serial command).
  2. Build 74-float observation from state objects (matches rl_recorder layout).
  3. Call policy.predict(obs) → normalised 4-float action.
  4. Denormalize → raw deltas (Δθ°, Δr mm, Δz mm, Δdelta).
  5. Clamp to workspace limits, solve IK, send SET ALL (and SET DELTA if changed).
  6. If an online trainer is attached, push (obs, action, reward, next_obs, done).

Goal conditioning:
  set_goal({'theta': …, 'r': …, 'z': …})   — used by SequenceRLRunner
  In "sweep mode" (no external goal) the sweeper alternates goal_theta between
  SWEEP_THETA_MIN and SWEEP_THETA_MAX so the policy's goal-delta features stay
  aligned with the sweep direction.

Human feedback (optional):
  add_human_feedback(±0.5)  — forwarded to the attached trainer which adjusts
  the last N_FEEDBACK_STEPS transitions in the replay buffer.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional, Any, Callable

import numpy as np

from typing import TYPE_CHECKING

from .ik_solver     import solve_ik, polar_to_cartesian, reachability
from .protocol      import cmd_set_all, cmd_set_delta
from .rl_env        import BraccioBaseEnv, IR_STOP_THRESHOLD
from .rl_recorder   import _encode_obs
from .rl_reward     import compute_reward
from .constants import (
    SWEEP_TICK_HZ, SWEEP_COLLISION_RADIUS_MM,
    DELTA_MIN, DELTA_MAX, R_MIN, R_MAX, Z_MIN, Z_MAX,
    IR_MIN, IR_MONITOR_HZ,
)

# Defer these imports to avoid pulling broken transitive deps at module load.
# The rl_sweeper module only uses these as type hints + runtime duck-typing
# via .snapshot() / .send_cmd() calls.
if TYPE_CHECKING:
    from .arm_state     import ArmState
    from .serial_bridge import SerialBridge
    from .tof_sensor    import ToFState
    from .obstacle_map  import ObstacleMap

log = logging.getLogger(__name__)


# ── Hot-swappable policy reference ────────────────────────────────────────────

class AtomicPolicyRef:
    """
    Thread-safe policy container that allows seamless weight hot-swap.
    Python GIL makes the attribute rebinding atomic in CPython, so .predict()
    always reads a coherent policy reference.
    """

    def __init__(self, policy: Any = None):
        self._policy = policy

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        p = self._policy
        if p is None:
            return np.zeros(4, dtype=np.float32)
        try:
            action, _ = p.predict(obs, deterministic=deterministic)
            return np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception as exc:
            log.warning("Policy.predict() failed: %s — falling back to zero action", exc)
            return np.zeros(4, dtype=np.float32)

    def swap(self, new_policy: Any) -> None:
        """Atomic replace (CPython GIL guarantees this single assignment is atomic)."""
        self._policy = new_policy


# ── Main sweeper ──────────────────────────────────────────────────────────────

class RLSweeper:
    """
    RL-policy-driven sweeper.  Drop-in substitute for AutoSweeper.

    Parameters
    ----------
    arm_state     : ArmState
    bridge        : SerialBridge
    tof_state     : ToFState
    obstacle_map  : ObstacleMap
    policy_ref    : AtomicPolicyRef  — container with .predict(obs)
    trainer       : optional rl_online_trainer.OnlineTrainer
    tick_hz       : control tick rate (default SWEEP_TICK_HZ)
    sweep_theta_min / sweep_theta_max : goal endpoints when no external goal set
    """

    def __init__(
        self,
        arm_state:    "ArmState",
        bridge:       "SerialBridge",
        tof_state:    "ToFState",
        obstacle_map: "ObstacleMap",
        policy_ref:   AtomicPolicyRef,
        trainer:      Optional[Any]  = None,
        tick_hz:      float          = SWEEP_TICK_HZ,
        sweep_theta_min: float = 0.0,
        sweep_theta_max: float = 180.0,
    ):
        self._arm_state    = arm_state
        self._bridge       = bridge
        self._tof_state    = tof_state
        self._obstacle_map = obstacle_map
        self._policy_ref   = policy_ref
        self._trainer      = trainer

        self.tick_hz = float(tick_hz)

        # Goal state (mode-agnostic).  None ⇒ fall back to sweep-mode endpoints.
        self._goal_lock = threading.Lock()
        self._goal: Optional[dict] = None
        self._sweep_theta_min = float(sweep_theta_min)
        self._sweep_theta_max = float(sweep_theta_max)
        self._sweep_goal_theta = self._sweep_theta_max   # starting target

        # Thread control
        self._stop    = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ir_estop = threading.Event()
        self._ir_thread: Optional[threading.Thread] = None

        # Previous transition (for reward computation + trainer push)
        self._prev_obs: Optional[np.ndarray]   = None
        self._prev_action_n: np.ndarray = np.zeros(4, dtype=np.float32)
        self._prev_prev_action_n: np.ndarray = np.zeros(4, dtype=np.float32)

    # ── Public control interface ──────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ir_estop.clear()
        self._prev_obs = None

        self._ir_thread = threading.Thread(
            target=self._ir_monitor_loop, daemon=True,
            name="rl-ir-monitor")
        self._ir_thread.start()

        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name="rl-sweeper")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ir_thread:
            self._ir_thread.join(timeout=1.0)
        self._ir_estop.clear()

    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── Goal conditioning (used by SequenceRLRunner) ─────────────────────────

    def set_goal(self, goal: Optional[dict]) -> None:
        """
        Set an absolute goal pose.  Pass None to fall back to sweep-mode goals.

        goal : {'theta': deg, 'r': mm, 'z': mm}
        """
        with self._goal_lock:
            self._goal = dict(goal) if goal else None

    def get_goal(self) -> Optional[dict]:
        with self._goal_lock:
            return dict(self._goal) if self._goal else None

    def goal_reached(self, tol_theta_deg: float = 5.0,
                     tol_r_mm: float = 8.0,
                     tol_z_mm: float = 8.0) -> bool:
        """True if arm is within tolerance of the current goal."""
        goal = self.get_goal()
        if goal is None:
            return False
        snap = self._arm_state.snapshot()
        return (abs(snap['theta'] - goal['theta']) < tol_theta_deg
                and abs(snap['r']  - goal['r'])   < tol_r_mm
                and abs(snap['z']  - goal['z'])   < tol_z_mm)

    # ── Human feedback routing ────────────────────────────────────────────────

    def add_human_feedback(self, delta: float) -> None:
        if self._trainer is not None and hasattr(self._trainer, "add_human_feedback"):
            self._trainer.add_human_feedback(delta)

    # ── Internal: IR emergency stop monitor ──────────────────────────────────

    def _ir_monitor_loop(self) -> None:
        interval = 1.0 / IR_MONITOR_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            ir = self._tof_state.snapshot().get('ir_bits', 0)
            if ir >= IR_MIN:
                self._ir_estop.set()
            else:
                self._ir_estop.clear()
            self._stop.wait(max(0.0, interval - (time.monotonic() - t0)))

    # ── Current effective goal (external or sweep-mode default) ──────────────

    def _effective_goal(self, snap: dict) -> dict:
        goal = self.get_goal()
        if goal is not None:
            return goal
        # Sweep mode: alternate between theta endpoints
        if abs(snap['theta'] - self._sweep_goal_theta) < 8.0:
            self._sweep_goal_theta = (self._sweep_theta_min
                                      if self._sweep_goal_theta >= self._sweep_theta_max - 1.0
                                      else self._sweep_theta_max)
        return {
            'theta': float(self._sweep_goal_theta),
            'r':     float(snap['r']),
            'z':     float(snap['z']),
        }

    # ── Observation build ─────────────────────────────────────────────────────

    def _build_obs(self, goal: dict) -> np.ndarray:
        arm_snap = self._arm_state.snapshot()
        tof_snap = self._tof_state.snapshot()
        obs_snap = self._obstacle_map.snapshot() if self._obstacle_map is not None else {}
        return _encode_obs(
            arm_snap     = arm_snap,
            tof_snap     = tof_snap,
            obs_snap     = obs_snap,
            direction    = np.sign(goal['theta'] - arm_snap['theta']) or 1.0,
            obstacle_map = self._obstacle_map,
            sweeper      = None,   # z-mask skips _is_position_clear when None
            goal_state   = goal,
        )

    # ── Motion primitives ─────────────────────────────────────────────────────

    def _send_move(self, theta: float, r: float, z: float) -> bool:
        """Solve IK for (theta, r, z) and send SET ALL.  Returns False if blocked."""
        if self._ir_estop.is_set():
            return False
        if reachability(theta, r, z) != 'ok':
            return False

        x, y = polar_to_cartesian(theta, r)
        ik = solve_ik(x, y, z)
        if ik is None:
            return False

        with self._arm_state._lock:
            wrist_off = self._arm_state.wrist_offset
            wrist_rot = self._arm_state.wrist_rot
            gripper   = self._arm_state.gripper
            self._arm_state.theta = theta
            self._arm_state.r     = r
            self._arm_state.z     = z
        self._arm_state.update_joints_from_ik(ik, wrist_off, wrist_rot, gripper)

        positions = self._arm_state.snapshot()['joints']
        cmd = cmd_set_all(positions)
        with self._arm_state._lock:
            self._arm_state.last_cmd = cmd.strip()
        self._bridge.send_cmd(cmd)
        return True

    def _send_delta(self, delta: int) -> None:
        if self._ir_estop.is_set():
            return
        delta = int(max(DELTA_MIN, min(DELTA_MAX, delta)))
        with self._arm_state._lock:
            if self._arm_state.delta == delta:
                return
            self._arm_state.delta = delta
        self._bridge.send_cmd(cmd_set_delta(delta))

    # ── Main control loop ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        tick = 1.0 / self.tick_hz
        while not self._stop.is_set():
            t0 = time.monotonic()

            # Fast-path: IR emergency stop halts everything before any serial I/O
            if self._ir_estop.is_set():
                self._record_done_if_needed()
                self._stop.wait(max(0.0, tick - (time.monotonic() - t0)))
                continue

            arm_snap = self._arm_state.snapshot()
            goal     = self._effective_goal(arm_snap)
            obs      = self._build_obs(goal)

            # Policy inference (atomic read via GIL)
            action_n = self._policy_ref.predict(obs, deterministic=True)
            action_n = np.clip(action_n, -1.0, 1.0).astype(np.float32)
            raw      = BraccioBaseEnv.denormalize_action(action_n)

            # Commanded pose (clamped)
            new_theta = float(np.clip(arm_snap['theta'] + raw[0], 0.0, 180.0))
            new_r     = float(np.clip(arm_snap['r']     + raw[1], R_MIN, R_MAX))
            new_z     = float(np.clip(arm_snap['z']     + raw[2], Z_MIN, Z_MAX))
            new_delta = int(np.clip(round(arm_snap['delta'] + raw[3]),
                                    DELTA_MIN, DELTA_MAX))

            self._send_move(new_theta, new_r, new_z)
            self._send_delta(new_delta)

            # Online training: push transition
            if self._trainer is not None and self._prev_obs is not None:
                info = self._build_info_for_reward()
                reward = compute_reward(
                    obs           = self._prev_obs,
                    action_n      = self._prev_action_n,
                    next_obs      = obs,
                    prev_action_n = self._prev_prev_action_n,
                    info          = info,
                )
                try:
                    self._trainer.add_transition(
                        self._prev_obs, self._prev_action_n, reward, obs, done=False)
                except Exception as exc:
                    log.debug("trainer.add_transition failed: %s", exc)

            self._prev_obs            = obs
            self._prev_prev_action_n  = self._prev_action_n
            self._prev_action_n       = action_n

            self._stop.wait(max(0.0, tick - (time.monotonic() - t0)))

        # Final cleanup: record last done=True transition
        self._record_done_if_needed(done=True)

    def _build_info_for_reward(self) -> dict:
        """Approximate info dict consumed by compute_reward()."""
        tof_snap = self._tof_state.snapshot()
        grids    = tof_snap.get('grids', [None, None, None, None])

        def _min_mm(g):
            if g is None:
                return 4000.0
            arr = np.asarray(g, dtype=np.float32).flatten()
            valid = arr[~np.isnan(arr)]
            return float(valid.min()) if valid.size > 0 else 4000.0

        ir_bits = int(tof_snap.get('ir_bits', 0))
        obs_response = 'clear'
        if ir_bits >= IR_STOP_THRESHOLD:
            obs_response = 'back_away'
        elif min(_min_mm(grids[0] if len(grids) > 0 else None),
                 _min_mm(grids[1] if len(grids) > 1 else None)) < 250.0:
            obs_response = 'replan'

        return {
            'obstacle_response': obs_response,
            'ir_bits':           ir_bits,
            'ch0_min_mm':        _min_mm(grids[0] if len(grids) > 0 else None),
            'ch1_min_mm':        _min_mm(grids[1] if len(grids) > 1 else None),
        }

    def _record_done_if_needed(self, done: bool = False) -> None:
        """Terminal-transition bookkeeping (e.g. emergency stop or shutdown)."""
        if self._trainer is None or self._prev_obs is None:
            return
        if not done:
            return
        try:
            # Use current obs as the terminal next_obs
            arm_snap = self._arm_state.snapshot()
            goal     = self._effective_goal(arm_snap)
            final_obs = self._build_obs(goal)
            info = self._build_info_for_reward()
            reward = compute_reward(
                obs           = self._prev_obs,
                action_n      = self._prev_action_n,
                next_obs      = final_obs,
                prev_action_n = self._prev_prev_action_n,
                info          = info,
            )
            self._trainer.add_transition(
                self._prev_obs, self._prev_action_n, reward, final_obs, done=True)
        except Exception as exc:
            log.debug("terminal add_transition failed: %s", exc)
        finally:
            self._prev_obs = None


# ── Safe hardware wrapper for fine-tuning ─────────────────────────────────────

class SafeHardwareEnv:
    """
    Thin safety wrapper used during hardware SAC fine-tuning.  Clamps action
    magnitudes for the first WARMUP_STEPS so the policy cannot lurch at full
    speed while still uncertain.  Forces a retreat action when ir_bits reaches
    the emergency threshold.
    """

    WARMUP_STEPS       = 10_000
    WARMUP_ACTION_CLIP = 0.5   # half-magnitude during warm-up

    def __init__(self, sweeper: RLSweeper):
        self._sweeper = sweeper
        self._step_count = 0
        self._prev_action_n = np.zeros(4, dtype=np.float32)

    def filter_action(self, action_n: np.ndarray) -> np.ndarray:
        action_n = np.asarray(action_n, dtype=np.float32).reshape(-1)

        ir = int(self._sweeper._tof_state.snapshot().get('ir_bits', 0))
        if ir >= IR_STOP_THRESHOLD:
            # Reverse last commanded motion at half magnitude
            return np.clip(-self._prev_action_n * 0.5, -1.0, 1.0)

        if self._step_count < self.WARMUP_STEPS:
            action_n = np.clip(action_n,
                               -self.WARMUP_ACTION_CLIP,
                                self.WARMUP_ACTION_CLIP)

        self._prev_action_n = action_n.copy()
        self._step_count   += 1
        return np.clip(action_n, -1.0, 1.0)
