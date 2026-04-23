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

from .ik_solver     import solve_ik, polar_to_cartesian, reachability, fk_polar
from .protocol      import cmd_set_all, cmd_set_delta
from .rl_env        import BraccioBaseEnv, IR_STOP_THRESHOLD
from .rl_recorder   import _encode_obs
from .rl_reward     import compute_reward
from .constants import (
    SWEEP_TICK_HZ, SWEEP_COLLISION_RADIUS_MM,
    DELTA_MIN, DELTA_MAX, R_MIN, R_MAX, Z_MIN, Z_MAX,
    IR_MIN, IR_MONITOR_HZ,
)

# Stall detector: after this many consecutive ticks where ``_send_move``
# rejected the policy's target (unreachable or zero-motion), the sweeper
# escalates to a safety-BT fallback plan so something moves. 40 ticks at
# SWEEP_TICK_HZ (20 Hz) ~= 2 s — long enough to not churn on jitter, short
# enough that the user doesn't perceive a long hang.
_STALL_THRESHOLD      = 40
# Log the first warning after this many consecutive stalls; then re-log
# every ``_STALL_LOG_EVERY_N`` thereafter so the log doesn't get spammed.
_STALL_LOG_FIRST      = 5
_STALL_LOG_EVERY_N    = 100
# Pause RLSweeper command dispatch for this many ticks after firing a
# fallback. Without this, the sweeper's per-tick _send_move() at 20 Hz
# interleaves with the BT's EMA-smoothed manual intent commands on the
# same serial bridge — the BT's slow motion is overwritten by the
# sweeper's near-zero "stay in place" commands and the arm never moves.
_FALLBACK_QUIET_TICKS = 60            # ~3 s at 20 Hz — enough for BT to walk
# Re-queue the fallback every N ticks of continued stall. The first
# fallback gets swallowed sometimes; periodic re-queue guarantees
# something eventually moves.
_FALLBACK_REFIRE_EVERY = 80           # ~4 s at 20 Hz
# Auto-abort the RLSweeper after this many consecutive stall ticks
# — the policy is demonstrably broken; stopping the sweep restores
# manual control immediately so the user doesn't have to press Z.
_STALL_ABORT_THRESHOLD = 200          # ~10 s at 20 Hz
# Threshold below which a commanded pose is treated as "no motion" even
# if reachability/IK nominally succeeded. Matches the ~0.5° joint
# quantisation noise floor.
_NO_MOTION_EPS        = 0.5  # mm / deg combined L-infinity on (θ, r, z)

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

    Failure handling: on prediction exception the first failure is
    log.error()'d (not just warning) and ``last_error`` is set so the
    controller's UI can surface it. Subsequent identical failures are
    rate-limited to one log every 100 calls so a crashed policy doesn't
    spam the log. Zero action is returned as a safe default, which the
    sweep tick applies — the arm stalls rather than lurches.
    """

    _LOG_EVERY_N = 100

    def __init__(self, policy: Any = None):
        self._policy = policy
        self.last_error: str | None = None
        self._error_count: int = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        p = self._policy
        if p is None:
            return np.zeros(4, dtype=np.float32)
        try:
            action, _ = p.predict(obs, deterministic=deterministic)
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"policy produced non-finite action: {arr.tolist()}")
            self.last_error = None  # clear on success after a failure
            return arr
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            self._error_count += 1
            if self._error_count == 1 or self._error_count % self._LOG_EVERY_N == 0:
                log.error(
                    "RLSweeper policy.predict() failed [#%d]: %s — "
                    "arm will stall until this is fixed. Check the policy "
                    "model path and the obs vector shape.",
                    self._error_count, err,
                )
            self.last_error = err
            return np.zeros(4, dtype=np.float32)

    def swap(self, new_policy: Any) -> None:
        """Atomic replace (CPython GIL guarantees this single assignment is atomic)."""
        self._policy = new_policy
        self._error_count = 0
        self.last_error = None


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
        safety_api:   Optional[Any]  = None,
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
        # Optional SafetyAPI handle — when the policy stalls for
        # _STALL_THRESHOLD consecutive ticks we queue_manual_intent() on
        # it to invoke the BT's cascading replanner. Without this the
        # deterministic policy can infinite-loop on an unreachable target.
        self._safety_api   = safety_api
        # Public read-only view — the controller's UI checks
        # ``policy_ref.last_error`` to surface prediction failures.
        self.policy_ref = policy_ref

        # Stall tracking (fix for deterministic-policy infinite-loop bug).
        self._stall_count:       int  = 0
        self.stall_warned:       bool = False   # surfaced in controller status
        self.stall_aborted:      bool = False   # set when auto-abort fires
        self._quiet_ticks_left:  int  = 0       # skip _send_move while > 0
        self._last_fallback_tick: int = -10_000 # last tick the fallback fired

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

    # ── Stall fallback (Fix B) ────────────────────────────────────────────────

    def _dispatch_stall_fallback(self, arm_snap: dict) -> None:
        """Queue a safety-BT manual intent that flips the sweep direction.

        Called after ``_STALL_THRESHOLD`` consecutive stalls. The target is
        a mid-workspace pose on the opposite side of the sweep arc, solved
        via the same ``solve_ik`` the BT uses internally. The BT's
        ``ManualBranch`` then runs the cascading replanner (polar-skip →
        Z-ladder → BiRRT → direction-flip) to walk the arm there around
        any known obstacles. Once achieved, BT returns to idle and the
        policy resumes from the new pose — the deterministic stall is
        broken because the policy now sees a different obs.
        """
        try:
            # Flip sweep direction: aim for whichever endpoint is farther.
            cur_theta = float(arm_snap.get('theta', 90.0))
            target_theta = (self._sweep_theta_max
                            if cur_theta < 90.0
                            else self._sweep_theta_min)
            # Mid-workspace reach at a known-safe z so the fallback plan is
            # almost always reachable by the BT's replanner.
            target_r = 150.0
            target_z = 60.0
            x, y = polar_to_cartesian(target_theta, target_r)
            ik = solve_ik(x, y, target_z, theta_hint_deg=target_theta)
            if ik is None:
                log.error("RLSweeper stall fallback: IK failed for "
                           "theta=%.0f r=%.0f z=%.0f — cannot escalate to BT",
                           target_theta, target_r, target_z)
                return
            with self._arm_state._lock:
                wrist_rot = self._arm_state.wrist_rot
                gripper   = self._arm_state.gripper
            fallback_q = [
                int(ik.base), int(ik.shoulder), int(ik.elbow),
                int(ik.wrist_vert), int(wrist_rot), int(gripper),
            ]
            log.warning(
                "RLSweeper stall fallback: queueing safety-BT manual "
                "intent to theta=%.0f (joints=%s)",
                target_theta, fallback_q,
            )
            self._safety_api.queue_manual_intent(fallback_q)
        except Exception as exc:
            log.exception("RLSweeper stall fallback failed: %s", exc)

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
            wrist_rot = self._arm_state.wrist_rot
            gripper   = self._arm_state.gripper
            self._arm_state.theta = theta
            self._arm_state.r     = r
            self._arm_state.z     = z
        self._arm_state.update_joints_from_ik(ik, wrist_rot, gripper)

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

            # Observation state — derive polar from the arm's actual joint
            # shadow so the policy can't drift from hardware reality when
            # previous ticks had commands rejected. (Fix E: was reading
            # state.theta/r/z, which are commanded-IK-parameter shadows and
            # stay frozen on any _send_move reject.)
            arm_snap = self._arm_state.snapshot()
            live_joints = list(arm_snap.get('joints', []) or [])
            if len(live_joints) >= 4:
                try:
                    th_live, r_live, z_live = fk_polar(live_joints)
                    arm_snap['theta'] = th_live
                    arm_snap['r']     = r_live
                    arm_snap['z']     = z_live
                except Exception:
                    pass   # keep whatever snapshot() returned

            goal     = self._effective_goal(arm_snap)
            obs      = self._build_obs(goal)

            # Policy inference. Stochastic at deployment so the policy
            # can escape deterministic stall loops: the mean action may
            # be unreachable or zero, but the sampled noise eventually
            # produces a different action and something moves. (Fix A.)
            action_n = self._policy_ref.predict(obs, deterministic=False)
            action_n = np.clip(action_n, -1.0, 1.0).astype(np.float32)
            raw      = BraccioBaseEnv.denormalize_action(action_n)

            # Commanded pose (clamped)
            new_theta = float(np.clip(arm_snap['theta'] + raw[0], 0.0, 180.0))
            new_r     = float(np.clip(arm_snap['r']     + raw[1], R_MIN, R_MAX))
            new_z     = float(np.clip(arm_snap['z']     + raw[2], Z_MIN, Z_MAX))
            new_delta = int(np.clip(round(arm_snap['delta'] + raw[3]),
                                    DELTA_MIN, DELTA_MAX))

            # Stall detection. A tick is a stall if _send_move rejected it
            # OR the commanded target is ~identical to the current pose
            # (policy produced near-zero action — "stopped").
            move_eps = max(
                abs(new_theta - arm_snap['theta']),
                abs(new_r     - arm_snap['r']),
                abs(new_z     - arm_snap['z']),
            )

            # Quiet-ticks window: while the BT is executing a fallback
            # manual intent, SKIP the sweeper's own command dispatch so
            # the two paths don't race on the serial bridge. Without
            # this the RLSweeper's per-tick near-zero-motion SET ALLs
            # overwrite the BT's EMA-smoothed manual-intent moves and
            # the arm never actually executes the fallback. Stall count
            # still advances during the quiet window so abort can fire
            # if the BT itself fails to produce motion.
            if self._quiet_ticks_left > 0:
                self._quiet_ticks_left -= 1
                sent_ok = False         # suppressed on purpose
            else:
                sent_ok = self._send_move(new_theta, new_r, new_z)

            is_stall = (not sent_ok) or (move_eps < _NO_MOTION_EPS)

            if is_stall:
                self._stall_count += 1
                # Log once at first crossing, then every N thereafter.
                if (self._stall_count == _STALL_LOG_FIRST
                        or self._stall_count % _STALL_LOG_EVERY_N == 0):
                    log.warning(
                        "RLSweeper stalled %d consecutive ticks "
                        "(policy action unreachable or zero). "
                        "Fallback=%d  abort=%d",
                        self._stall_count, _STALL_THRESHOLD,
                        _STALL_ABORT_THRESHOLD,
                    )
                    self.stall_warned = True

                # Escalate to the safety-BT cascading replanner. Fire
                # at the initial threshold AND re-fire periodically
                # while the stall persists. A single queued manual
                # intent is sometimes silently dropped by the BT (e.g.,
                # its manual branch refused the plan). Periodic re-fire
                # guarantees we keep trying.
                if self._safety_api is not None:
                    ticks_since_last_fallback = (
                        self._stall_count - self._last_fallback_tick
                    )
                    should_fire = (
                        self._stall_count == _STALL_THRESHOLD
                        or (self._stall_count > _STALL_THRESHOLD
                            and ticks_since_last_fallback >= _FALLBACK_REFIRE_EVERY)
                    )
                    if should_fire:
                        self._dispatch_stall_fallback(arm_snap)
                        self._last_fallback_tick = self._stall_count
                        self._quiet_ticks_left   = _FALLBACK_QUIET_TICKS

                # Auto-abort: the policy is demonstrably broken and
                # even the BT fallback isn't moving the arm. Stop the
                # sweeper so the user gets manual control back without
                # having to press Z through a frozen UI.
                if self._stall_count >= _STALL_ABORT_THRESHOLD:
                    log.error(
                        "RLSweeper AUTO-ABORTING after %d stalled ticks "
                        "(~%.1f s). Policy is not producing motion and "
                        "the safety-BT fallback failed to recover. "
                        "Stopping sweeper so manual control is restored.",
                        self._stall_count,
                        self._stall_count / float(self.tick_hz),
                    )
                    self.stall_aborted = True
                    self._stop.set()
                    break
            else:
                if self._stall_count >= _STALL_LOG_FIRST:
                    log.info("RLSweeper recovered after %d stalled ticks",
                             self._stall_count)
                self._stall_count = 0
                self.stall_warned = False

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
