"""
rl_env.py — Abstract Gymnasium base environment for the Braccio RL policy.

Two concrete subclasses implement this interface:
  BraccioSimEnv      (braccio_ctrl/sim/braccio_env.py)  — PyBullet physics
  BraccioHardwareEnv (rl_sweeper.py, Stage 5)           — real hardware loop

Observation space : Box(-2.0, +2.0, shape=(74,), dtype=float32)
  Layout is identical to rl_recorder._encode_obs (see that module's docstring).
  obs[3] = dir_to_goal = sign(goal_theta - theta) ∈ {-1, 0, +1}.
  Sweep mode is a special case: goals alternate between 0° and 180°.

Action space : Box(-1.0, +1.0, shape=(4,), dtype=float32)
  [0] Δtheta  normalised  (+1.0 == one THETA_STEP° forward)
  [1] Δr      normalised  (+1.0 == one R_STEP mm outward)
  [2] Δz      normalised  (+1.0 == one Z_STEP mm upward)
  [3] Δdelta  normalised  (+1.0 == full slew-rate range toward DELTA_MAX)

  Raw (physical) deltas:  use denormalize_action() to convert back.
  Normalised actions:     pass directly to compute_reward().

Reward function: rl_reward.compute_reward() — 7 terms, same for sim + hardware.
  Mode-agnostic: no sweep-specific terms.  Hold behaviour during sequence-editor
  wait_ms is handled outside the policy by SequenceRLRunner.

Episode termination:
  terminated = IR DANGER (ir_bits >= 3) — collision
  truncated  = step count >= max_episode_steps
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .rl_reward import compute_reward
from .constants import THETA_STEP, R_STEP, Z_STEP, DELTA_MIN, DELTA_MAX

# ── Shared constants ──────────────────────────────────────────────────────────

_OBS_DIM          = 74
_ACT_DIM          = 4
_OBS_LOW          = -2.0
_OBS_HIGH         =  2.0
_DELTA_SPAN       = float(max(1, DELTA_MAX - DELTA_MIN))

MAX_EPISODE_STEPS = 500    # default truncation limit (~50 s at 10 Hz)
IR_STOP_THRESHOLD = 3      # ir_bits >= this → terminated


class BraccioBaseEnv(gym.Env):
    """
    Abstract base class for all Braccio Gymnasium environments.

    Subclasses **must** implement the three abstract methods below.

    Abstract methods
    ----------------
    _reset_state() → None
        Randomise (or deterministically set) the episode starting state.
        Called once per reset() before _get_obs().

    _get_obs() → np.ndarray  shape (74,) float32
        Return the current observation vector.  Must be consistent with
        rl_recorder._encode_obs layout.

    _apply_action(action_n: np.ndarray) → dict
        Apply a normalised 4-float action to the environment for one timestep.
        Return an info dict consumed by compute_reward() and _compute_done():
          'obstacle_response' : str   'clear' | 'replan' | 'back_away'
          'ir_bits'           : int   0 or 3  (NOR-gate hardware)
          'ch0_min_mm'        : float nearest CH0 cell distance (mm)
          'ch1_min_mm'        : float nearest CH1 cell distance (mm)

    Optional overrides
    ------------------
    _compute_done(info, next_obs) → bool
        Default: terminated when ir_bits >= IR_STOP_THRESHOLD.
    render() → None | np.ndarray
        Default: no-op.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    observation_space = spaces.Box(
        low  = _OBS_LOW,
        high = _OBS_HIGH,
        shape = (_OBS_DIM,),
        dtype = np.float32,
    )
    action_space = spaces.Box(
        low  = -1.0,
        high =  1.0,
        shape = (_ACT_DIM,),
        dtype = np.float32,
    )

    def __init__(self, max_episode_steps: int = MAX_EPISODE_STEPS):
        super().__init__()
        self._max_episode_steps   = max_episode_steps
        self._step_count          = 0
        self._prev_action_n: Optional[np.ndarray] = None

    # ── Abstract interface (must override) ────────────────────────────────────

    @abc.abstractmethod
    def _reset_state(self) -> None:
        """Set up a new episode starting state (randomised or fixed)."""

    @abc.abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Return a 74-float observation from the current environment state."""

    @abc.abstractmethod
    def _apply_action(self, action_n: np.ndarray) -> dict:
        """
        Apply normalised action; advance physics / wait one hardware tick.

        Parameters
        ----------
        action_n : (3,) float32 in [-1, +1]

        Returns
        -------
        info : dict with keys 'obstacle_response', 'ir_bits',
               'ch0_min_mm', 'ch1_min_mm'
        """

    # ── Optional overrides ────────────────────────────────────────────────────

    def _compute_done(self, info: dict, next_obs: np.ndarray) -> bool:
        return int(info.get('ir_bits', 0)) >= IR_STOP_THRESHOLD

    def render(self) -> None:
        pass

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count    = 0
        self._prev_action_n = np.zeros(_ACT_DIM, dtype=np.float32)
        self._reset_state()
        obs = self._get_obs()
        return obs.astype(np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_n = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        obs_before = self._get_obs()
        info       = self._apply_action(action_n)
        obs_after  = self._get_obs()

        reward = compute_reward(
            obs          = obs_before,
            action_n     = action_n,
            next_obs     = obs_after,
            prev_action_n = self._prev_action_n,
            info         = info,
        )

        terminated = self._compute_done(info, obs_after)
        self._step_count += 1
        truncated  = (self._step_count >= self._max_episode_steps)

        self._prev_action_n = action_n.copy()
        return obs_after.astype(np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        pass

    # ── Utility helpers (available to all subclasses) ─────────────────────────

    @staticmethod
    def denormalize_action(action_n: np.ndarray) -> np.ndarray:
        """
        Convert a normalised action back to raw physical deltas.

        Returns
        -------
        raw : (4,) float32  [Δtheta°, Δr mm, Δz mm, Δdelta (float)]
        """
        a = np.asarray(action_n, dtype=np.float32)
        return np.array([
            float(a[0]) * THETA_STEP,
            float(a[1]) * R_STEP,
            float(a[2]) * Z_STEP,
            float(a[3]) * _DELTA_SPAN,
        ], dtype=np.float32)

    @staticmethod
    def obs_to_arm_pose(obs: np.ndarray) -> dict:
        """
        Decode arm-pose features from obs[0:5] back to physical units.

        Returns
        -------
        dict with keys 'theta' (deg), 'r' (mm), 'z' (mm), 'dir_to_goal', 'delta'
          dir_to_goal : sign(goal_theta - theta) ∈ {-1, 0, +1}
        """
        return {
            'theta':       float((obs[0] + 1.0) * 90.0),
            'r':           float(obs[1] * 115.0 + 125.0),
            'z':           float(obs[2] * 125.0),
            'dir_to_goal': float(obs[3]),   # {-1, 0, +1}
            'delta':       float((obs[4] + 1.0) * 3.0),
        }
