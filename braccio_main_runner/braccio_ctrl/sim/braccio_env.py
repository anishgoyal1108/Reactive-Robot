"""
braccio_env.py — PyBullet simulation environment for Braccio RL training.

Inherits BraccioBaseEnv (rl_env.py).  Physics are analytical:
- Arm state tracked as (theta, r, z, delta) — no PyBullet FK/IK loop.
- Servo lag modelled as first-order scalar, randomised per episode.
- ToF readings computed analytically via ray-sphere intersection + noise.
- One spherical obstacle per episode, domain-randomised at each reset().
- PyBullet used for URDF visualisation and joint state display only.

Episode termination:
  terminated  — arm tip enters obstacle + tip_radius collision zone
  truncated   — step count >= max_episode_steps

Goal distribution (per episode):
  80% random workspace goals uniformly sampled from reach envelope
  20% sweep-mode goals (alternating theta=0° and 180°)

Usage:
  from braccio_ctrl.sim.braccio_env import BraccioSimEnv
  env = BraccioSimEnv(gui=True)
  obs, _ = env.reset()
  obs, rew, term, trunc, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pybullet as p
import pybullet_data

from ..rl_env import BraccioBaseEnv, IR_STOP_THRESHOLD
from ..ik_solver import solve_ik, reachability
from ..constants import (
    THETA_STEP, R_STEP, Z_STEP, DELTA_MIN, DELTA_MAX,
    R_MIN, R_MAX, Z_MIN, Z_MAX,
    SWEEP_Z_CANDIDATES, SWEEP_Z_DEFAULT_MM as SWEEP_Z_DEFAULT, OBS_MAP_MAX_AGE_S,
    SWEEP_STEP_DEG, SWEEP_COLLISION_RADIUS_MM,
    TOF_THRESHOLDS_MM,
)

# ── File paths ────────────────────────────────────────────────────────────────

_HERE    = Path(__file__).parent
_PKG     = _HERE.parents[1]           # braccio_main_runner/
_URDF    = _PKG / "assets" / "braccio_description" / "urdf" / "braccio_model.urdf"
_NOISE_P = _HERE / "noise_params.json"

# ── Physical constants ────────────────────────────────────────────────────────

SHOULDER_HEIGHT_M = 0.072   # shoulder pivot above base plate (from URDF)
ARM_TIP_RADIUS_MM = 30.0    # collision sphere around gripper tip

# ToF sensor geometry
_TOF_FOV_HALF_DEG = 16.0    # half field-of-view per axis (total 32°×32°)
_TOF_GRID_N       = 4       # 4×4 cells per channel
_TOF_MAX_MM       = 4000.0
_TOF_MIN_MM       = 30.0

# Cell centres: 4 angles spanning ±FOV_HALF (≈ -12°, -4°, +4°, +12°)
_GRID_ANGS = np.radians(np.linspace(
    -_TOF_FOV_HALF_DEG * (1.0 - 1.0 / _TOF_GRID_N),
     _TOF_FOV_HALF_DEG * (1.0 - 1.0 / _TOF_GRID_N),
    _TOF_GRID_N,
))

# ── PyBullet joint indices ────────────────────────────────────────────────────
# URDF: 0=world_joint (fixed), 1=base, 2=shoulder, 3=elbow, 4=wrist_pitch, 5=wrist_roll
_J_BASE    = 1
_J_SHLDR   = 2
_J_ELBOW   = 3
_J_WRIST_P = 4
_J_WRIST_R = 5

# ── Goal distribution ─────────────────────────────────────────────────────────

_SWEEP_PROB = 0.20
_GT_LO, _GT_HI = 20.0,  160.0   # goal theta range (deg)
_GR_LO, _GR_HI = 80.0,  200.0   # goal r range (mm)
_GZ_LO, _GZ_HI = -50.0, 95.0    # goal z range (mm)
_SWEEP_R_MM     = 152.0          # r used for sweep-style goals


def _load_noise_params(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "servo_lag_factor": 0.92, "servo_lag_factor_std": 0.08,
            "tof_noise_sigma_mm": 10.0, "tof_noise_sigma_std": 3.0,
            "cell_dropout_rate": 0.15, "cell_dropout_rate_std": 0.05,
            "domain_randomisation": {
                "servo_lag_factor":  [0.80, 1.20],
                "tof_noise_sigma_mm": [5.0, 15.0],
                "cell_dropout_rate":  [0.05, 0.25],
                "obstacle_radius_mm": [40.0, 120.0],
                "obstacle_theta_deg": [20.0, 160.0],
                "obstacle_r_mm":      [80.0, 200.0],
                "obstacle_z_mm":      [-100.0, 100.0],
            },
        }


def _ray_sphere_hit(
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    sph_c: np.ndarray,
    sph_r: float,
) -> Optional[float]:
    """Return distance to first sphere intersection along ray, or None."""
    oc  = ray_o - sph_c
    b   = 2.0 * float(np.dot(ray_d, oc))
    c   = float(np.dot(oc, oc)) - sph_r * sph_r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t1 = (-b - sq) * 0.5
    if t1 > 0.0:
        return t1
    t2 = (-b + sq) * 0.5
    return t2 if t2 > 0.0 else None


class BraccioSimEnv(BraccioBaseEnv):
    """
    Gymnasium environment that simulates the Braccio arm against a single
    spherical obstacle with domain-randomised noise parameters.
    """

    def __init__(
        self,
        gui: bool = False,
        noise_params_path: Optional[str] = None,
        urdf_path: Optional[str] = None,
        max_episode_steps: int = 500,
        seed: Optional[int] = None,
    ):
        super().__init__(max_episode_steps=max_episode_steps)

        self._rng   = np.random.default_rng(seed)
        self._noise = _load_noise_params(Path(noise_params_path) if noise_params_path else _NOISE_P)

        # Arm state (updated analytically)
        self._theta = 90.0
        self._r     = _SWEEP_R_MM
        self._z     = float(SWEEP_Z_DEFAULT)
        self._delta = float((DELTA_MIN + DELTA_MAX) // 2)

        # Per-episode domain-randomised values (set in _reset_state)
        self._lag       = float(self._noise.get("servo_lag_factor", 0.92))
        self._tof_sigma = float(self._noise.get("tof_noise_sigma_mm", 10.0))
        self._dropout   = float(self._noise.get("cell_dropout_rate", 0.15))

        # Obstacle state (set in _reset_state)
        self._obs_theta  = 90.0
        self._obs_r_mm   = 150.0
        self._obs_z_mm   = 0.0
        self._obs_radius = 80.0

        # Goal state (set in _reset_state)
        self._goal_theta  = 90.0
        self._goal_r      = _SWEEP_R_MM
        self._goal_z      = float(SWEEP_Z_DEFAULT)
        self._sweep_parity = 1   # alternates sweep goal between 0° and 180°

        # ── PyBullet setup ────────────────────────────────────────────────────
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        urdf = str(urdf_path or _URDF)
        self._robot = p.loadURDF(urdf, useFixedBase=True,
                                 physicsClientId=self._client)

        # Obstacle body — recreated with correct radius in _reset_state
        self._obs_body  = None
        self._obs_r_pyb = 0.08   # placeholder; updated in _reset_state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _obs_center_mm(self) -> np.ndarray:
        """Obstacle centre in mm (shoulder-origin coordinate frame)."""
        t = math.radians(self._obs_theta)
        return np.array([
            self._obs_r_mm * math.cos(t),
            self._obs_r_mm * math.sin(t),
            self._obs_z_mm,
        ], dtype=np.float64)

    def _tip_pos_mm(self) -> np.ndarray:
        """Arm tip (end-effector) in mm (shoulder-origin frame)."""
        t = math.radians(self._theta)
        return np.array([
            self._r * math.cos(t),
            self._r * math.sin(t),
            self._z,
        ], dtype=np.float64)

    def _set_arm_joints(self, theta: float, r: float, z: float) -> None:
        """Teleport PyBullet arm to match (theta, r, z) via our IK."""
        t_rad = math.radians(theta)
        ik = solve_ik(r * math.cos(t_rad), r * math.sin(t_rad), z)
        if ik is None:
            return
        for jidx, deg in [
            (_J_BASE,    ik.base),
            (_J_SHLDR,   ik.shoulder),
            (_J_ELBOW,   ik.elbow),
            (_J_WRIST_P, ik.wrist_vert),
        ]:
            p.resetJointState(self._robot, jidx, math.radians(deg),
                              physicsClientId=self._client)

    def _recreate_obstacle(self) -> None:
        """Remove old obstacle body and spawn a new sphere at self._obs_*."""
        if self._obs_body is not None:
            p.removeBody(self._obs_body, physicsClientId=self._client)

        r_m = self._obs_radius * 0.001
        t_rad = math.radians(self._obs_theta)
        ox_m = self._obs_r_mm * 0.001 * math.cos(t_rad)
        oy_m = self._obs_r_mm * 0.001 * math.sin(t_rad)
        oz_m = SHOULDER_HEIGHT_M + self._obs_z_mm * 0.001

        col  = p.createCollisionShape(p.GEOM_SPHERE, radius=r_m,
                                      physicsClientId=self._client)
        vis  = p.createVisualShape(p.GEOM_SPHERE, radius=r_m,
                                   rgbaColor=[1.0, 0.3, 0.0, 0.7],
                                   physicsClientId=self._client)
        self._obs_body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[ox_m, oy_m, oz_m],
            physicsClientId=self._client,
        )

    def _sample_reachable(self) -> tuple[float, float, float]:
        """Sample a random (theta, r, z) that passes reachability check."""
        for _ in range(200):
            theta = float(self._rng.uniform(_GT_LO, _GT_HI))
            r     = float(self._rng.uniform(_GR_LO, _GR_HI))
            z     = float(self._rng.uniform(_GZ_LO, _GZ_HI))
            if reachability(theta, r, z) == 'ok':
                return theta, r, z
        return 90.0, _SWEEP_R_MM, float(SWEEP_Z_DEFAULT)

    # ── ToF raycasting ────────────────────────────────────────────────────────

    def _sensor_frame(self, ch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (fwd, rgt, up) unit vectors for sensor channel ch in world frame.
          ch=0  CH0 — faces radially outward (+r direction)
          ch=1  CH1 — faces radially inward  (-r direction)
          ch=2  CH2 — faces straight up (+z)
        """
        t  = math.radians(self._theta)
        ct, st = math.cos(t), math.sin(t)
        radial_out  = np.array([ ct,  st, 0.0])
        tangential  = np.array([-st,  ct, 0.0])  # 90° CCW from radial
        up_world    = np.array([ 0.0, 0.0, 1.0])

        if ch == 0:
            return radial_out,  tangential, up_world
        elif ch == 1:
            return -radial_out, tangential, up_world
        else:  # ch == 2
            return up_world, tangential, -radial_out

    def _compute_tof_grid(self, ch: int) -> np.ndarray:
        """
        Return a (4, 4) float32 distance array in mm.
        NaN cells = no hit (obstacle not in FOV or beyond range).
        """
        origin = self._tip_pos_mm()
        obs_c  = self._obs_center_mm()
        fwd, rgt, up = self._sensor_frame(ch)

        grid = np.full(16, np.nan, dtype=np.float32)
        idx  = 0
        for az in _GRID_ANGS:         # azimuth: left-right sweep
            cos_az, sin_az = math.cos(az), math.sin(az)
            for el in _GRID_ANGS:     # elevation: up-down sweep
                cos_el, sin_el = math.cos(el), math.sin(el)
                # Unit direction in world frame (unit by construction)
                d = (cos_el * cos_az * fwd
                     + cos_el * sin_az * rgt
                     + sin_el * up)

                hit = _ray_sphere_hit(origin, d, obs_c, self._obs_radius)
                if hit is not None and hit < _TOF_MAX_MM:
                    dist = max(_TOF_MIN_MM, hit)
                    dist += float(self._rng.normal(0.0, self._tof_sigma))
                    dist  = float(np.clip(dist, _TOF_MIN_MM, _TOF_MAX_MM))
                    if self._rng.random() >= self._dropout:
                        grid[idx] = dist
                    # else: cell dropped out → stays NaN
                idx += 1

        return grid.reshape(_TOF_GRID_N, _TOF_GRID_N)

    # ── Collision / IR ────────────────────────────────────────────────────────

    def _compute_ir_bits(self) -> int:
        """Return 0 (clear) or 3 (collision) based on tip-obstacle distance."""
        dist = float(np.linalg.norm(self._tip_pos_mm() - self._obs_center_mm()))
        return 3 if dist < self._obs_radius + ARM_TIP_RADIUS_MM else 0

    # ── Forbidden theta-band features ─────────────────────────────────────────

    def _compute_forbidden_band(self) -> np.ndarray:
        """
        Compute the 4-float forbidden-theta feature block:
          [f_min_n, f_max_n, f_span_n, mem_ahead]
        All zeros when the obstacle is far above/below the arm's z-level.
        """
        obs_c = self._obs_center_mm()
        obs_x, obs_y, obs_z = obs_c

        # Only report a forbidden band if the obstacle is within reach in z
        z_sep = abs(self._z - obs_z)
        if z_sep > self._obs_radius + 60.0:
            return np.zeros(4, dtype=np.float32)

        # Angular half-width of the obstacle as seen from the origin at its r
        r_to_obs = math.sqrt(obs_x ** 2 + obs_y ** 2)
        if r_to_obs < 1e-3:
            return np.zeros(4, dtype=np.float32)

        sin_half = min(1.0, (self._obs_radius + ARM_TIP_RADIUS_MM) / r_to_obs)
        ang_half  = math.degrees(math.asin(sin_half))

        f_lo = self._obs_theta - ang_half
        f_hi = self._obs_theta + ang_half

        f_min_n  = f_lo / 90.0 - 1.0
        f_max_n  = f_hi / 90.0 - 1.0
        f_span_n = (f_hi - f_lo) / 180.0

        # Lookahead: would moving 2 sweep steps ahead cause a collision?
        direction  = math.copysign(1.0, self._goal_theta - self._theta) or 1.0
        ahead_theta = self._theta + direction * SWEEP_STEP_DEG * 2.0
        ahead_theta = max(0.0, min(180.0, ahead_theta))
        ahead_t = math.radians(ahead_theta)
        ahead_pos = np.array([
            self._r * math.cos(ahead_t),
            self._r * math.sin(ahead_t),
            self._z,
        ])
        dist_ahead = float(np.linalg.norm(ahead_pos - obs_c))
        mem_ahead  = float(dist_ahead < self._obs_radius + SWEEP_COLLISION_RADIUS_MM)

        return np.array([f_min_n, f_max_n, f_span_n, mem_ahead], dtype=np.float32)

    # ── BraccioBaseEnv interface ───────────────────────────────────────────────

    def _reset_state(self) -> None:
        dr = self._noise.get("domain_randomisation", {})

        def _u(key: str, default: list) -> float:
            lo, hi = dr.get(key, default)
            return float(self._rng.uniform(lo, hi))

        # Per-episode DR parameters
        self._lag       = _u("servo_lag_factor",  [0.80, 1.20])
        self._tof_sigma = _u("tof_noise_sigma_mm", [5.0, 15.0])
        self._dropout   = _u("cell_dropout_rate",  [0.05, 0.25])
        self._obs_radius = _u("obstacle_radius_mm", [40.0, 120.0])

        # Arm starting state
        self._theta, self._r, self._z = self._sample_reachable()
        self._delta = float(self._rng.integers(DELTA_MIN, DELTA_MAX + 1))

        # Goal
        if self._rng.random() < _SWEEP_PROB:
            self._goal_theta  = 0.0 if self._sweep_parity > 0 else 180.0
            self._goal_r      = _SWEEP_R_MM
            self._goal_z      = float(SWEEP_Z_DEFAULT)
            self._sweep_parity *= -1
        else:
            self._goal_theta, self._goal_r, self._goal_z = self._sample_reachable()

        # Obstacle — ensure it's not on top of start or goal
        for _ in range(20):
            self._obs_theta  = _u("obstacle_theta_deg", [20.0, 160.0])
            self._obs_r_mm   = _u("obstacle_r_mm",      [80.0, 200.0])
            self._obs_z_mm   = _u("obstacle_z_mm",     [-100.0, 100.0])
            obs_c = self._obs_center_mm()
            if (float(np.linalg.norm(self._tip_pos_mm() - obs_c))
                    > self._obs_radius + ARM_TIP_RADIUS_MM + 50.0):
                break

        self._recreate_obstacle()
        self._set_arm_joints(self._theta, self._r, self._z)
        p.stepSimulation(physicsClientId=self._client)

    def _get_obs(self) -> np.ndarray:
        # ── Arm pose (5) ────────────────────────────────────────────────────
        theta_n = self._theta / 90.0 - 1.0
        r_n     = (self._r - 125.0) / 115.0
        z_n     = self._z / 125.0
        delta_n = self._delta / 3.0 - 1.0

        d_theta = self._goal_theta - self._theta
        dir_to_goal = float(np.sign(d_theta)) if abs(d_theta) > 0.5 else 0.0

        # ── ToF grids (48) ──────────────────────────────────────────────────
        def _encode(grid: np.ndarray, norm: float) -> np.ndarray:
            flat = grid.flatten().astype(np.float32)
            return np.where(np.isnan(flat), 1.0,
                            np.clip(flat / norm, 0.0, 2.0))

        ch0 = _encode(self._compute_tof_grid(0), 250.0)
        ch1 = _encode(self._compute_tof_grid(1), 250.0)
        ch2 = _encode(self._compute_tof_grid(2),  50.0)

        # ── IR (1) ───────────────────────────────────────────────────────────
        ir_n = self._compute_ir_bits() / 3.0

        # ── Obstacle map summary (8) ─────────────────────────────────────────
        obs_c = self._obs_center_mm()
        obs_feat = np.array([
            1.0,                          # has_active
            0.0,                          # age (always fresh)
            obs_c[0] / 200.0, obs_c[1] / 200.0, obs_c[2] / 100.0,  # centroid
            obs_c[0] / 200.0, obs_c[1] / 200.0, obs_c[2] / 100.0,  # kalman ≈ centroid
        ], dtype=np.float32)

        # ── Z reachability mask (5) ──────────────────────────────────────────
        z_mask = np.array([
            float(reachability(self._theta, self._r, float(z_c)) == 'ok')
            for z_c in SWEEP_Z_CANDIDATES
        ], dtype=np.float32)

        # ── Goal delta (3) ───────────────────────────────────────────────────
        goal_delt = np.array([
            (self._goal_theta - self._theta) / 90.0,
            (self._goal_r     - self._r)     / 115.0,
            (self._goal_z     - self._z)     / 125.0,
        ], dtype=np.float32)

        # ── Forbidden theta band (4) ─────────────────────────────────────────
        forbidden = self._compute_forbidden_band()

        return np.concatenate([
            [theta_n, r_n, z_n, dir_to_goal, delta_n],  # 5
            ch0, ch1, ch2,                               # 48
            [ir_n],                                      # 1
            obs_feat,                                    # 8
            z_mask,                                      # 5
            goal_delt,                                   # 3
            forbidden,                                   # 4
        ]).astype(np.float32)                            # total 74

    def _apply_action(self, action_n: np.ndarray) -> dict:
        raw = BraccioBaseEnv.denormalize_action(action_n)
        # [Δtheta°, Δr_mm, Δz_mm, Δdelta]

        # Servo lag on spatial axes; slew rate responds instantly
        d_theta = raw[0] * self._lag
        d_r     = raw[1] * self._lag
        d_z     = raw[2] * self._lag
        d_delta = raw[3]

        new_theta = float(np.clip(self._theta + d_theta, 0.0, 180.0))
        new_r     = float(np.clip(self._r + d_r, R_MIN, R_MAX))
        new_z     = float(np.clip(self._z + d_z, Z_MIN, Z_MAX))
        new_delta = float(np.clip(self._delta + d_delta, DELTA_MIN, DELTA_MAX))

        # Only apply if new pose is kinematically reachable
        if reachability(new_theta, new_r, new_z) == 'ok':
            self._theta = new_theta
            self._r     = new_r
            self._z     = new_z
        self._delta = new_delta

        self._set_arm_joints(self._theta, self._r, self._z)
        p.stepSimulation(physicsClientId=self._client)

        # ToF min-distance for reward / info
        ch0_grid = self._compute_tof_grid(0)
        ch1_grid = self._compute_tof_grid(1)

        ch0_vals = ch0_grid[~np.isnan(ch0_grid)]
        ch1_vals = ch1_grid[~np.isnan(ch1_grid)]
        ch0_min  = float(ch0_vals.min()) if ch0_vals.size > 0 else _TOF_MAX_MM
        ch1_min  = float(ch1_vals.min()) if ch1_vals.size > 0 else _TOF_MAX_MM

        ir_bits = self._compute_ir_bits()

        thresh = TOF_THRESHOLDS_MM[0]
        if ir_bits >= IR_STOP_THRESHOLD:
            obs_response = "back_away"
        elif min(ch0_min, ch1_min) < thresh:
            obs_response = "replan"
        else:
            obs_response = "clear"

        return {
            "obstacle_response": obs_response,
            "ir_bits":           ir_bits,
            "ch0_min_mm":        ch0_min,
            "ch1_min_mm":        ch1_min,
        }

    def render(self) -> None:
        pass   # GUI mode renders automatically; DIRECT mode has no output

    def close(self) -> None:
        if p.isConnected(self._client):
            p.disconnect(self._client)
