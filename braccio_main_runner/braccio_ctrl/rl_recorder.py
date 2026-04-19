"""
rl_recorder.py — Background recorder for RL training data collection.

Records (obs, action, reward=0, next_obs, done, theta, r, z, timestamps)
tuples at SWEEP_TICK_HZ in a daemon thread.  Data is flushed to a compressed
NPZ file at shutdown (or when the 50 MB cap is hit, triggering log rotation).

Observation vector layout (74 floats, all float32):
  [0:5]   arm pose — theta/90-1, (r-125)/115, z/125, dir_to_goal, delta/3-1
            dir_to_goal = sign(goal_theta - theta): {-1, 0, +1}
            Mode-agnostic: sweep is just goals=[0°,180°] repeated.
  [5:21]  CH0 ToF grid flattened (4×4=16), NaN→1.0, clipped to [0,2]×(1/250mm)
  [21:37] CH1 ToF grid flattened, same normalisation
  [37:53] CH2 ToF grid flattened, clipped to [0,2]×(1/50mm)
  [53]    IR severity (ir_bits/3)
  [54:62] obstacle map summary (8 floats)
  [62:67] Z-level reachability mask — reachable() AND is_position_clear()
          for each SWEEP_Z_CANDIDATES[i] at the current (theta, r)
  [67:70] goal delta (goal_theta - theta)/90, (goal_r - r)/115, (goal_z - z)/125
  [70:74] forbidden theta-band summary: f_min/90-1, f_max/90-1, f_span/180,
          mem_ahead (lookahead voxel-memory occupancy flag)

NOTE: obs[3] was previously the sweep `direction` flag ({-1,+1}), which was
sweep-mode-specific and meaningless in sequence-editor mode.  It is now
`dir_to_goal` = np.sign(goal_theta - theta), which is universally valid for
any goal (sweep goals and named-state waypoints alike).  OBS_DIM stays 74.

Action vector (3 floats):
  [0] Δtheta  (degrees, raw)
  [1] Δz      (mm, raw)
  [2] Δdelta  (slew-rate integer change)

Human feedback: call add_human_feedback(+/-0.5) from the UI thread to adjust
the rewards of the most recent N_FEEDBACK_STEPS transitions.
"""

import os
import threading
import time
import logging
from typing import Optional

import numpy as np

from .constants import (
    SWEEP_TICK_HZ, OBS_MAP_MAX_AGE_S, LOG_DIR,
    SWEEP_Z_CANDIDATES, SWEEP_COLLISION_RADIUS_MM, SWEEP_STEP_DEG,
)
from .ik_solver import reachability, polar_to_cartesian

log = logging.getLogger(__name__)

# ── Tuning ────────────────────────────────────────────────────────────────────
RECORD_HZ          = SWEEP_TICK_HZ       # matches BT_TICK_HZ (20 Hz)
MAX_FILE_BYTES     = 50 * 1024 * 1024    # 50 MB per file
N_FEEDBACK_STEPS   = 20                  # last N transitions adjusted on +/- key
OBS_DIM            = 74                  # obs vector length (see module docstring)


def _encode_obs(arm_snap: dict, tof_snap: dict, obs_snap: dict,
                direction: float = 1.0,
                obstacle_map=None,
                sweeper=None,
                goal_state: Optional[dict] = None) -> np.ndarray:
    """
    Build a 74-float observation vector from state snapshots.

    Optional inputs allow filling the last three feature groups:
      obstacle_map : ObstacleMap — used for get_obstacle_thetas(),
                     memory_occupied_thetas(), memory_occupied_near()
      sweeper      : AutoSweeper — used for _is_position_clear() in
                     Z-level mask.
      goal_state   : dict with 'theta', 'r', 'z' (absolute target pose).
                     When None, goal delta and dir_to_goal are zeros,
                     meaning "no active goal / at goal already".

    The `direction` arg is kept for the lookahead voxel check only;
    it is no longer placed in the obs vector.  obs[3] is now dir_to_goal.
    """
    theta = arm_snap.get('theta', 90.0)
    r     = arm_snap.get('r',     152.0)
    z     = arm_snap.get('z',     -50.0)
    delta = arm_snap.get('delta', 1)

    theta_n  = theta / 90.0 - 1.0
    r_n      = (r - 125.0) / 115.0
    z_n      = z / 125.0
    delta_n  = delta / 3.0 - 1.0

    # dir_to_goal replaces the old sweep `direction` flag.
    # Computed here so we can use goal_theta before the goal_delt block.
    if goal_state is not None:
        gt = float(goal_state.get('theta', theta))
        dir_to_goal = float(np.sign(gt - theta))   # {-1, 0, +1}
    else:
        dir_to_goal = 0.0   # no active goal → no directional preference

    grids = tof_snap.get('grids', [None, None, None, None])

    def _encode_grid(g, norm: float) -> np.ndarray:
        target = 16
        if g is None or (hasattr(g, 'size') and g.size == 0):
            return np.ones(target, dtype=np.float32)
        flat = np.asarray(g, dtype=np.float32).flatten()
        if len(flat) >= target:
            flat = flat[:target]
        else:
            flat = np.pad(flat, (0, target - len(flat)), constant_values=np.nan)
        return np.where(np.isnan(flat), 1.0,
                        np.clip(flat / norm, 0.0, 2.0)).astype(np.float32)

    ch0 = _encode_grid(grids[0] if len(grids) > 0 else None, 250.0)
    ch1 = _encode_grid(grids[1] if len(grids) > 1 else None, 250.0)
    ch2 = _encode_grid(grids[2] if len(grids) > 2 else None,  50.0)

    ir_n = tof_snap.get('ir_bits', 0) / 3.0

    centroid = obs_snap.get('centroid') or [0.0, 0.0, 0.0]
    kalman   = obs_snap.get('kalman_estimate') or [0.0, 0.0, 0.0]
    age_s    = obs_snap.get('last_obs_age_s', OBS_MAP_MAX_AGE_S)
    obs_feat = np.array([
        float(obs_snap.get('has_active', False)),
        min(age_s / max(OBS_MAP_MAX_AGE_S, 1e-6), 1.0),
        centroid[0] / 200.0, centroid[1] / 200.0, centroid[2] / 100.0,
        kalman[0]   / 200.0, kalman[1]   / 200.0, kalman[2]   / 100.0,
    ], dtype=np.float32)

    # ── Z-level reachability mask (5) ──────────────────────────────────
    # Which SWEEP_Z_CANDIDATES are both kinematically reachable AND clear
    # of known obstacles at the current theta?
    n_z = len(SWEEP_Z_CANDIDATES)
    z_mask = np.zeros(n_z, dtype=np.float32)
    for i, z_cand in enumerate(SWEEP_Z_CANDIDATES):
        reach_ok = False
        try:
            reach_ok = (reachability(theta, r, float(z_cand)) == 'ok')
        except Exception:
            reach_ok = False
        clear_ok = True
        if sweeper is not None:
            try:
                clear_ok = bool(sweeper._is_position_clear(theta, float(z_cand)))
            except Exception:
                clear_ok = True
        z_mask[i] = float(reach_ok and clear_ok)

    # ── Goal delta (3) ─────────────────────────────────────────────────
    if goal_state is not None:
        gt = float(goal_state.get('theta', theta))   # already used above for dir_to_goal
        gr = float(goal_state.get('r',     r))
        gz = float(goal_state.get('z',     z))
        goal_delt = np.array([
            (gt - theta) / 90.0,
            (gr - r)     / 115.0,
            (gz - z)     / 125.0,
        ], dtype=np.float32)
    else:
        goal_delt = np.zeros(3, dtype=np.float32)

    # ── Forbidden theta band summary (4) ───────────────────────────────
    f_min = f_max = f_span = 0.0
    mem_ahead = 0.0
    if obstacle_map is not None:
        try:
            forbidden_list = list(obstacle_map.get_obstacle_thetas(r, z) or [])
        except Exception:
            forbidden_list = []
        try:
            forbidden_list += list(obstacle_map.memory_occupied_thetas(r, z) or [])
        except Exception:
            pass
        if forbidden_list:
            lo = float(min(forbidden_list))
            hi = float(max(forbidden_list))
            f_min  = (lo / 90.0) - 1.0
            f_max  = (hi / 90.0) - 1.0
            f_span = (hi - lo) / 180.0
        # lookahead: is the voxel memory occupied ~2 sweep steps ahead?
        try:
            look_theta = theta + direction * SWEEP_STEP_DEG * 2.0
            lx, ly = polar_to_cartesian(look_theta, r)
            mem_ahead = float(obstacle_map.memory_occupied_near(
                (lx, ly, z), SWEEP_COLLISION_RADIUS_MM))
        except Exception:
            mem_ahead = 0.0
    forbidden = np.array([f_min, f_max, f_span, mem_ahead], dtype=np.float32)

    return np.concatenate([
        [theta_n, r_n, z_n, dir_to_goal, delta_n],
        ch0, ch1, ch2,
        [ir_n],
        obs_feat,
        z_mask,
        goal_delt,
        forbidden,
    ]).astype(np.float32)  # 74 total


class RLRecorder:
    """
    Daemon-thread recorder that saves RL transition data at RECORD_HZ.

    Usage
    -----
    rec = RLRecorder(arm_state, tof_state, imu_state, obstacle_map)
    rec.start()
    # ... controller runs ...
    rec.stop()   # flushes NPZ to disk
    """

    def __init__(self, arm_state, tof_state, imu_state, obstacle_map,
                 sweeper=None):
        self._arm      = arm_state
        self._tof      = tof_state
        self._imu      = imu_state
        self._obs_map  = obstacle_map
        self._sweeper  = sweeper          # optional — enables Z-mask + direction

        self._stop     = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock     = threading.Lock()     # protects _reward_buffer
        self._goal_state: Optional[dict] = None   # optional — set via set_goal()

        # ── Ring buffers (grown dynamically, rotated at MAX_FILE_BYTES) ───
        self._obs_buf:   list = []
        self._act_buf:   list = []
        self._rew_buf:   list = []
        self._nobs_buf:  list = []
        self._done_buf:  list = []
        self._theta_buf: list = []
        self._r_buf:     list = []
        self._z_buf:     list = []
        self._ts_buf:    list = []

        self._prev_obs:   Optional[np.ndarray] = None
        self._prev_theta: Optional[float]      = None
        self._prev_z:     Optional[float]      = None
        self._prev_delta: Optional[int]        = None

        os.makedirs(LOG_DIR, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._record_loop,
            daemon=True,
            name='rl-recorder',
        )
        self._thread.start()
        log.info("RLRecorder started at %.1f Hz", RECORD_HZ)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._flush()

    def add_human_feedback(self, delta: float) -> None:
        """Adjust rewards of last N_FEEDBACK_STEPS transitions by delta."""
        with self._lock:
            n = min(N_FEEDBACK_STEPS, len(self._rew_buf))
            for i in range(1, n + 1):
                self._rew_buf[-i] += delta

    def set_goal(self, goal: Optional[dict]) -> None:
        """
        Update the active goal pose used to fill the goal-delta feature.
        Pass None to clear (obs goal delta falls back to zeros, which the
        network reads as "no active goal" — suitable for free sweep).
        """
        self._goal_state = goal

    def set_sweeper(self, sweeper) -> None:
        """Attach an AutoSweeper / RLSweeper for Z-mask + direction features."""
        self._sweeper = sweeper

    # ── Recording loop ────────────────────────────────────────────────────

    def _record_loop(self) -> None:
        tick = 1.0 / RECORD_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
            except Exception:
                log.exception("RLRecorder tick error")
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, tick - elapsed))

    def _tick(self) -> None:
        now = time.time()
        arm_snap = self._arm.snapshot()
        tof_snap = self._tof.snapshot()
        obs_snap = self._obs_map.snapshot()

        theta = arm_snap['theta']
        r     = arm_snap['r']
        z     = arm_snap['z']
        delta = arm_snap['delta']

        # direction is used only for the voxel-memory lookahead in _encode_obs;
        # it no longer appears in the obs vector (obs[3] is now dir_to_goal).
        direction = 1.0
        if self._sweeper is not None:
            try:
                direction = float(self._sweeper.get_status().get('direction', 1))
            except Exception:
                direction = 1.0

        obs = _encode_obs(
            arm_snap, tof_snap, obs_snap,
            direction=direction,
            obstacle_map=self._obs_map,
            sweeper=self._sweeper,
            goal_state=self._goal_state,
        )

        if self._prev_obs is not None:
            # Compute action: raw deltas (normalized by caller when used in training)
            action = np.array([
                theta - (self._prev_theta or theta),
                z     - (self._prev_z     or z),
                float(delta - (self._prev_delta or delta)),
            ], dtype=np.float32)

            with self._lock:
                self._obs_buf.append(self._prev_obs)
                self._act_buf.append(action)
                self._rew_buf.append(0.0)
                self._nobs_buf.append(obs)
                self._done_buf.append(False)
                self._theta_buf.append(theta)
                self._r_buf.append(r)
                self._z_buf.append(z)
                self._ts_buf.append(now)

            # Rotate file if over size limit
            if self._approx_bytes() > MAX_FILE_BYTES:
                self._flush()
                self._clear_buffers()

        self._prev_obs   = obs
        self._prev_theta = theta
        self._prev_z     = z
        self._prev_delta = delta

    def _approx_bytes(self) -> int:
        n = len(self._obs_buf)
        return n * (OBS_DIM * 2 + 3 + 1 + 3 + 1 + 8) * 4  # rough estimate

    # ── Flush to disk ─────────────────────────────────────────────────────

    def _flush(self) -> None:
        with self._lock:
            n = len(self._obs_buf)
            if n == 0:
                return
            obs      = np.array(self._obs_buf,  dtype=np.float32)
            actions  = np.array(self._act_buf,  dtype=np.float32)
            rewards  = np.array(self._rew_buf,  dtype=np.float32)
            next_obs = np.array(self._nobs_buf, dtype=np.float32)
            dones    = np.array(self._done_buf, dtype=bool)
            theta    = np.array(self._theta_buf, dtype=np.float32)
            r_arr    = np.array(self._r_buf,     dtype=np.float32)
            z_arr    = np.array(self._z_buf,     dtype=np.float32)
            ts       = np.array(self._ts_buf,    dtype=np.float64)

        stamp = time.strftime('%Y%m%d_%H%M%S')
        path  = os.path.join(LOG_DIR, f'rl_transitions_{stamp}.npz')
        np.savez_compressed(
            path,
            obs=obs, actions=actions, rewards=rewards,
            next_obs=next_obs, dones=dones,
            theta=theta, r=r_arr, z=z_arr, timestamps=ts,
        )
        log.info("RLRecorder saved %d transitions → %s", n, path)

    def _clear_buffers(self) -> None:
        with self._lock:
            self._obs_buf.clear();  self._act_buf.clear()
            self._rew_buf.clear();  self._nobs_buf.clear()
            self._done_buf.clear(); self._theta_buf.clear()
            self._r_buf.clear();    self._z_buf.clear()
            self._ts_buf.clear()
        self._prev_obs   = None
        self._prev_theta = None
        self._prev_z     = None
        self._prev_delta = None
