"""
rl_reward.py — Reward function for the Braccio RL policy.

Operates on observation vectors produced by `rl_recorder._encode_obs` (74
floats, see that module's docstring for the layout).  All reward terms sum
each tick at SWEEP_TICK_HZ.  Designed to be cheap enough to call inline in
the online training loop.

Seven terms (mode-agnostic — valid for sweep, sequence-editor, and hardware):
  1. Goal progress         — reward forward motion toward goal
  2. Waypoint reached      — bonus when goal dist crosses threshold
  3. Collision / IR        — hard penalty on IR DANGER (NOR-gate: 0 or 3 only)
  4. Proximity shaping     — soft quadratic ramp as ToF closes in
  5. Speed cost            — quadratic penalty on action magnitude
  6. Proximity speed limit — extra penalty for fast motion near obstacles
  7. Jerk                  — penalise sudden direction/speed changes

Term 8 (holding penalty) was removed: it punished near-zero action when the
goal was far, but the sequence editor legitimately holds at waypoints during
wait_ms.  Goal progress (term 1) already penalises stalling implicitly.

Action convention
-----------------
action_n is 4-float normalised: [Δtheta_n, Δr_n, Δz_n, Δdelta_n].
  action_n[0] Δtheta — primary collision axis (sweeping into obstacles)
  action_n[1] Δr     — radial reach change
  action_n[2] Δz     — height change
  action_n[3] Δdelta — slew rate (not penalised by speed terms)

Three ways to use this module
-----------------------------
1. In-loop (live training / sweeper):
     r = compute_reward(obs, action_n, next_obs,
                        prev_action_n=prev_a,
                        info={'obstacle_response': ..., 'ir_bits': ...})

2. Offline session annotation (overwrite rewards[] in an NPZ):
     annotate_session_file('logs/rl_transitions_20260419.npz')

3. Composition on top of DTW path-annotator rewards (they are blended
   50/50 by the annotator, this module is additive):
     rewards_total = rewards_dtw + rewards_compute

All reward terms assume the action vector is **normalised** to roughly
[-1, +1] per component.  Raw (theta°, r mm, z mm, delta_int) deltas recorded
by `RLRecorder` must be passed through `normalize_action()` first.

"""

from __future__ import annotations

from typing import Optional, Mapping

import numpy as np

from .constants import (
    THETA_STEP, Z_STEP, DELTA_MAX, DELTA_MIN,
    TOF_THRESHOLDS_MM, SWEEP_COLLISION_RADIUS_MM,
)


# ── Obs vector slice constants (keep in sync with rl_recorder._encode_obs) ──
OBS_ARM_SLICE        = slice(0, 5)
OBS_CH0_GRID_SLICE   = slice(5, 21)    # 16 cells, normalised by / 250mm
OBS_CH1_GRID_SLICE   = slice(21, 37)   # 16 cells, normalised by / 250mm
OBS_CH2_GRID_SLICE   = slice(37, 53)   # 16 cells, normalised by / 50mm (advisory)
OBS_IR_IDX           = 53
OBS_MAP_SLICE        = slice(54, 62)
OBS_Z_MASK_SLICE     = slice(62, 67)
OBS_GOAL_DELTA_SLICE = slice(67, 70)
OBS_FORBIDDEN_SLICE  = slice(70, 74)

# Normalisation used at encode time for un-normalising back to mm
CH0_CH1_NORM_MM      = 250.0
CH2_NORM_MM          =  50.0
CLEAR_SENTINEL       =   1.0   # obs cell value when ToF returned NaN / clear

# ── Reward tunables ─────────────────────────────────────────────────────────
WAYPOINT_TOLERANCE_N = 0.05    # normalised goal distance considered "reached"
GOAL_PROGRESS_GAIN   = 0.10    # ~+0.1 per tick of meaningful forward progress
GOAL_REACHED_BONUS   = 2.00

IR_DANGER_PENALTY    = -5.00
# IR_CLOSE_PENALTY removed: NOR-gate hardware only outputs 0 or 3; ir==2 is unreachable.
BACK_AWAY_PENALTY    = -3.00   # applied on top of IR when obstacle_response=='back_away'

PROXIMITY_GAIN       = 0.50    # per sensor: quadratic ramp up to this at dist→0
SPEED_COST_GAIN      = 0.02    # quadratic on action[0], action[1]
SPEED_LIMIT_ZONE_MM  = 150.0   # below this range proximity throttles speed
SPEED_LIMIT_GAIN     = 1.00

JERK_GAIN_THETA      = 0.02    # |Δaction[0]|  theta
JERK_GAIN_R          = 0.01    # |Δaction[1]|  r
JERK_GAIN_Z          = 0.01    # |Δaction[2]|  z

# Fix F: Replan-progress bonus. Rewards moving SIDEWAYS (theta axis)
# when an obstacle is close. Without this term the speed + proximity
# penalties converge to "stop" as Pareto-optimal — moving earns a
# penalty, stopping earns nothing. This bonus makes continuing to sweep
# around the obstacle strictly better than freezing.
#
# Coefficients retuned after the first retrain plateaued with the
# policy exploiting half-sweep bonuses in free-sweep episodes while
# still freezing on on-path episodes. The recovery bonus is now the
# DOMINANT on-path reward (+5 per successful clear) so SAC can't
# plateau at "stop near obstacles, sweep when clear" — on-path
# success must be worth comparable to free-sweep success.
REPLAN_PROGRESS_COEFF    = 2.00
REPLAN_OBSTACLE_THRESH_N = 0.40   # obs cell < 0.40 (= 100 mm for CH0/CH1) → "close"
REPLAN_RECOVERY_BONUS    = 5.00   # one-shot, fires when obstacle clears

# Fix H: Sweep-cycle completion bonus. Delivered by the sim env each
# time the arm crosses θ=90 from one side to the other without
# colliding. Reduced from 2.5 → 1.0 because at 2.5 the free-sweep case
# dominated the expected-reward surface and SAC ignored on-path
# episodes entirely. The bonus is still meaningful but no longer the
# only way to earn large reward.
HALF_SWEEP_BONUS     = 1.00


# ── Helpers: un-normalise obs fields ────────────────────────────────────────

def extract_min_mm(obs: np.ndarray, slc: slice, norm_mm: float) -> float:
    """
    Return the minimum un-normalised distance (mm) across a flattened ToF grid
    slice of `obs`.  `CLEAR_SENTINEL` (1.0) cells, which originally came from
    NaN / beyond-threshold returns, are ignored.  If every cell is clear we
    return `norm_mm * 2` (the clip ceiling) so downstream penalties don't
    fire on empty observations.
    """
    cells = obs[slc]
    mask  = cells < CLEAR_SENTINEL - 1e-6
    if not np.any(mask):
        return float(norm_mm * 2.0)
    return float(cells[mask].min() * norm_mm)


def extract_ir_bits(obs: np.ndarray) -> int:
    """0..3 severity decoded from the normalised IR channel."""
    return int(round(float(obs[OBS_IR_IDX]) * 3.0))


def extract_goal_distance(obs: np.ndarray) -> float:
    """L2 norm of normalised (goal_dtheta, goal_dr, goal_dz)."""
    return float(np.linalg.norm(obs[OBS_GOAL_DELTA_SLICE]))


# ── Action normalisation ────────────────────────────────────────────────────

_DELTA_SPAN = max(1.0, float(DELTA_MAX - DELTA_MIN))

def normalize_action(raw_action: np.ndarray) -> np.ndarray:
    """
    Convert the raw (Δtheta°, Δr mm, Δz mm, Δdelta int) action recorded by
    `RLRecorder` into the normalised [-1, +1] convention the reward expects.

    One step keypress maps to ±1.0, saturating further.
    """
    from .constants import R_STEP
    raw = np.asarray(raw_action, dtype=np.float32)
    a0 = np.clip(raw[0] / THETA_STEP,  -1.0, 1.0)
    a1 = np.clip(raw[1] / R_STEP,      -1.0, 1.0)
    a2 = np.clip(raw[2] / Z_STEP,      -1.0, 1.0)
    a3 = np.clip(raw[3] / _DELTA_SPAN, -1.0, 1.0)
    return np.array([a0, a1, a2, a3], dtype=np.float32)


# ── Reward computation ─────────────────────────────────────────────────────

def compute_reward(obs:          np.ndarray,
                   action_n:     np.ndarray,
                   next_obs:     np.ndarray,
                   prev_action_n: Optional[np.ndarray] = None,
                   info:          Optional[Mapping] = None) -> float:
    """
    Return the scalar reward for one transition.

    Parameters
    ----------
    obs, next_obs   : 74-float observation vectors (see rl_recorder).
    action_n        : 3-float normalised action (see normalize_action()).
    prev_action_n   : previous tick's normalised action (for jerk term).
                      If None, jerk term is skipped (first tick of episode).
    info            : optional dict supplying categorical / raw data that
                      isn't in the obs vector.  Recognised keys:
                        'obstacle_response' — str, 'back_away' adds penalty
                        'ir_bits'           — int 0..3, overrides obs[53]
                                              (used when training offline from
                                              a session where obs IR is stale)
    """
    if info is None:
        info = {}

    r = 0.0

    # ── 1. Goal progress ──────────────────────────────────────────────────
    dist_before = extract_goal_distance(obs)
    dist_after  = extract_goal_distance(next_obs)
    r += GOAL_PROGRESS_GAIN * (dist_before - dist_after)

    # ── 2. Waypoint reached bonus ─────────────────────────────────────────
    if dist_after < WAYPOINT_TOLERANCE_N and dist_before >= WAYPOINT_TOLERANCE_N:
        r += GOAL_REACHED_BONUS

    # ── 3. Collision severity (IR-based) ──────────────────────────────────
    # Hardware note: NOR-gate wiring means ir_bits is 0 or 3 only.
    # ir==1 and ir==2 are impossible but the check is kept as ir>=3 for safety.
    ir = int(info.get('ir_bits', extract_ir_bits(obs)))
    if ir >= 3:
        r += IR_DANGER_PENALTY
    if info.get('obstacle_response') == 'back_away':
        r += BACK_AWAY_PENALTY

    # ── 4. Proximity shaping (quadratic ramp up as ToF closes in) ────────
    ch0_thr = TOF_THRESHOLDS_MM[0] if len(TOF_THRESHOLDS_MM) > 0 else 250.0
    ch1_thr = TOF_THRESHOLDS_MM[1] if len(TOF_THRESHOLDS_MM) > 1 else 250.0
    ch0_min_mm = extract_min_mm(obs, OBS_CH0_GRID_SLICE, CH0_CH1_NORM_MM)
    ch1_min_mm = extract_min_mm(obs, OBS_CH1_GRID_SLICE, CH0_CH1_NORM_MM)
    for dist, thr in ((ch0_min_mm, ch0_thr), (ch1_min_mm, ch1_thr)):
        if dist < thr:
            ramp = 1.0 - (dist / thr)
            r -= PROXIMITY_GAIN * (ramp * ramp)

    # ── 5. Speed penalty (quadratic on all spatial action dims) ──────────
    # action: [Δtheta_n, Δr_n, Δz_n, Δdelta_n] — delta not penalised here
    a0 = float(action_n[0])   # Δtheta
    a1 = float(action_n[1])   # Δr
    a2 = float(action_n[2])   # Δz
    r -= SPEED_COST_GAIN * (a0 * a0 + a1 * a1 + a2 * a2)

    # ── 6. Proximity-scaled speed limit ──────────────────────────────────
    # Theta and r both move the arm laterally toward obstacles; z is safer.
    min_dist = min(ch0_min_mm, ch1_min_mm)
    if min_dist < SPEED_LIMIT_ZONE_MM:
        allowed_speed = (min_dist / SPEED_LIMIT_ZONE_MM) ** 2   # ∈ [0, 1]
        excess_theta = max(0.0, abs(a0) - allowed_speed)
        excess_r     = max(0.0, abs(a1) - allowed_speed)
        r -= SPEED_LIMIT_GAIN * excess_theta
        r -= SPEED_LIMIT_GAIN * 0.5 * excess_r   # r less collision-critical than theta

    # ── 7. Jerk penalty ──────────────────────────────────────────────────
    if prev_action_n is not None:
        p0 = float(prev_action_n[0])
        p1 = float(prev_action_n[1])
        p2 = float(prev_action_n[2])
        r -= JERK_GAIN_THETA * abs(a0 - p0)
        r -= JERK_GAIN_R     * abs(a1 - p1)
        r -= JERK_GAIN_Z     * abs(a2 - p2)

    # ── 8. Replan-progress bonus (Fix F) ─────────────────────────────────
    # When an obstacle is close to either primary ToF channel in the
    # CURRENT or PREVIOUS observation, reward continuing to move along
    # the sweep axis (theta). This counter-balances the speed +
    # proximity penalties that otherwise make "stop" the Pareto-optimal
    # action near an obstacle. Direction of motion doesn't matter —
    # either way around the obstacle is fine.
    min_thresh_mm = REPLAN_OBSTACLE_THRESH_N * CH0_CH1_NORM_MM
    prev_ch0_mm = extract_min_mm(obs,      OBS_CH0_GRID_SLICE, CH0_CH1_NORM_MM)
    prev_ch1_mm = extract_min_mm(obs,      OBS_CH1_GRID_SLICE, CH0_CH1_NORM_MM)
    next_ch0_mm = extract_min_mm(next_obs, OBS_CH0_GRID_SLICE, CH0_CH1_NORM_MM)
    next_ch1_mm = extract_min_mm(next_obs, OBS_CH1_GRID_SLICE, CH0_CH1_NORM_MM)
    was_close  = (prev_ch0_mm < min_thresh_mm) or (prev_ch1_mm < min_thresh_mm)
    now_close  = (next_ch0_mm < min_thresh_mm) or (next_ch1_mm < min_thresh_mm)
    if was_close or now_close:
        # Dead-zone for noise. Only rewards actual motion, not jitter.
        replan_drive = max(0.0, abs(a0) - 0.05)
        r += REPLAN_PROGRESS_COEFF * replan_drive
        # Bonus for ACTIVELY moving away: obstacle was close, now it
        # isn't. Strict upgrade over "still detecting close obstacle."
        if was_close and not now_close:
            r += REPLAN_RECOVERY_BONUS

    # ── 9. Sweep-cycle bonus (Fix H) ─────────────────────────────────────
    # The sim env sets info['half_sweep_completed']=True on the step
    # where θ crosses 90° from one side to the other (without collision).
    # Each crossing earns HALF_SWEEP_BONUS; two crossings = full cycle,
    # matching the project's top-line objective ("continuous 0↔180
    # obstacle-aware sweep").
    if info.get('half_sweep_completed', False):
        r += HALF_SWEEP_BONUS

    return float(r)


def compute_reward_array(obs_seq:      np.ndarray,
                         actions_raw:  np.ndarray,
                         next_obs_seq: np.ndarray,
                         info_seq:     Optional[list] = None) -> np.ndarray:
    """
    Vectorised-ish driver for offline session annotation.

    Parameters
    ----------
    obs_seq      : (N, 74) float
    actions_raw  : (N, 3)  float  — raw deltas as stored by RLRecorder
    next_obs_seq : (N, 74) float
    info_seq     : optional list of N dicts (see compute_reward.info)

    Returns
    -------
    rewards : (N,) float32
    """
    n = len(obs_seq)
    out = np.zeros(n, dtype=np.float32)
    prev_an = None
    for i in range(n):
        an = normalize_action(actions_raw[i])
        info = info_seq[i] if info_seq is not None else None
        out[i] = compute_reward(obs_seq[i], an, next_obs_seq[i],
                                prev_action_n=prev_an, info=info)
        prev_an = an
    return out


# ── Offline NPZ annotator ──────────────────────────────────────────────────

def annotate_session_file(npz_path: str,
                          mode: str = 'replace',
                          sidecar: bool = True) -> np.ndarray:
    """
    Load a session NPZ, compute per-tick rewards, and write them back.

    Parameters
    ----------
    npz_path : path to an `rl_transitions_*.npz`
    mode     : 'replace' overwrites the rewards[] array.
               'blend'   averages the new rewards with the existing ones
                         (useful on top of DTW-annotated files).
    sidecar  : also write `<name>_reward_annotated.npz` alongside.

    Returns
    -------
    rewards : (N,) float32 — the rewards written to disk
    """
    import os

    z = np.load(npz_path, allow_pickle=False)
    obs      = z['obs']
    actions  = z['actions']
    next_obs = z['next_obs']
    old_rew  = z['rewards']

    new_rew = compute_reward_array(obs, actions, next_obs)

    if mode == 'blend':
        final = (new_rew + old_rew.astype(np.float32)) * 0.5
    elif mode == 'replace':
        final = new_rew
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    arrays = {k: z[k] for k in z.files}
    arrays['rewards'] = final.astype(np.float32)
    np.savez_compressed(npz_path, **arrays)

    if sidecar:
        base, ext = os.path.splitext(npz_path)
        np.savez_compressed(base + '_reward_annotated' + ext, **arrays)

    return final


def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Compute RL rewards for a recorded session NPZ.")
    p.add_argument('npz_path', help="Path to rl_transitions_*.npz")
    p.add_argument('--mode', choices=('replace', 'blend'), default='replace',
                   help="replace existing rewards[] (default) or blend 50/50")
    p.add_argument('--no-sidecar', dest='sidecar', action='store_false',
                   help="don't emit *_reward_annotated.npz")
    args = p.parse_args()
    r = annotate_session_file(args.npz_path,
                              mode=args.mode, sidecar=args.sidecar)
    print(f"[rl_reward] {len(r)} rewards written; "
          f"min={r.min():.3f} mean={r.mean():.3f} max={r.max():.3f}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_cli())

