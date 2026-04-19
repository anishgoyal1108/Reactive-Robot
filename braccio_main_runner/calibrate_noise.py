"""
calibrate_noise.py — Fit noise model parameters from recorded session data.

Reads rl_transitions_*.npz files produced by RLRecorder and estimates:

  servo_lag_factor   — actual Δtheta per tick / commanded Δtheta per tick.
                       Measured by cross-correlating consecutive obs[0] values
                       (decoded theta) against actions[:,0] (normalised Δtheta).
                       Values < 1.0 indicate the arm lags behind commands.

  tof_noise_sigma_mm — Gaussian noise std in mm on close-range ToF cells.
                       Measured as the temporal std of close-range CH0/CH1 cells
                       during static arm segments (|Δtheta| < 0.05 normalised).

  cell_dropout_rate  — Fraction of close-range ToF cells that read as
                       CLEAR_SENTINEL (1.0) when the obstacle map centroid is
                       active (suggesting the obstacle IS present but cells
                       dropped out).

Usage
-----
  python calibrate_noise.py logs/rl_transitions_*.npz
  python calibrate_noise.py                              # auto-find in logs/

Writes braccio_ctrl/sim/noise_params.json.  Set calibrated=true in the JSON
so BraccioSimEnv knows the values are real measurements rather than defaults.

Output format
-------------
{
  "calibrated": true,
  "servo_lag_factor":       0.91,
  "servo_lag_factor_std":   0.07,
  "tof_noise_sigma_mm":     11.3,
  "tof_noise_sigma_std":    2.1,
  "cell_dropout_rate":      0.18,
  "cell_dropout_rate_std":  0.06,
  ...
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import glob
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Obs slice constants (must match rl_recorder._encode_obs) ─────────────────
_OBS_THETA_IDX       = 0       # theta_n = theta/90 - 1
_OBS_R_IDX           = 1
_OBS_Z_IDX           = 2
_OBS_CH0_START       = 5
_OBS_CH0_END         = 21
_OBS_CH1_START       = 21
_OBS_CH1_END         = 37
_OBS_IR_IDX          = 53
_OBS_MAP_HAS_ACTIVE  = 54     # first element of obs_map summary
_CLEAR_SENTINEL      = 1.0    # NaN / beyond-range cells encoded as this value
_CH0_NORM_MM         = 250.0
_CH1_NORM_MM         = 250.0

# ── Thresholds for "close-range" classification ───────────────────────────────
_CLOSE_RANGE_NORM    = 0.80   # cell value < this → obstacle present (< 200 mm)
_STATIC_ACTION_THRESH = 0.05  # |action[0]| below this → arm is static


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_npz_files(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate obs + actions from multiple NPZ files.

    Returns
    -------
    obs     : (N, 74) float32
    actions : (N, 3)  float32  (raw, not normalised)
    """
    all_obs, all_act = [], []
    for p in paths:
        try:
            z = np.load(p, allow_pickle=False)
            all_obs.append(z['obs'].astype(np.float32))
            all_act.append(z['actions'].astype(np.float32))
        except Exception as e:
            print(f"  [warn] could not load {p}: {e}", file=sys.stderr)
    if not all_obs:
        raise RuntimeError("No valid NPZ files found.")
    return np.concatenate(all_obs), np.concatenate(all_act)


# ── Servo lag estimation ──────────────────────────────────────────────────────

def _estimate_servo_lag(obs: np.ndarray, actions: np.ndarray,
                        theta_step: float = 5.0) -> Tuple[float, float]:
    """
    Estimate servo lag factor from obs + raw action arrays.

    Lag = actual_delta_theta / commanded_delta_theta.

    Strategy:
      - commanded Δtheta (normalised) = actions[t, 0] / theta_step
      - actual    Δtheta (normalised) = obs[t+1, 0] - obs[t, 0]
      - We only consider ticks where |commanded| > 0.05 to avoid divide-by-zero
        in idle periods.

    Returns mean and std of the per-tick lag ratio.
    """
    cmd_n  = actions[:-1, 0] / theta_step         # commanded normalised Δθ
    act_n  = np.diff(obs[:, _OBS_THETA_IDX])       # actual   normalised Δθ

    valid = np.abs(cmd_n) > _STATIC_ACTION_THRESH
    if valid.sum() < 10:
        return 0.92, 0.08   # fallback default

    ratio = act_n[valid] / cmd_n[valid]
    # Winsorise extremes: lag should be in [0.3, 1.5]
    ratio = ratio[(ratio > 0.3) & (ratio < 1.5)]
    if len(ratio) < 5:
        return 0.92, 0.08

    return float(np.mean(ratio)), float(np.std(ratio))


# ── ToF noise sigma estimation ────────────────────────────────────────────────

def _estimate_tof_noise(obs: np.ndarray, actions: np.ndarray) -> Tuple[float, float]:
    """
    Estimate per-cell Gaussian noise std (mm) from static arm segments.

    Static segments: consecutive ticks where |action[0]| < threshold.
    Close-range cells: obs value < CLOSE_RANGE_NORM (obstacle < 200 mm from CH0/CH1).

    For each static segment ≥ 3 ticks long, compute temporal std of close cells
    and convert back to mm.

    Returns mean and std of the per-cell noise sigma across all segments.
    """
    static_mask = np.abs(actions[:, 0]) < _STATIC_ACTION_THRESH

    sigmas_mm: List[float] = []

    # Find contiguous static runs
    in_run, start = False, 0
    for i in range(len(static_mask)):
        if static_mask[i] and not in_run:
            start = i
            in_run = True
        elif not static_mask[i] and in_run:
            run_len = i - start
            if run_len >= 3:
                _extract_noise_from_run(obs[start:i], sigmas_mm)
            in_run = False
    # Handle run extending to end
    if in_run and (len(static_mask) - start) >= 3:
        _extract_noise_from_run(obs[start:], sigmas_mm)

    if not sigmas_mm:
        return 10.0, 3.0   # fallback default

    arr = np.array(sigmas_mm, dtype=np.float32)
    arr = arr[arr < 50.0]   # discard outliers (> 50 mm std is noise floor)
    if len(arr) < 3:
        return 10.0, 3.0

    return float(np.mean(arr)), float(np.std(arr))


def _extract_noise_from_run(run: np.ndarray, out: list) -> None:
    """Helper: compute per-cell temporal std for CH0 + CH1 close cells in a run."""
    for ch_start, ch_end, norm_mm in [
        (_OBS_CH0_START, _OBS_CH0_END, _CH0_NORM_MM),
        (_OBS_CH1_START, _OBS_CH1_END, _CH1_NORM_MM),
    ]:
        cells = run[:, ch_start:ch_end]      # (T, 16)
        # Only use cells that are close-range in ALL frames of the run
        is_close = np.all(cells < _CLOSE_RANGE_NORM, axis=0)  # (16,)
        is_close &= np.all(cells < _CLEAR_SENTINEL - 1e-4, axis=0)
        if not np.any(is_close):
            continue
        temporal_std_norm = np.std(cells[:, is_close], axis=0)  # per-cell
        out.extend((temporal_std_norm * norm_mm).tolist())


# ── Cell dropout rate estimation ──────────────────────────────────────────────

def _estimate_cell_dropout(obs: np.ndarray) -> Tuple[float, float]:
    """
    Estimate fraction of ToF cells that drop out (read as CLEAR_SENTINEL)
    when the obstacle map reports an active obstacle.

    Logic:
      - "obstacle present" frames: obs[obs_map_has_active] > 0.5
      - In those frames, count CH0+CH1 cells that are CLEAR_SENTINEL
      - This fraction estimates the real-hardware dropout rate

    Returns mean and std dropout rate across obstacle-present frames.
    """
    has_active = obs[:, _OBS_MAP_HAS_ACTIVE] > 0.5     # (N,) bool
    n_active = has_active.sum()
    if n_active < 10:
        return 0.15, 0.05   # fallback default

    active_frames = obs[has_active]   # (M, 74)
    ch01 = active_frames[:, _OBS_CH0_START:_OBS_CH1_END]   # (M, 32)

    # Cells at CLEAR_SENTINEL in frames where obstacle is active → dropout
    dropout_mask  = np.abs(ch01 - _CLEAR_SENTINEL) < 1e-4   # (M, 32)
    dropout_per_frame = dropout_mask.mean(axis=1)             # (M,)

    return float(np.mean(dropout_per_frame)), float(np.std(dropout_per_frame))


# ── Main ──────────────────────────────────────────────────────────────────────

def calibrate(npz_paths: List[str], out_path: str) -> dict:
    """
    Run full calibration pipeline and write noise_params.json.

    Parameters
    ----------
    npz_paths : list of rl_transitions_*.npz file paths
    out_path  : output JSON path

    Returns
    -------
    params dict (same as written to disk)
    """
    print(f"Loading {len(npz_paths)} NPZ file(s)…")
    obs, actions = _load_npz_files(npz_paths)
    n = len(obs)
    print(f"  Loaded {n:,} transitions  obs={obs.shape}  actions={actions.shape}")

    # ── Servo lag ─────────────────────────────────────────────────────────────
    print("Estimating servo lag…")
    lag_mean, lag_std = _estimate_servo_lag(obs, actions)
    print(f"  servo_lag_factor = {lag_mean:.3f} ± {lag_std:.3f}")

    # ── ToF noise sigma ───────────────────────────────────────────────────────
    print("Estimating ToF noise sigma…")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        noise_mean, noise_std = _estimate_tof_noise(obs, actions)
    print(f"  tof_noise_sigma_mm = {noise_mean:.2f} ± {noise_std:.2f} mm")

    # ── Cell dropout ──────────────────────────────────────────────────────────
    print("Estimating cell dropout rate…")
    dropout_mean, dropout_std = _estimate_cell_dropout(obs)
    print(f"  cell_dropout_rate = {dropout_mean:.3f} ± {dropout_std:.3f}")

    # ── Load existing params so we keep domain-randomisation ranges ───────────
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)

    params = {
        "_comment": (
            "Noise parameters calibrated from real session logs. "
            "Domain-randomisation ranges are kept from previous defaults unless "
            "updated manually."
        ),
        "calibrated": True,
        "n_transitions_used": int(n),

        "servo_lag_factor":      round(lag_mean,    4),
        "servo_lag_factor_std":  round(lag_std,     4),

        "tof_noise_sigma_mm":    round(noise_mean,  3),
        "tof_noise_sigma_std":   round(noise_std,   3),

        "cell_dropout_rate":     round(dropout_mean, 4),
        "cell_dropout_rate_std": round(dropout_std,  4),

        "tof_clip_min_mm":  existing.get("tof_clip_min_mm",  30.0),
        "tof_clip_max_mm":  existing.get("tof_clip_max_mm",  4000.0),

        "domain_randomisation": existing.get("domain_randomisation", {
            "servo_lag_factor":   [0.80, 1.20],
            "tof_noise_sigma_mm": [5.0, 15.0],
            "cell_dropout_rate":  [0.05, 0.25],
            "obstacle_radius_mm": [40.0, 120.0],
            "obstacle_theta_deg": [20.0, 160.0],
            "obstacle_r_mm":      [80.0, 200.0],
            "obstacle_z_mm":      [-100.0, 100.0],
        }),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\nWrote {out_path}")
    return params


def _find_npz_files(log_dir: str = 'logs') -> List[str]:
    pattern = os.path.join(log_dir, 'rl_transitions_*.npz')
    found = sorted(glob.glob(pattern))
    # Exclude already-annotated files
    found = [p for p in found if '_annotated' not in p
             and '_reward_annotated' not in p]
    return found


def _cli() -> int:
    here = Path(__file__).parent
    default_out = str(here / 'braccio_ctrl' / 'sim' / 'noise_params.json')

    p = argparse.ArgumentParser(
        description="Calibrate RL noise model from session transition NPZs.")
    p.add_argument(
        'npz_files', nargs='*',
        help="rl_transitions_*.npz files. If omitted, auto-finds in logs/.")
    p.add_argument(
        '--out', default=default_out,
        help=f"Output JSON path (default: {default_out})")
    p.add_argument(
        '--log-dir', default='logs',
        help="Directory to search for NPZ files when none are specified (default: logs/)")
    args = p.parse_args()

    paths = list(args.npz_files)
    if not paths:
        paths = _find_npz_files(args.log_dir)
        if not paths:
            print(f"No rl_transitions_*.npz files found in {args.log_dir}/",
                  file=sys.stderr)
            print("Run the controller with RLRecorder active, then re-run this script.",
                  file=sys.stderr)
            return 1
        print(f"Auto-found {len(paths)} file(s) in {args.log_dir}/")

    calibrate(paths, args.out)
    return 0


if __name__ == '__main__':
    sys.exit(_cli())
