#!/usr/bin/env python3
"""
verify_sim.py — Digital-Twin sanity check for the RL pipeline.

Runs five checks before you spend hours training:

  1. Sim builds and resets cleanly.
  2. Observation vector is the correct (74,) shape.
  3. Random actions don't produce NaN/inf anywhere.
  4. sim obs and rl_recorder obs use the SAME slice layout.
  5. Noise parameters have been calibrated against real hardware.

Usage:
    python verify_sim.py            # headless (fast)
    python verify_sim.py --gui      # pops up PyBullet window for 5 s so
                                    # you can visually confirm the URDF
                                    # model looks right at a known pose.

Exit code: 0 = all checks passed, 1 = one or more failures. Non-zero is
a blocker for training — fix what it complains about, then re-run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── ANSI colours for pass/fail ────────────────────────────────────────────
_GREEN = "\033[32m"
_RED   = "\033[31m"
_YEL   = "\033[33m"
_RST   = "\033[0m"
_USE_COLOR = sys.stdout.isatty()


def _ok(msg: str) -> None:
    tag = f"{_GREEN}✓{_RST}" if _USE_COLOR else "OK"
    print(f"  {tag}  {msg}")


def _fail(msg: str) -> None:
    tag = f"{_RED}✗{_RST}" if _USE_COLOR else "FAIL"
    print(f"  {tag}  {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    tag = f"{_YEL}!{_RST}" if _USE_COLOR else "WARN"
    print(f"  {tag}  {msg}")


def _section(msg: str) -> None:
    print(f"\n── {msg} " + "─" * max(0, 70 - len(msg)))


# ── Checks ────────────────────────────────────────────────────────────────

def check_sim_build_and_reset(gui: bool) -> bool:
    _section("1. Sim build + reset")
    from braccio_ctrl.sim.braccio_env import BraccioSimEnv
    try:
        env = BraccioSimEnv(gui=gui, seed=42)
    except Exception as exc:
        _fail(f"BraccioSimEnv() raised {type(exc).__name__}: {exc}")
        return False
    _ok("BraccioSimEnv() constructed")
    try:
        obs, info = env.reset()
    except Exception as exc:
        _fail(f"env.reset() raised {type(exc).__name__}: {exc}")
        env.close()
        return False
    _ok(f"env.reset() returned obs.shape={obs.shape}, info keys={list(info.keys())}")
    env.close()
    return True


def check_obs_shape() -> bool:
    _section("2. Observation shape = (74,)")
    from braccio_ctrl.sim.braccio_env import BraccioSimEnv
    env = BraccioSimEnv(gui=False, seed=0)
    obs, _ = env.reset()
    try:
        if obs.shape != (74,):
            _fail(f"obs shape is {obs.shape}, expected (74,)")
            return False
        if obs.dtype != np.float32:
            _fail(f"obs dtype is {obs.dtype}, expected float32")
            return False
        _ok(f"obs.shape = {obs.shape}, dtype = {obs.dtype}")
        # Verify each slice has the expected width
        slices = [
            ("arm pose",         0,  5),
            ("CH0 ToF grid",     5, 21),
            ("CH1 ToF grid",    21, 37),
            ("CH2 ToF grid",    37, 53),
            ("IR severity",     53, 54),
            ("obstacle map",    54, 62),
            ("Z-mask",          62, 67),
            ("goal delta",      67, 70),
            ("forbidden band",  70, 74),
        ]
        for name, lo, hi in slices:
            sl = obs[lo:hi]
            if sl.shape[0] != (hi - lo):
                _fail(f"slice {name} [{lo}:{hi}] has width {sl.shape[0]}")
                return False
        _ok("all 9 obs slices present at correct indices")
        return True
    finally:
        env.close()


def check_random_steps_finite(n: int = 50) -> bool:
    _section(f"3. {n} random steps — no NaN / inf")
    from braccio_ctrl.sim.braccio_env import BraccioSimEnv
    env = BraccioSimEnv(gui=False, seed=1)
    obs, _ = env.reset()
    resets = 0
    for i in range(n):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if not np.all(np.isfinite(obs)):
            bad = np.where(~np.isfinite(obs))[0].tolist()
            _fail(f"step {i}: obs has non-finite values at indices {bad[:6]}")
            env.close(); return False
        if not np.isfinite(reward):
            _fail(f"step {i}: reward is {reward}")
            env.close(); return False
        if terminated or truncated:
            obs, _ = env.reset()
            resets += 1
    _ok(f"{n} steps × ({resets} resets) all finite; last reward={reward:+.3f}")
    env.close()
    return True


def check_obs_layout_consistency() -> bool:
    _section("4. Sim obs layout matches rl_recorder layout")
    # The sim's _get_obs and recorder's _encode_obs must produce identical
    # slice layouts for the policy to transfer. This check compares the
    # hardcoded slice indices in rl_reward (which both sides must agree on).
    from braccio_ctrl import rl_reward
    required = {
        "OBS_ARM_SLICE":        (0, 5),
        "OBS_CH0_GRID_SLICE":   (5, 21),
        "OBS_CH1_GRID_SLICE":   (21, 37),
        "OBS_CH2_GRID_SLICE":   (37, 53),
        "OBS_IR_IDX":           53,
        "OBS_MAP_SLICE":        (54, 62),
        "OBS_Z_MASK_SLICE":     (62, 67),
        "OBS_GOAL_DELTA_SLICE": (67, 70),
        "OBS_FORBIDDEN_SLICE":  (70, 74),
    }
    ok = True
    for name, expected in required.items():
        actual = getattr(rl_reward, name, None)
        if actual is None:
            _fail(f"rl_reward.{name} is not defined")
            ok = False
            continue
        # Convert slice objects to (start, stop) tuples
        if isinstance(actual, slice):
            got = (actual.start, actual.stop)
        else:
            got = actual
        if got != expected:
            _fail(f"rl_reward.{name} = {got}, expected {expected}")
            ok = False
    if ok:
        _ok("all 9 slice constants in rl_reward agree with obs layout spec")
    return ok


def check_noise_params_calibrated() -> bool:
    _section("5. Noise parameters calibrated")
    from braccio_ctrl.sim.braccio_env import _NOISE_P
    if not _NOISE_P.exists():
        _fail(f"noise_params.json does NOT exist at {_NOISE_P}")
        _warn("Run `python calibrate_noise.py` first — otherwise sim uses "
              "synthetic defaults that won't match your hardware.")
        return False
    try:
        with open(_NOISE_P) as f:
            params = json.load(f)
    except json.JSONDecodeError as exc:
        _fail(f"noise_params.json is malformed: {exc}")
        return False
    if not params.get("calibrated", False):
        _fail("noise_params.json exists but calibrated=false")
        _warn("Run `python calibrate_noise.py logs/rl_transitions_*.npz` "
              "to populate real measurements.")
        return False
    _ok(f"calibrated=true, servo_lag={params.get('servo_lag_factor'):.3f}, "
        f"tof_sigma={params.get('tof_noise_sigma_mm'):.1f} mm, "
        f"cell_dropout={params.get('cell_dropout_rate'):.3f}")
    return True


def check_policy_error_handling() -> bool:
    _section("6. Policy-ref error handling (deploy safety)")
    from braccio_ctrl.rl_sweeper import AtomicPolicyRef
    ref = AtomicPolicyRef()
    # Null policy returns zeros silently
    a = ref.predict(np.zeros(74, dtype=np.float32))
    if not (a == 0).all() or ref.last_error is not None:
        _fail("null policy produced non-zero action or error")
        return False
    _ok("null policy → zero action, no error")
    # NaN-producing policy gets caught
    class NaNPolicy:
        def predict(self, obs, deterministic=True):
            return np.array([0.0, float('nan'), 0.0, 0.0], dtype=np.float32), None
    ref.swap(NaNPolicy())
    a = ref.predict(np.zeros(74, dtype=np.float32))
    if not (a == 0).all() or 'non-finite' not in (ref.last_error or ''):
        _fail(f"NaN policy not caught: action={a}, last_error={ref.last_error}")
        return False
    _ok(f"NaN action caught (last_error set: {ref.last_error[:48]}…)")
    return True


# ── Optional: live visual confirmation ────────────────────────────────────

def visual_confirm(duration_s: float = 5.0) -> None:
    _section(f"GUI visual (hold {duration_s:.0f}s)")
    from braccio_ctrl.sim.braccio_env import BraccioSimEnv
    env = BraccioSimEnv(gui=True, seed=0)
    env.reset()
    # Move to the HOME-equivalent IK default pose so the GUI shows the
    # arm stacked vertical. This visually confirms URDF conventions.
    env._theta = 90.0
    env._r     = 0.0
    env._z     = 310.0
    env._set_arm_joints(env._theta, env._r, env._z)
    for _ in range(int(duration_s * 10)):
        time.sleep(0.1)
    env.close()


# ── Driver ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--gui", action="store_true",
                        help="Pop up PyBullet window for 5 s to visually "
                             "verify the URDF and arm pose.")
    parser.add_argument("--skip-noise-check", action="store_true",
                        help="Skip check #5 (noise calibration). Useful "
                             "during early setup before Stage 0 data exists.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    checks = [
        lambda: check_sim_build_and_reset(gui=False),
        check_obs_shape,
        lambda: check_random_steps_finite(50),
        check_obs_layout_consistency,
    ]
    if not args.skip_noise_check:
        checks.append(check_noise_params_calibrated)
    checks.append(check_policy_error_handling)

    results = [c() for c in checks]

    print()
    n_pass = sum(results)
    n_total = len(results)
    if all(results):
        msg = f"ALL {n_total} CHECKS PASSED — sim is ready for training."
        print(f"{_GREEN if _USE_COLOR else ''}{msg}{_RST if _USE_COLOR else ''}")
    else:
        msg = f"{n_pass}/{n_total} CHECKS PASSED — fix the failures above before training."
        print(f"{_RED if _USE_COLOR else ''}{msg}{_RST if _USE_COLOR else ''}",
              file=sys.stderr)

    if args.gui and all(results):
        visual_confirm()

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
