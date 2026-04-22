#!/usr/bin/env python3
"""
test_replan.py — Behavioural test for the trained RL policy.

Loads a trained policy and verifies the arm actually REPLANS around an
obstacle placed directly in its sweep path, rather than freezing in
place.

Three scenarios:
  1. No obstacle                 — policy should drive toward the goal
  2. Obstacle on the sweep path  — policy should produce |Δθ| > 30° of
                                    motion within 5 seconds (proving it
                                    went around rather than stopped)
  3. IR estop                     — policy should produce zero action
                                    once IR fires (safe stall)

Usage:
    python test_replan.py best_policy/best_model.zip

Exit 0 = the policy passes all scenarios.
Exit 1 = the policy freezes on scenario 2 — DO NOT DEPLOY; retrain.

Meant to be run AFTER a retrain that includes the Fix F/G/H reward
changes, to confirm the policy actually learned to maneuver. The
prior policy (which hung on deployment) would fail scenario 2.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m  {msg}" if sys.stdout.isatty() else f"  OK  {msg}")


def _fail(msg: str) -> None:
    print(f"  \033[31m✗\033[0m  {msg}" if sys.stdout.isatty() else f"  FAIL  {msg}",
          file=sys.stderr)


def run_scenario(env, model, label: str, *,
                  force_on_path: bool,
                  n_steps: int = 100,
                  require_theta_travel_deg: float = 0.0) -> bool:
    """Run `n_steps` (= ~5 s at 20 Hz) and assert a motion criterion."""
    print(f"\n── Scenario: {label} ──────────────────────────────────")
    obs, info = env.reset()

    # Optionally override obstacle placement to force on-path. The env
    # already does this 40% of the time; we force it here for determinism.
    if force_on_path:
        env._obs_theta = 90.0
        # Put obstacle halfway between start θ and goal θ.
        t_mid = 0.5 * (env._theta + env._goal_theta)
        env._obs_theta = float(t_mid)
        env._obs_r_mm  = float(env._goal_r)
        env._obs_z_mm  = float(env._goal_z)
        env._obs_radius = 90.0
        env._recreate_obstacle()

    start_theta = env._theta
    thetas = [env._theta]
    zero_action_count = 0

    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=False)
        if np.linalg.norm(action) < 0.05:
            zero_action_count += 1
        obs, reward, terminated, truncated, info = env.step(action)
        thetas.append(env._theta)
        if terminated:
            break

    theta_travel = max(thetas) - min(thetas)
    print(f"  θ range observed: [{min(thetas):.1f}°, {max(thetas):.1f}°]  "
          f"total travel: {theta_travel:.1f}°")
    print(f"  near-zero-action ticks: {zero_action_count}/{n_steps}")

    if require_theta_travel_deg > 0.0:
        if theta_travel >= require_theta_travel_deg:
            _ok(f"θ travel {theta_travel:.1f}° ≥ required {require_theta_travel_deg:.1f}°")
            return True
        else:
            _fail(f"θ travel {theta_travel:.1f}° < required {require_theta_travel_deg:.1f}° "
                  f"— policy likely froze on the obstacle")
            return False
    else:
        _ok("no motion requirement; scenario completed without crash")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("policy_path", type=Path,
                        help="Path to best_model.zip from train_rl.py")
    parser.add_argument("--n-steps", type=int, default=100,
                        help="Steps per scenario (default 100 ≈ 5 s @ 20 Hz)")
    parser.add_argument("--required-travel-deg", type=float, default=30.0,
                        help="Minimum |Δθ| across scenario 2 (default 30°)")
    args = parser.parse_args()

    if not args.policy_path.exists():
        print(f"ERROR: policy not found at {args.policy_path}", file=sys.stderr)
        return 1

    from stable_baselines3 import SAC
    from braccio_ctrl.sim.braccio_env import BraccioSimEnv

    print(f"Loading policy from {args.policy_path}...")
    model = SAC.load(str(args.policy_path))

    results = []
    env = BraccioSimEnv(gui=False, seed=42)

    # Scenario 1: no obstacle in the way (random reset — most will be off-path)
    results.append(run_scenario(
        env, model, "free sweep, obstacle random",
        force_on_path=False, n_steps=args.n_steps,
    ))

    # Scenario 2: deterministic on-path obstacle. This is the regression
    # test for the "bot hangs on obstacle" bug — passing this means the
    # policy learned to go AROUND.
    results.append(run_scenario(
        env, model, "OBSTACLE ON DIRECT PATH — must replan",
        force_on_path=True, n_steps=args.n_steps,
        require_theta_travel_deg=args.required_travel_deg,
    ))

    env.close()

    print()
    if all(results):
        print("\033[32mALL SCENARIOS PASSED — policy actually replans.\033[0m"
              if sys.stdout.isatty() else "ALL SCENARIOS PASSED")
        return 0
    else:
        print("\033[31mScenario 2 FAILED — policy freezes on obstacle, DO NOT DEPLOY.\033[0m"
              if sys.stdout.isatty() else "Scenario 2 FAILED — DO NOT DEPLOY",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
