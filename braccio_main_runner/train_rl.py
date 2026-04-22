"""
train_rl.py — SAC simulation training for the Braccio RL policy.

Pipeline:
  1. (Optional) warm-start actor from bc_policy.pt via load_bc_warmstart().
  2. Train with SAC in BraccioSimEnv for --total-steps steps (~1 M default).
  3. Save best policy (by eval return) to --out.

Domain randomisation and noise calibration are baked into BraccioSimEnv.
Running calibrate_noise.py first (after data collection) will overwrite
braccio_ctrl/sim/noise_params.json with real measured values so the sim
distribution matches the hardware.

Usage:
  python train_rl.py                            # auto, 1M steps
  python train_rl.py --bc-policy bc_policy.pt   # warm-start from BC
  python train_rl.py --total-steps 500000 --eval-freq 20000
  python train_rl.py --gui                      # show PyBullet viewer

Outputs:
  best_policy/best_model.zip   — best eval checkpoint (SB3 format)
  rl_policy_final.zip          — final model after training

The saved model can be loaded by RLSweeper (Stage 5) via:
  from stable_baselines3 import SAC
  model = SAC.load("best_policy/best_model.zip")
  action, _ = model.predict(obs, deterministic=True)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TOTAL_STEPS  = 1_000_000
DEFAULT_EVAL_FREQ    = 25_000
DEFAULT_EVAL_EPISODES = 20
DEFAULT_N_ENVS       = 4        # parallel training envs (SubprocVecEnv)
DEFAULT_OUT          = "rl_policy_final.zip"
DEFAULT_BEST_DIR     = "best_policy"

# SAC hyper-parameters
SAC_LR              = 3e-4
SAC_BUFFER          = 500_000
SAC_BATCH           = 256
SAC_GAMMA           = 0.99
SAC_TAU             = 0.005
SAC_ENT_COEF        = "auto"
SAC_NET_ARCH        = [256, 256]
SAC_LEARNING_STARTS = 10_000    # random exploration before learning


def _make_env(gui: bool = False, seed: Optional[int] = None):
    """Factory for a single BraccioSimEnv (used by VecEnv)."""
    # Lazy import to avoid py_trees / pybullet_planning at module level
    from braccio_ctrl.sim.braccio_env import BraccioSimEnv  # type: ignore

    def _init():
        env = BraccioSimEnv(gui=gui, seed=seed)
        return env

    return _init


def _build_vec_env(n_envs: int, gui: bool, base_seed: int):
    """
    Build a vectorised training environment.
    Uses SubprocVecEnv for n_envs > 1 (parallel processes), DummyVecEnv for 1.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    factories = [_make_env(gui=(gui and i == 0), seed=base_seed + i)
                 for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="fork")


def train(
    total_steps: int    = DEFAULT_TOTAL_STEPS,
    eval_freq: int      = DEFAULT_EVAL_FREQ,
    eval_episodes: int  = DEFAULT_EVAL_EPISODES,
    n_envs: int         = DEFAULT_N_ENVS,
    bc_policy_path: Optional[str] = None,
    out_path: str       = DEFAULT_OUT,
    best_dir: str       = DEFAULT_BEST_DIR,
    gui: bool           = False,
    seed: int           = 42,
    device: str         = "auto",
) -> str:
    """
    Run SAC training in BraccioSimEnv.

    Returns path to the best saved model checkpoint.
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    # ── Environments ─────────────────────────────────────────────────────────
    log.info("Building %d training env(s) ...", n_envs)
    train_env = _build_vec_env(n_envs, gui=gui, base_seed=seed)

    log.info("Building eval env ...")
    eval_env = DummyVecEnv([_make_env(gui=False, seed=seed + 1000)])

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Constructing SAC model ...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate    = SAC_LR,
        buffer_size      = SAC_BUFFER,
        batch_size       = SAC_BATCH,
        gamma            = SAC_GAMMA,
        tau              = SAC_TAU,
        ent_coef         = SAC_ENT_COEF,
        learning_starts  = SAC_LEARNING_STARTS,
        policy_kwargs    = dict(net_arch=SAC_NET_ARCH),
        verbose          = 1,
        seed             = seed,
        device           = device,
    )

    # ── BC warm-start ─────────────────────────────────────────────────────────
    if bc_policy_path and Path(bc_policy_path).exists():
        log.info("Loading BC warm-start from %s ...", bc_policy_path)
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from train_bc import load_bc_warmstart
            load_bc_warmstart(model, bc_policy_path)
        except Exception as exc:
            log.warning("BC warm-start failed (%s) — training from random init", exc)
    elif bc_policy_path:
        log.warning("bc_policy.pt not found at %s — skipping warm-start", bc_policy_path)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(best_dir, exist_ok=True)

    # Early-stopping on mean reward is DISABLED. With the replan + half-
    # sweep bonuses introduced in rl_reward.py (Fix F/H), the previous
    # threshold=50 triggered at 10% of the budget because the policy
    # harvested half-sweep bonuses in free-sweep episodes without ever
    # learning on-path replan behavior. The only signal that actually
    # tells you if the policy works is the behavioural test_replan.py,
    # run AFTER training completes. Let 1M steps run.
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = best_dir,
        log_path             = best_dir,
        eval_freq            = max(eval_freq // n_envs, 1),
        n_eval_episodes      = eval_episodes,
        deterministic        = True,
        render               = False,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info("Starting SAC training for %d steps ...", total_steps)
    model.learn(
        total_timesteps = total_steps,
        callback        = eval_cb,
        progress_bar    = True,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save(out_path)
    log.info("Final model saved → %s", out_path)

    best_path = str(Path(best_dir) / "best_model.zip")
    log.info("Best checkpoint at  → %s", best_path)

    train_env.close()
    eval_env.close()

    return best_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(
        description="SAC training for Braccio RL policy in PyBullet simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--total-steps",    type=int,   default=DEFAULT_TOTAL_STEPS,
                   help="Total environment steps")
    p.add_argument("--eval-freq",      type=int,   default=DEFAULT_EVAL_FREQ,
                   help="Evaluation frequency in env steps")
    p.add_argument("--eval-episodes",  type=int,   default=DEFAULT_EVAL_EPISODES,
                   help="Number of episodes per evaluation")
    p.add_argument("--n-envs",         type=int,   default=DEFAULT_N_ENVS,
                   help="Number of parallel training environments")
    p.add_argument("--bc-policy",      default=None,
                   help="Path to bc_policy.pt for warm-start (optional)")
    p.add_argument("--out",            default=DEFAULT_OUT,
                   help="Output path for final model (.zip)")
    p.add_argument("--best-dir",       default=DEFAULT_BEST_DIR,
                   help="Directory to save best checkpoint")
    p.add_argument("--gui",            action="store_true",
                   help="Enable PyBullet GUI for first training env")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="auto",
                   help="PyTorch device: auto | cpu | cuda | cuda:0")
    p.add_argument("--bc-only-warmstart", action="store_true",
                   help="Only run BC warm-start then exit (for debugging)")
    args = p.parse_args()

    if args.bc_only_warmstart:
        # Just build the model + warm-start, print shapes, exit
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([_make_env(gui=False, seed=args.seed)])
        model = SAC("MlpPolicy", env,
                    policy_kwargs=dict(net_arch=SAC_NET_ARCH),
                    device=args.device)
        if args.bc_policy and Path(args.bc_policy).exists():
            sys.path.insert(0, str(Path(__file__).parent))
            from train_bc import load_bc_warmstart
            load_bc_warmstart(model, args.bc_policy)
            print("Warm-start OK — actor shapes:")
            for name, p_ in model.policy.actor.named_parameters():
                print(f"  {name:40s} {tuple(p_.shape)}")
        env.close()
        return 0

    best = train(
        total_steps    = args.total_steps,
        eval_freq      = args.eval_freq,
        eval_episodes  = args.eval_episodes,
        n_envs         = args.n_envs,
        bc_policy_path = args.bc_policy,
        out_path       = args.out,
        best_dir       = args.best_dir,
        gui            = args.gui,
        seed           = args.seed,
        device         = args.device,
    )
    print(f"\nTraining complete.  Best model: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
