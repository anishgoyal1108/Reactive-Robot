"""
train_bc.py — Behavioural cloning warm-start for the Braccio RL policy.

Labels come from the goal-delta already encoded in obs[67:70], NOT from the
logged NPZ actions.  The logged actions were produced by the hard-coded state
machine, which makes its own replanning mistakes.  The goal-delta is always
unambiguous: it is the geometric vector from the current arm pose to the
current goal, computed at record time from the sequence editor's target state.

BC target for every transition:
  action[0] = clip(goal_dtheta, -1, 1)   move theta toward goal
  action[1] = clip(goal_dr,     -1, 1)   move r toward goal
  action[2] = clip(goal_dz,     -1, 1)   move z toward goal
  action[3] = 0.0                         hold current slew rate

Transitions are filtered to:
  - ir_bits == 0  (arm not in collision or danger during collection)
  - |goal_delta| > GOAL_MIN_NORM  (goal is set and arm is not already there)

Output: bc_policy.pt
  Compatible with load_bc_warmstart() for transfer into a Stable-Baselines3
  SAC actor (net_arch=[256, 256]).

Usage:
  python train_bc.py                         # auto-finds logs/rl_transitions_*.npz
  python train_bc.py logs/session1.npz ...   # explicit files
  python train_bc.py --epochs 200 --lr 3e-4
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Obs layout (must match rl_recorder._encode_obs) ─────────────────────────
OBS_DIM          = 74
OBS_IR_IDX       = 53               # obs[53] = ir_bits / 3.0
OBS_GOAL_SLICE   = slice(67, 70)    # [goal_dtheta, goal_dr, goal_dz]

# ── Action layout ────────────────────────────────────────────────────────────
ACT_DIM          = 4                # [Δtheta_n, Δr_n, Δz_n, Δdelta_n]

# ── Filtering thresholds ─────────────────────────────────────────────────────
IR_CLEAR_THRESH  = 0.12             # obs[53] < this  →  ir_bits == 0
GOAL_MIN_NORM    = 0.05             # skip transitions where arm is already at goal

# ── Training hyperparameters ─────────────────────────────────────────────────
HIDDEN           = 256
BATCH_SIZE       = 256
EPOCHS           = 100
LR               = 1e-3
VAL_FRAC         = 0.10
PATIENCE         = 10               # epochs without val improvement → stop early
GRAD_CLIP        = 1.0

DEFAULT_OUT      = "bc_policy.pt"
LOG_DIR          = "logs"

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Network ──────────────────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    """
    74 → 256 → 256 → 3 MLP with Tanh output.

    Matches the actor architecture used by Stable-Baselines3 SAC when
    policy_kwargs=dict(net_arch=[256, 256]).  See load_bc_warmstart() for
    the weight-transfer mapping.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
            nn.Linear(hidden,  act_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Label derivation ─────────────────────────────────────────────────────────

def _bc_label(obs: np.ndarray) -> np.ndarray:
    """
    Derive a BC target action from the goal-delta already in the obs vector.

    obs[67] = (goal_theta - theta) / 90   →  action[0]  (Δtheta, normalised)
    obs[68] = (goal_r    - r)     / 115   →  action[1]  (Δr,     normalised)
    obs[69] = (goal_z    - z)     / 125   →  action[2]  (Δz,     normalised)
    action[3] = 0.0                        →  hold current slew rate
    """
    goal = obs[OBS_GOAL_SLICE]                          # [dtheta, dr, dz]
    a0   = float(np.clip(goal[0], -1.0, 1.0))          # theta toward goal
    a1   = float(np.clip(goal[1], -1.0, 1.0))          # r toward goal
    a2   = float(np.clip(goal[2], -1.0, 1.0))          # z toward goal
    return np.array([a0, a1, a2, 0.0], dtype=np.float32)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_transitions(paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and filter obs from NPZ files; compute BC labels.

    Filters applied per transition:
      1. ir_bits == 0  (safe — arm not in collision)
      2. |goal_delta| > GOAL_MIN_NORM  (goal is active and not yet reached)

    Returns
    -------
    obs_out    : (N, 74) float32  — real sensor observations
    labels_out : (N, 3)  float32  — goal-derived target actions
    """
    all_obs: List[np.ndarray]    = []
    all_labels: List[np.ndarray] = []

    for path in paths:
        try:
            z = np.load(path, allow_pickle=False)
        except Exception as exc:
            log.warning("Skip %s: %s", path, exc)
            continue

        obs   = z['obs'].astype(np.float32)   # (N, 74); actions array ignored
        n_raw = len(obs)

        ir_clear  = obs[:, OBS_IR_IDX] < IR_CLEAR_THRESH
        goal_norm = np.linalg.norm(obs[:, OBS_GOAL_SLICE], axis=1)
        has_goal  = goal_norm > GOAL_MIN_NORM

        mask   = ir_clear & has_goal
        obs    = obs[mask]
        labels = np.stack([_bc_label(o) for o in obs])

        all_obs.append(obs)
        all_labels.append(labels)
        log.info("%-45s  %d / %d transitions kept",
                 Path(path).name, len(obs), n_raw)

    if not all_obs:
        log.error("No valid transitions found.  Collect data first.")
        sys.exit(1)

    obs_out    = np.concatenate(all_obs,    axis=0)
    labels_out = np.concatenate(all_labels, axis=0)
    log.info("Total: %d transitions from %d file(s)", len(obs_out), len(all_obs))
    return obs_out, labels_out


# ── Training ─────────────────────────────────────────────────────────────────

def train(npz_paths: List[str],
          out_path:  str           = DEFAULT_OUT,
          epochs:    int           = EPOCHS,
          lr:        float         = LR,
          device_str: Optional[str] = None) -> BCPolicy:

    device = torch.device(
        device_str if device_str
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info("Device: %s", device)

    obs_np, labels_np = load_transitions(npz_paths)

    obs_t    = torch.from_numpy(obs_np)
    labels_t = torch.from_numpy(labels_np)

    full_ds = TensorDataset(obs_t, labels_t)
    n_val   = max(1, int(len(full_ds) * VAL_FRAC))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)

    policy    = BCPolicy().to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    best_val   = float("inf")
    best_state: Optional[dict] = None
    patience   = 0

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        policy.train()
        t_loss = 0.0
        for obs_b, lbl_b in train_loader:
            obs_b, lbl_b = obs_b.to(device), lbl_b.to(device)
            optimiser.zero_grad()
            loss = loss_fn(policy(obs_b), lbl_b)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimiser.step()
            t_loss += loss.item() * len(obs_b)
        t_loss /= n_train

        # ── validate ──────────────────────────────────────────────────────
        policy.eval()
        v_loss = 0.0
        with torch.no_grad():
            for obs_b, lbl_b in val_loader:
                obs_b, lbl_b = obs_b.to(device), lbl_b.to(device)
                v_loss += loss_fn(policy(obs_b), lbl_b).item() * len(obs_b)
        v_loss /= n_val

        if epoch % 10 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  train=%.5f  val=%.5f",
                     epoch, epochs, t_loss, v_loss)

        # ── early stopping ────────────────────────────────────────────────
        if v_loss < best_val - 1e-6:
            best_val   = v_loss
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                log.info("Early stop at epoch %d  best_val=%.5f", epoch, best_val)
                break

    if best_state is not None:
        policy.load_state_dict(best_state)

    torch.save({
        "state_dict": policy.state_dict(),
        "obs_dim":    OBS_DIM,
        "act_dim":    ACT_DIM,
        "hidden":     HIDDEN,
        "val_mse":    best_val,
    }, out_path)
    log.info("Saved → %s  (best val MSE = %.5f)", out_path, best_val)
    return policy


# ── SB3 warm-start helper (imported by train_rl.py) ──────────────────────────

def load_bc_warmstart(sac_model, bc_path: str = DEFAULT_OUT,
                      device_str: Optional[str] = None) -> None:
    """
    Transfer BC weights into a Stable-Baselines3 SAC actor.

    SB3 SAC actor layout with policy_kwargs=dict(net_arch=[256, 256]):
      actor.latent_pi : Sequential  [Linear(74→256), ReLU,
                                     Linear(256→256), ReLU]
      actor.mu        : Linear(256→4)

    BCPolicy layout:
      net : Sequential  [Linear(74→256),  ReLU,    # net.0 / net.1
                         Linear(256→256), ReLU,    # net.2 / net.3
                         Linear(256→4),   Tanh]    # net.4 / net.5

    SB3 applies its own squashing (Tanh + log-prob correction) separately,
    so we copy only the weight matrices — the Tanh from BCPolicy is not
    transferred.
    """
    device = torch.device(
        device_str if device_str
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt  = torch.load(bc_path, map_location=device)
    sd    = ckpt["state_dict"]
    actor = sac_model.policy.actor

    with torch.no_grad():
        actor.latent_pi[0].weight.copy_(sd["net.0.weight"])
        actor.latent_pi[0].bias.copy_(  sd["net.0.bias"])
        actor.latent_pi[2].weight.copy_(sd["net.2.weight"])
        actor.latent_pi[2].bias.copy_(  sd["net.2.bias"])
        actor.mu.weight.copy_(          sd["net.4.weight"])
        actor.mu.bias.copy_(            sd["net.4.bias"])

    log.info("BC warm-start loaded from %s  (val MSE=%.5f)",
             bc_path, ckpt.get("val_mse", float("nan")))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(
        description="BC warm-start training for Braccio RL policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("npz_paths", nargs="*",
                   help="NPZ files to train on.  Defaults to logs/rl_transitions_*.npz")
    p.add_argument("--out",     default=DEFAULT_OUT, help="Output path for bc_policy.pt")
    p.add_argument("--epochs",  type=int,   default=EPOCHS)
    p.add_argument("--lr",      type=float, default=LR)
    p.add_argument("--device",  default=None, help="'cuda' / 'cpu' / None=auto")
    args = p.parse_args()

    paths = args.npz_paths or sorted(
        glob.glob(os.path.join(LOG_DIR, "rl_transitions_*.npz"))
    )
    if not paths:
        log.error("No NPZ files found.  Run the controller first to collect data.")
        return 1

    train(paths, out_path=args.out, epochs=args.epochs,
          lr=args.lr, device_str=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
