# Digital-Twin verification workflow

Run this after finishing Stage 0 data collection and before launching
`train_rl.py`. Takes ~30 seconds.

## Quick check (headless)

```bash
cd braccio_main_runner
python verify_sim.py
```

Expected output (all green):

```
── 1. Sim build + reset ──────────────────────────────────────────────────
  ✓  BraccioSimEnv() constructed
  ✓  env.reset() returned obs.shape=(74,), info keys=[...]

── 2. Observation shape = (74,) ──────────────────────────────────────────
  ✓  obs.shape = (74,), dtype = float32
  ✓  all 9 obs slices present at correct indices

── 3. 50 random steps — no NaN / inf ─────────────────────────────────────
  ✓  50 steps × (N resets) all finite; last reward=-0.017

── 4. Sim obs layout matches rl_recorder layout ──────────────────────────
  ✓  all 9 slice constants in rl_reward agree with obs layout spec

── 5. Noise parameters calibrated ────────────────────────────────────────
  ✓  calibrated=true, servo_lag=0.91, tof_sigma=12.4 mm, cell_dropout=0.18

── 6. Policy-ref error handling (deploy safety) ──────────────────────────
  ✓  null policy → zero action, no error
  ✓  NaN action caught (last_error set: …)

ALL 6 CHECKS PASSED — sim is ready for training.
```

**Exit code 0 = go; exit code 1 = don't train until fixed.** Check 5 will
fail until you run `python calibrate_noise.py`; temporarily skip it with
`--skip-noise-check` if you want to test the other items first.

## Visual confirmation (GUI)

```bash
python verify_sim.py --gui
```

Pops up a PyBullet window for 5 s at the HOME-equivalent IK pose
`(θ=90°, r=0, z=310)`. Visually confirm:

- Arm is stacked vertical (upper arm, forearm, gripper all pointing up).
- Base is centered (not rotated 90° or 180° off).
- The URDF mesh renders correctly (no missing links).

**If the arm appears flipped or rotated in the GUI,** that's *only* a
visualisation issue — the sim's physics are analytical and don't use
PyBullet FK. Training will still be correct. But it's a useful sanity
check for debugging.

## What each check protects you from

| # | Check | If it fails, you'd see... |
|---|---|---|
| 1 | Sim builds | Training crashes immediately on import |
| 2 | Obs shape `(74,)` | SAC throws on first rollout |
| 3 | 50 random steps finite | Policy gradient explodes after a handful of steps |
| 4 | Obs-layout agreement | Reward function reads wrong slices → reward is noise |
| 5 | Noise calibrated | Policy transfers terribly: trained on synthetic noise ≠ real sensor |
| 6 | Policy error handling | On deployment the arm stalls silently without explanation |

## After training, before deploying

Also sanity-check the saved model:

```bash
python -c "
from stable_baselines3 import SAC
import numpy as np
m = SAC.load('best_policy/best_model.zip')
a, _ = m.predict(np.zeros(74, dtype=np.float32), deterministic=True)
assert a.shape == (4,), f'action shape {a.shape} != (4,)'
assert np.all(np.isfinite(a)), f'non-finite action: {a}'
print(f'policy OK — sample action: {a}')
"
```

If this fails, **don't deploy** — the saved model is corrupt or the
observation/action spaces diverged between training and the saved zip.

Then launch the controller with the policy:

```bash
python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1 \
    --rl-policy best_policy/best_model.zip
```

Press Z to toggle RL sweep. Watch the controller UI for:
- `RL POLICY ERR: …` in the status line → policy prediction is failing.
  Tail `logs/controller.log` for the full exception; the arm will stall
  until this is fixed.
- Silent no-motion → the sweep may have terminated on IR DANGER;
  press Z again to resume.

## Common failures and fixes

**"noise_params.json has calibrated=false"** — you haven't run
`calibrate_noise.py` since your Stage 0 data collection finished. Run it
now and re-verify. (If you try to proceed anyway, the policy will train
on generic sensor noise and probably fail to transfer.)

**"rl_reward.OBS_*_SLICE is not defined"** — obs layout has drifted
between the sim/recorder and the reward function. Check that all three
(`rl_recorder._encode_obs`, `braccio_env._get_obs`, `rl_reward`) agree
on slice indices. The 74-float contract is documented in
`CLAUDE.md` under "Observation Vector."

**"env.reset() raised ..."** — PyBullet couldn't load the URDF. Check
`assets/braccio_description/urdf/braccio_model.urdf` exists; if not,
retrain using `assets/braccio_description/urdf/braccio_planning_model.urdf`
via `BraccioSimEnv(urdf_path=...)`.
