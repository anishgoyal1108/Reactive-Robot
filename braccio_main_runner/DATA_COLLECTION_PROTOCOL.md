# Data Collection Protocol — Stage 0

Run **all 6 scenarios** in a single session if possible (~70 min total).
Press `G` (start ToF log) **before each scenario**.
The RL recorder logs automatically — just have the controller running.

---

## Pipeline this NPZ feeds into (Q2: SAC-from-sim)

```
NPZ files (this protocol)
   │
   ▼
calibrate_noise.py   — fits ToF sigma + servo lag + cell dropout rate
   │
   ▼
sim/noise_params.json
   │
   ▼
train_rl.py          — SAC-from-random-init in PyBullet, ~1M steps
   │
   ▼
best_policy/best_model.zip
   │
   ▼
python -m braccio_ctrl … --rl-policy best_policy/best_model.zip
   │
   ▼
(optional) rl_online_trainer.py  — live fine-tune with hot-swap
```

**No behavioral cloning step.** The safety BT does not expose its internal
target waypoint to `rl_recorder.set_goal()`, so the goal-delta labels in
collected NPZ are all zero — actively harmful as warm-start labels. The
sim policy starts from random and learns purely from `rl_reward.py`.

---

## What NPZ data is used for

| Data | Used for | NOT used for |
|------|----------|--------------|
| `obs` arrays | Noise calibration (`calibrate_noise.py`) | Training actions |
| `actions` arrays | Noise calibration (servo lag estimate) | Behavioral cloning |
| `rewards` arrays | Ignored (placeholder zeros) | Any training |
| Annotated NPZ (path annotator output) | Optional supervised fine-tune signal | Main training loop |

The obs arrays are real sensor data and are used only to calibrate simulation
noise parameters (ToF sigma, servo lag, cell dropout rate) so the sim matches
the real hardware distribution.

---

## Why hardware testing is part of the pipeline at all

Two distinct hardware phases:

1. **Stage 0 (this protocol — TONIGHT)**: Capture real ToF noise / servo lag
   / cell dropout statistics so the sim distributions match hardware. Without
   this, the sim policy learns against an idealised noise model and degrades
   on transfer.

2. **Stage 4 (after deployment, OPTIONAL)**: Run `rl_online_trainer.py`
   alongside the deployed policy. Real transitions accumulate in
   `SharedReplayBuffer`; SAC gradient steps refine the policy at a
   conservative learning rate; weights hot-swap into the live RLSweeper
   without pausing the arm. Skip Stage 4 if Stage 3 deployment behaves well.

---

## About the safety BT during data collection

**Let it replan. Do not try to suppress it.** When you hold a hand or
cylinder near the ToF sensors, the safety BT will trigger polar-skip,
Z-ladder, BiRRT, or direction-flip recovery. Every one of those maneuvers
is recorded by RLRecorder and feeds straight into noise calibration:
- Replan transitions stress-test the servo lag estimator (commanded vs.
  actual joint deltas under abrupt motion changes).
- Partial occlusion during avoidance maneuvers exposes ToF cell dropout
  patterns the noise model needs.
- The full obstacle distance distribution from "far" to "near-collision"
  is captured naturally as the BT reacts.

If you suppressed BT replanning to get "clean" motion data, you would
get a less accurate noise model. The recorder is observational — it does
not care what made the arm move.

---

## About the IR sensor

IR severity (`ir_bits`) appears in **obs[53]** as a passive informational
feature. The trained RL policy will see it, weight it as it sees fit, and
use it as one of many obstacle signals.

**Not used for episode termination during data collection.** The recorder
never inserts artificial done=True boundaries on IR DANGER. Doing so would
corrupt the Bellman targets during downstream training. Physical safety is
already provided by the safety BT's emergency-stop pathway and the IR
hardware itself; the recorder just observes.

If your IR wiring is loose or producing false positives, launch with
`--no-ir` to keep ir_bits forced to zero (ToF stays active).

---

## Object for obstacle scenarios

Use a **cylindrical object 6–10 cm diameter** (water bottle, coffee mug, tin
can).  This size matches the simulated obstacle sphere (40–120 mm radius) used
in domain randomisation.

---

## Step 1 — Save workspace states (~5 min)

Open controller: `python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1`

Navigate with A/D/W/S/Q/E keys to each pose, then press
**M → "Save current state" → type name → Enter**.

| State name   | theta (°) | r (mm) | z (mm) | Notes                         |
|--------------|-----------|--------|--------|-------------------------------|
| `seq_start`  |  30       |  152   |  60    | Left edge of sweep            |
| `seq_mid`    |  90       |  152   |  60    | Center                        |
| `seq_end`    | 150       |  152   |  60    | Right edge of sweep           |
| `seq_high`   |  90       |  152   |  95    | Upper Z level                 |
| `seq_low`    |  90       |  152   |   0    | Lower Z level                 |
| `seq_close`  |  90       |   80   |  60    | Short reach                   |
| `seq_far`    |  90       |  220   |  60    | Extended reach                |
| `corner_tl`  |  30       |  152   |  95    | Top-left workspace corner     |
| `corner_tr`  | 150       |  152   |  95    | Top-right workspace corner    |
| `corner_bl`  |  30       |  152   |   0    | Bottom-left workspace corner  |
| `corner_br`  | 150       |  152   |   0    | Bottom-right workspace corner |

> A/D: theta ±5°, W/S: r ±10 mm, Q/E: z ±10 mm.

---

## Scenario 1 — Free sweep, no obstacles (15 min)

1. Press `G`, then `Z` to start sweep.
2. Walk away. Let it run 15 min completely unattended.
3. Press `Z` to stop.

**Calibration value**: Captures the clean ToF noise floor and servo lag at
typical sweep velocities.  This is the primary input to `calibrate_noise.py`.

---

## Scenario 2 — Hand obstacle, CH0 side (15 min)

Start sweep (`Z`), then interact:

| Sub-scenario | What to do | Duration |
|---|---|---|
| 2a | Palm toward arm, theta≈45°, z≈60 mm, 20 cm away. Slowly move 25 cm → 12 cm. | 4 min |
| 2b | Same but z≈0 mm (low). Arm tries Z-bypass upward. | 3 min |
| 2c | Same but z≈90 mm (high). Arm tries Z-bypass downward. | 3 min |
| 2d | Slowly move hand along theta (45° → 90°) at r≈20 cm while arm sweeps. | 5 min |

**Calibration value**: ToF partial-occlusion patterns and dropout rates in the
presence of a soft obstacle.  Also captures cell-level noise at varying distances.

---

## Scenario 3 — Both-side obstacles (10 min)

Place the cylinder at theta=90°, r≈150 mm, z≈60 mm.
Start sweep (`Z`), then hold a hand on the opposite side.

| Sub-scenario | What to do |
|---|---|
| 3a | Cylinder only: 3 min. |
| 3b | Cylinder + hand opposite: 4 min (forces Z bypass). |
| 3c | Cylinder at z=30 mm (low): 3 min (arm should go over it). |

**Calibration value**: ToF noise under two simultaneous near-field returns —
the hardest scenario for the noise model to calibrate.

---

## Scenario 4 — Sequence editor runs (10 min)

1. Press `X` to open sequence editor.
2. Load `collect_seq.txt`.
3. Run 3–4 cycles. While running, place obstacle at theta=60°, r=150 mm.

**Calibration value**: Non-zero goal delta in obs[67:70] throughout.  Verifies
that the goal-delta feature captures real workspace distances correctly and that
obs[3] (dir_to_goal) flips correctly as the arm passes through each waypoint.

---

## Scenario 5 — Manual joystick exploration (10 min)

No sweep. Use A/D/W/S/Q/E to drive the arm through the full workspace.

| Area | Keys |
|---|---|
| Theta limits 5° and 175° | A/D to extremes |
| r limits 80 mm and 220 mm | S/W to extremes |
| z limits 0 mm and 95 mm | E/Q to extremes |
| All 4 workspace corners | Combine extremes |
| Mid-range with obstacle at 15 cm | Move arm toward it slowly |

**Calibration value**: Obs coverage at workspace edges that sweep never visits.
Essential for the noise model to generalise across the full (theta, r, z) range.

---

## Scenario 6 — Controlled near-misses (10 min)

Cylinder at theta=90°, r=150 mm, z=60 mm. Start sweep (`Z`).

| Sub-scenario | Distance from arm path | Effect |
|---|---|---|
| 6a | ≈220 mm | Soft REPLAN, gentle arc |
| 6b | ≈120 mm | Hard REPLAN, back-away |
| 6c | ≈80 mm | IR may fire |
| 6d | 300 mm → 80 mm slowly over 2 min | Gradual proximity ramp |

**Calibration value**: The proximity shaping terms in `rl_reward.py` (terms 4
and 6) need realistic close-range ToF distributions to be meaningful in sim.
These transitions give the noise model its close-range sigma estimate.

---

## After collection

1. Press `ESC` to quit — this flushes NPZ buffers to `logs/`.
   The controller prints `✓ RL transitions saved: N → logs/rl_transitions_*.npz`.
   If you instead see `WARNING: rl_recorder stopped with 0 transitions buffered`
   or `REC FATAL: …`, check `logs/controller.log` before re-running.
2. Verify: `ls -lh logs/rl_transitions_*.npz` (expect ≥3 files, ≥20 MB total).
3. Run noise calibration:
   ```
   cd braccio_main_runner
   python calibrate_noise.py
   ```
   This writes `braccio_ctrl/sim/noise_params.json` with `calibrated: true`.
   Check the output — if `servo_lag_factor` is outside [0.75, 1.25] or
   `tof_noise_sigma_mm` is outside [5, 20], inspect the NPZ files for anomalies.

4. **Optional** — annotate one obstacle-avoidance session for supervised signal:
   ```
   python path_annotator.py logs/rl_transitions_YYYYMMDD_HHMMSS.npz
   ```
   Draw ideal paths over the actual arm path in the GUI, then export.
   The annotated NPZ is used in `train_rl.py` as a small supervised fine-tune
   after SAC convergence — it is not used for pre-training.

5. Proceed to sim training:
   ```
   python train_rl.py
   ```
   SAC trains from random initialisation in PyBullet using `rl_reward.py` as
   ground truth.  No behavioral cloning step.

6. Deploy on hardware:
   ```
   python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1 \
       --rl-policy best_policy/best_model.zip
   ```
   The Z key now starts/stops the RLSweeper instead of the safety BT sweep.
   `--rl-policy` is the only switch — everything else (sequence editor,
   manual keys, IMU calibration, ToF logging) behaves identically.

---

## Logging reminder

- `G` key → start/stop **ToF CSV** log (`logs/tof_TIMESTAMP.csv`)
- RL transitions log automatically at SWEEP_TICK_HZ — no key needed.
- The recorder's status (running / buffered samples / last save / fatal)
  is shown live in the controller's status line via `last_resp` /
  `last_error` fields, so you can confirm during a long walk-away session
  that transitions are still accumulating.
- NPZ `actions` arrays are logged but **not used for training**.
  They exist solely so `calibrate_noise.py` can estimate servo lag from
  commanded vs. actual position changes.

---

## Sanity check before walking away

Before each scenario, glance at the controller status line:
- `rec N samp` should be incrementing — if it sticks at 0, something is
  wrong (check `logs/controller.log` for `RLRecorder tick error`).
- `REC FATAL: …` — recorder gave up after 10 consecutive errors. Stop
  and investigate before continuing; otherwise the rest of the session
  produces no NPZ.
- ToF panel should show non-NaN cells in CH0/CH1 within ~1 second of
  enabling sweep. If the grid is solid grey, the Teensy isn't streaming.
