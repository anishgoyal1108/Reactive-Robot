# Data Collection Protocol — Stage 0 / Stage 4

Run **all 6 scenarios** in a single session if possible (~70 min total).
Press `G` (start ToF log) and `9` (start arm log in plotter) **before each scenario**.
The RL recorder logs automatically — just have the controller running.

---

## Object for obstacle scenarios

Use a **cylindrical object 6–10 cm diameter** (water bottle, coffee mug, tin can).
This size roughly matches the simulated obstacle sphere (40–120 mm radius).

---

## Step 1 — Save workspace states (do this first, ~5 min)

Open controller: `python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1`

For each state below: navigate with A/D/W/S/Q/E keys to the target pose,
then press **M → "Save current state" → type the name → Enter**.

| State name   | theta (°) | r (mm) | z (mm) | Notes                        |
|--------------|-----------|--------|--------|------------------------------|
| `seq_start`  |  30       |  152   |  60    | Left edge of sweep           |
| `seq_mid`    |  90       |  152   |  60    | Center — use as home base    |
| `seq_end`    | 150       |  152   |  60    | Right edge of sweep          |
| `seq_high`   |  90       |  152   |  95    | Upper Z level                |
| `seq_low`    |  90       |  152   |   0    | Lower Z level                |
| `seq_close`  |  90       |   80   |  60    | Short reach                  |
| `seq_far`    |  90       |  220   |  60    | Extended reach               |
| `corner_tl`  |  30       |  152   |  95    | Top-left workspace corner    |
| `corner_tr`  | 150       |  152   |  95    | Top-right workspace corner   |
| `corner_bl`  |  30       |  152   |   0    | Bottom-left workspace corner |
| `corner_br`  | 150       |  152   |   0    | Bottom-right workspace corner|

> **Navigation tip**: A/D move theta in 5° steps, W/S move r in 10 mm steps,
> Q/E move z in 10 mm steps. Watch the curses display for current values.

---

## Scenario 1 — Free sweep, no obstacles (15 min)

1. Press `G`, then `Z` to start sweep.
2. Walk away. Let it run 15 min completely unattended.
3. Press `Z` again to stop.

**What this captures**: Normal goal-directed sweep behavior — the policy learns
that sweeping produces positive goal-progress reward.

---

## Scenario 2 — Hand obstacle, CH0 side (15 min)

Start sweep (`Z`), then interact:

| Sub-scenario | What to do | Duration |
|---|---|---|
| 2a | Hold open hand, palm toward arm, at theta≈45°, z≈60mm, distance ≈20 cm. Move slowly from 25 cm → 12 cm (causes REPLAN). | 4 min |
| 2b | Same position but at z≈0 mm (low height). Arm should try Z-bypass upward. | 3 min |
| 2c | Same position but at z≈90 mm (high height). Arm should try Z-bypass downward. | 3 min |
| 2d | Move hand slowly along theta axis (from 45° toward 90°) at r≈20 cm while arm sweeps. | 5 min |

**What this captures**: ToF-based replan + Z-axis avoidance. The hand produces
partial-occlusion patterns the full ToF grid encodes, training the policy to
recognise incomplete obstacle signatures.

---

## Scenario 3 — Both-side obstacles (10 min)

Place the **cylindrical object** at theta=90°, r≈150 mm, z≈60 mm.
Start sweep (`Z`), then also hold a second hand on the opposite side of the arm.

| Sub-scenario | What to do |
|---|---|
| 3a | Object center, no hand: let arm try theta bypass. | 3 min |
| 3b | Object + hand on opposite side: arm is blocked both ways, triggers Z bypass. | 4 min |
| 3c | Object at z=30mm (low): arm should go over it. | 3 min |

---

## Scenario 4 — Sequence editor runs (10 min)

Load `collect_seq.txt` into the sequence editor:

1. Press `X` to open sequence editor.
2. Type the contents of `collect_seq.txt` (or paste if terminal allows).
3. Press `F5` or `Ctrl+R` to run. Let it cycle 3–4 times (≈7 min each full cycle).
4. While sequence is running, place the obstacle at theta=60°, r=150mm to
   force avoidance decisions between `seq_start` and `seq_mid`.

**What this captures**: Goal-conditioned motion — transitions where `goal_delta`
feature is non-zero. Critical for teaching the policy goal-directed behavior.

---

## Scenario 5 — Manual joystick exploration (10 min)

Use A/D/W/S/Q/E to drive the arm manually through the workspace.
No sweep running for this scenario.

| Area to explore | Keys |
|---|---|
| Theta limits: 5° and 175° | A/D to extremes |
| r limits: 80 mm and 220 mm | S/W to extremes |
| z limits: 0 mm and 95 mm | E/Q to extremes |
| All 4 workspace corners | Combine theta + z extremes |
| Mid-range with obstacle near | Hold obstacle at 15 cm, move arm toward it |

**What this captures**: Diverse (theta, r, z) coverage — areas the auto-sweep
never visits. Essential for generalisation.

---

## Scenario 6 — Controlled near-misses (10 min)

Place the obstacle **stationary** at theta=90°, r=150mm, z=60mm.
Start sweep (`Z`). The arm will stop at ~250mm threshold.

| Sub-scenario | Obstacle distance from arm path | What happens |
|---|---|---|
| 6a | ≈220 mm — just inside threshold | Soft REPLAN, gentle arc |
| 6b | ≈120 mm — well inside threshold | Hard REPLAN, back-away |
| 6c | ≈80 mm — very close | IR may fire |
| 6d | Move obstacle from 300mm → 80mm slowly over 2 min | Gradual proximity ramp |

**What this captures**: Proximity reward shaping — the soft quadratic penalty
ramp as the arm closes in. These are the most valuable near-collision examples.

---

## After collection

1. Press `ESC` to quit the controller — this flushes NPZ buffers to `logs/`.
2. Verify files exist: `ls -lh logs/rl_transitions_*.npz`
3. You should have ≥3 NPZ files totalling ≥ 20 MB.
4. Run the noise calibration script:
   ```
   cd braccio_main_runner
   python calibrate_noise.py
   ```
5. Optionally run the path annotator on the best sweep session:
   ```
   python path_annotator.py logs/rl_transitions_YYYYMMDD_HHMMSS.npz
   ```

---

## Logging reminder

- `G` key → start/stop **ToF CSV** log (`logs/tof_TIMESTAMP.csv`)
- `9` key → start/stop **arm CSV** log in arm plotter (if plotter window is open)
- RL transitions log automatically — no key needed.
- Run plotter in a separate terminal if you want arm CSV: `python arm_plotter_app.py --port /dev/ttyACM0`
