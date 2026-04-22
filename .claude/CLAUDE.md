# Reactive Robot — Project Context for Claude

## Project Goal

This is an **autonomous obstacle-avoiding robotic arm system** built on the Braccio V2 (6-DOF servo arm). The overall goal is to perform a continuous theta sweep (0°→180° and back) while detecting and avoiding obstacles in real time using multiple sensor modalities.

The arm should:
1. Sweep its base angle back and forth autonomously
2. Detect obstacles via ToF sensors and IR proximity sensors
3. Replan its path (horizontally or vertically) around detected obstacles
4. Remember obstacle positions in a world-frame coordinate system so it doesn't re-enter known-bad zones
5. React quickly (within one loop tick, ~100 ms) to new detections

---

## Hardware

| Component | Details |
|-----------|---------|
| Arm | Braccio V2, 6-DOF (base, shoulder, elbow, wrist-vertical, wrist-rotation, gripper) |
| Arm controller | Arduino Mega (runs existing Braccio firmware) via `/dev/ttyACM0` |
| Sensor controller | Teensy 4.1 via `/dev/ttyACM1` |
| ToF sensors | 4× VL53L5CX on TCA9548A I2C mux (channels 0–3), 8×8 distance grid |
| IR sensors | 4× EC-Buying active-LOW IR obstacle sensors wired to Teensy pins 2–5 |
| IMU | MPU-6050 on Teensy (roll/pitch from accelerometer, no yaw fusion) |

**Sensor mounting:**
- CH0 (front/side): faces the sweep direction — primary obstacle authority
- CH1 (back/side): opposite side — primary obstacle authority
- CH2 (top): faces upward — advisory only, no replanning
- CH3 (bottom): faces ground — completely ignored (floor false positives)

**IR wiring:** All 4 sensors share 5V and GND. Signal pins → Teensy 2, 3, 4, 5. Active LOW with `INPUT_PULLUP`. A count of firing sensors maps to severity: 0=CLEAR, 1=FAR, 2=CLOSE, 3=DANGER.

---

## Codebase Layout

```
braccio_main_runner/
  braccio_ctrl/
    __main__.py          — entry point (argparse, wires all objects together)
                           NEW: --rl-policy PATH switches Z key to RLSweeper
    controller.py        — main loop: keyboard input, obstacle checks, display refresh
                           Builds RLSweeper when --rl-policy is set; otherwise
                           Z drives the safety BT sweep.
    arm_state.py         — thread-safe 6-DOF joint state + IK helpers
    serial_bridge.py     — serial comms to Arduino (SET ALL / SET DELTA / GET POS)
    tof_sensor.py        — ToFBridge serial reader + ToFState (grids, IR, obstacle response)
    imu_state.py         — thread-safe IMU state with cached rotation matrix
    ik_solver.py         — 2-link planar IK (law of cosines), reachability check
    constants.py         — ALL tunable parameters in one place
    display.py           — curses TUI
    protocol.py          — cmd_set_all(), cmd_set_delta() string builders
    state_library.py     — saved named arm poses (states.json)
    data_publisher.py    — UDP publisher for arm/ToF telemetry
    session_logger.py    — JSONL telemetry logger for offline review
    keyboard_handler.py  — curses key → action string mapping

    # ── Safety stack (replaces legacy auto_sweep + obstacle_map) ─────────
    safety/              — BT-driven obstacle-aware motion planning
                           WorldModel + KDTree + capsule collision + BiRRT
                           + cascading replanner + py_trees orchestration.
                           See safety/__init__.py docstring for details.

    # ── RL pipeline (Stage 0 → Stage 5) ──────────────────────────────────
    rl_recorder.py       — daemon that writes (obs, action, reward=0,
                           next_obs, done) to logs/rl_transitions_*.npz
                           at SWEEP_TICK_HZ. Always-on once controller starts.
    rl_env.py            — Gymnasium base env: 74-float obs, 4-float action
                           (Δθ, Δr, Δz, Δdelta), normalisation + denormalise.
    rl_reward.py         — compute_reward(): goal progress, collision, ToF
                           proximity shaping, jerk + holding penalties.
    rl_sweeper.py        — Drop-in AutoSweeper replacement. Wraps a hot-
                           swappable AtomicPolicyRef; runs the RL policy at
                           SWEEP_TICK_HZ; pushes transitions into an
                           OnlineTrainer if attached.
    rl_online_trainer.py — Background SAC fine-tuner. SharedReplayBuffer +
                           KL stability guard + zero-downtime weight swap.
    rl_sequence_runner.py — Goal-conditioned SequenceRunner. Each step calls
                            sweeper.set_goal({theta,r,z}) and waits for
                            goal_reached() or timeout.

    # ── Simulation (PyBullet, used by train_rl.py) ───────────────────────
    sim/
      braccio_env.py        — Analytical BraccioSimEnv with domain
                              randomisation. Single spherical obstacle
                              per episode; ToF via ray-sphere intersection.
                              obs[54:62] and obs[70:74] are zeroed to match
                              the hardware null state (obstacle_map=None).
      noise_params.json     — Calibrated noise params (filled in by
                              calibrate_noise.py from real NPZ files).

  # ── Top-level training scripts ─────────────────────────────────────────
  train_rl.py            — SAC training in BraccioSimEnv. Optional --bc-policy
                           warm-start (currently unused — see Q2 pipeline note
                           below). Writes best_policy/best_model.zip.
  calibrate_noise.py     — One-shot. Reads logs/rl_transitions_*.npz, fits
                           ToF sigma + servo lag + cell dropout, writes to
                           sim/noise_params.json.
  path_annotator.py      — Optional matplotlib tool for drawing ideal paths
                           over a session NPZ → DTW reward overrides.

ToF_RR_Sensing/arduino/vl5_tca_4x4/
  vl5_tca_4x4.ino        — Teensy firmware (ToF + IR + IMU serial output)
```

**Removed in safety-stack refactor (do not re-add):**
`auto_sweep.py`, `obstacle_map.py`, `obstacle_memory.py` — legacy modules
replaced by `safety/`. They were deleted in commit 2a11d00.

---

## Key Constants (constants.py)

```python
# Link lengths
L1, L2, L3 = 125.0, 125.0, 60.0   # mm

# ToF per-channel thresholds [CH0, CH1, CH2, CH3]
TOF_THRESHOLDS_MM = [250.0, 250.0, 50.0, 50.0]

# Sensor authority
SENSOR_REPLAN_CHANNELS   = [0, 1]   # trigger REPLAN
SENSOR_ADVISORY_CHANNELS = [2]      # log only
SENSOR_IGNORE_CHANNELS   = [3]      # skip entirely

# Sweep
SWEEP_Z_DEFAULT           = 60.0    # mm (raised from -50 to clear floor)
SWEEP_Z_CANDIDATES        = [0.0, 20.0, 55.0, 70.0, 95.0]  # Z-axis avoidance levels
SWEEP_COLLISION_RADIUS_MM = 80.0    # pre-command obstacle clearance sphere

# Obstacle memory (voxel grid)
OBS_MEM_CELL_MM           = 40.0    # voxel size
OBS_MEM_INC               = 0.30    # confidence boost on observation
OBS_MEM_DECAY_PER_SEC     = 0.15    # decay rate
OBS_MEM_OCCUPIED_THRESHOLD = 0.20   # voxel considered occupied above this
```

---

## Serial Protocols

**Arduino (arm):**
- Send: `SET ALL B<deg> S<deg> E<deg> WV<deg> WR<deg> G<deg>\n`
- Send: `SET DELTA <n>\n` (1–5, slew rate)
- Receive: `POS B<deg> S<deg> ...` (actual positions)

**Teensy (ToF/IR/IMU):**
- Receive: `TF,<seq>,<ms>,S<ch>,<ch>,<joint>,<status>,<rows>,<cols>,<d0..dN>,<v0..vN>`
- Receive: `IR,<0-3>` (2-bit severity)
- Receive: `IMU,<seq>,<ms>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<temp>,<status>`
- Send: `ACT 4\n` (enable all 4 channels), `MUX\n`, `CH0\n`–`CH3\n`

---

## Coordinate System

- Origin: arm base / shoulder pivot
- +X: arm forward at theta = 0°
- +Y: arm left at theta = 90°
- +Z: upward
- Theta: base rotation angle in degrees (0–180)
- r: radial reach in mm
- z: height above shoulder pivot in mm

---

## Running the Software

```bash
# Full controller (arm + ToF) — safety BT drives the sweep
python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1

# Same, but Z key drives a trained RL policy (Stage 5 deployment)
python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1 \
    --rl-policy best_policy/best_model.zip

# Disable IR sensors (loose wiring / false positives)
python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1 --no-ir

# Arm only (no ToF)
python -m braccio_ctrl /dev/ttyACM0

# ToF plotter only
python braccio_main_runner/tof_plotter_app.py --port /dev/ttyACM1
```

**Key keyboard bindings:**
- `Z` — toggle autonomous sweep (safety BT, or RLSweeper if --rl-policy)
- `C` — calibrate IMU yaw reference
- `A/D` — theta (base rotation)
- `W/S` — r (reach)
- `Q/E` — z (height)
- `H` — go home
- `M` — saved states menu
- `X` — sequence editor
- `T` — toggle strict mode (refusal dialog on collision)
- `G` — start/stop ToF CSV log
- `ESC` — quit (also flushes RL recorder NPZ to logs/)

---

## Architecture Notes

- All shared state objects use `threading.RLock()` for thread safety
- The sweep daemon thread runs at `SWEEP_TICK_HZ` (20 Hz)
- The safety BT (`safety/behavior.py`) replans every tick via the
  cascading replanner: polar-skip → Z-ladder → BiRRT → direction-flip
- IMU rotation matrix is cached; invalidated only when roll/pitch/yaw changes
- Logs go to `logs/controller.log` because curses owns the terminal — tail
  it post-session for any stack traces from background threads
- The RLRecorder is always-on while the controller runs and writes
  `logs/rl_transitions_*.npz` at shutdown (or every 50 MB)

---

## RL Pipeline (Q2: SAC-from-sim + optional online fine-tuning)

```
┌─ Stage 0 ──────────────────────────────────────────────────────────────────┐
│ Hardware data collection (60–90 min). Run controller normally,             │
│ vary obstacles by hand. RLRecorder writes logs/rl_transitions_*.npz.       │
│ The safety BT replans naturally — that is desired training data.           │
│ See DATA_COLLECTION_PROTOCOL.md.                                           │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ Stage 1 ──────────────────────────────────────────────────────────────────┐
│ python calibrate_noise.py                                                  │
│ Fits ToF sigma, servo lag, cell dropout from NPZ files.                    │
│ Writes braccio_ctrl/sim/noise_params.json.                                 │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ Stage 2 ──────────────────────────────────────────────────────────────────┐
│ python train_rl.py                                                         │
│ SAC trains in BraccioSimEnv with domain-randomised noise from Stage 1.     │
│ ~1M steps, 4–6 h on a laptop GPU.                                          │
│ Writes best_policy/best_model.zip.                                         │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─ Stage 3 (deploy) ─────────────────────────────────────────────────────────┐
│ python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1 \           │
│       --rl-policy best_policy/best_model.zip                               │
│ Z key now starts/stops the RLSweeper instead of the safety BT sweep.       │
└────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ (optional)
┌─ Stage 4 ──────────────────────────────────────────────────────────────────┐
│ Online fine-tuning via rl_online_trainer.py.                               │
│ Hot-swaps policy weights on the live RLSweeper without stopping the arm.   │
└────────────────────────────────────────────────────────────────────────────┘
```

**BC warm-start is intentionally disabled.** `rl_recorder.set_goal()` is
never called during hardware collection (the safety BT does not expose its
internal target waypoint), so the goal-delta labels in collected NPZ files
are all zero — useless for behavioral cloning.

---

## Observation Vector (74 floats — RL pipeline contract)

Layout produced by `rl_recorder._encode_obs()` and `BraccioSimEnv._get_obs()`.
Both must match exactly or the policy won't transfer.

| Slice    | Width | Meaning |
|----------|-------|---------|
| [0:5]    |  5    | arm pose: theta_n, r_n, z_n, dir_to_goal, delta_n |
| [5:21]   | 16    | CH0 ToF grid 4×4 flattened, NaN→1.0, /250 mm |
| [21:37]  | 16    | CH1 ToF grid same normalisation |
| [37:53]  | 16    | CH2 ToF grid (advisory), /50 mm |
| [53]     |  1    | IR severity (ir_bits / 3) |
| [54:62]  |  8    | obstacle map summary — **always [0,1,0,0,0,0,0,0]** on hardware |
| [62:67]  |  5    | Z-level reachability mask for SWEEP_Z_CANDIDATES |
| [67:70]  |  3    | goal delta (gt-θ)/90, (gr-r)/115, (gz-z)/125 |
| [70:74]  |  4    | forbidden theta band — **always [0,0,0,0]** on hardware |

The trailing `obs[54:62]` and `obs[70:74]` blocks are zeroed in sim
(`braccio_env.py`) to match the hardware null state, so the policy is
forced to learn obstacle avoidance from the ToF grids, not from features
that don't exist at deploy time.

**Action vector (4 floats):** `[Δθ°, Δr_mm, Δz_mm, Δdelta]` raw; the RL env
normalises/denormalises around the THETA_STEP / R_STEP / Z_STEP magnitudes.
