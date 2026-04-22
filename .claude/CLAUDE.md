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
    __main__.py          — entry point (argparse, logging.basicConfig to
                           logs/controller.log, wires all objects together)
                           NEW: --rl-policy PATH switches Z key to RLSweeper
    controller.py        — main loop: keyboard input, obstacle checks, display refresh
                           Builds RLSweeper when --rl-policy is set; otherwise
                           Z drives the safety BT sweep.
    arm_state.py         — thread-safe 6-DOF joint state + IK polar shadow
    serial_bridge.py     — serial comms to Arduino (SET ALL / SET DELTA / GET POS)
    tof_sensor.py        — ToFBridge serial reader + ToFState (grids, IR, obstacle response)
                           _parse_imu() populates imu_state from IMU,... lines
    imu_state.py         — thread-safe IMU state with cached rotation matrix
    ik_solver.py         — CGx-InverseK port: solve_ik / fk_polar (CGx internal
                           convention) + fk_tip_physical (physical tip position)
    constants.py         — ALL tunable parameters in one place
    display.py           — curses TUI — shows both commanded and physical tip z
    protocol.py          — cmd_set_all(), cmd_set_delta(), cmd_set_joint() builders
    state_library.py     — saved named arm poses (states.json)
    data_publisher.py    — UDP publisher for arm/ToF telemetry
    session_logger.py    — JSONL telemetry logger; logs IMU + physical tip now
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
                           Wired into controller.py lifecycle with surfaced
                           error reporting on flush failure.
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
                              IMPORTANT: uses ANALYTICAL tip position from
                              (theta, r, z) — PyBullet is only for visualisation.
                              The URDF axis mismatches don't affect training.
      noise_params.json     — Calibrated noise params (filled in by
                              calibrate_noise.py from real NPZ files).

    assets/braccio_description/
      urdf/braccio_model.urdf — For PyBullet GUI visualisation only.
      meshes/…                — CAD mesh files (collision + visual).

  # ── Top-level training scripts ─────────────────────────────────────────
  train_rl.py            — SAC training in BraccioSimEnv. Optional --bc-policy
                           warm-start (currently unused — see Q2 pipeline note
                           below). Writes best_policy/best_model.zip.
  train_bc.py            — Behavioural-cloning pretraining head. Optional;
                           BC warm-start currently disabled (see notes).
  calibrate_noise.py     — One-shot. Reads logs/rl_transitions_*.npz, fits
                           ToF sigma + servo lag + cell dropout, writes to
                           sim/noise_params.json.
  path_annotator.py      — Optional matplotlib tool for drawing ideal paths
                           over a session NPZ → DTW reward overrides.

  # ── Data collection tools (Stage 0) ────────────────────────────────────
  run_scenario_2.py      — Automated Scenario 2 (palm-obstacle) data collection.
                           Cycles the arm through saved poses while you do the
                           palm motions. Writes logs/rl_transitions_*.npz.
  validate_fk.py         — Drives the arm through a grid of known joint sets
                           and asks the user for ruler measurements, then
                           writes logs/fk_calibration_*.json for offline fitting.
  probe_joint_conventions.py — Debug tool: moves each joint through its
                           extremes so you can empirically verify the servo
                           convention (which is how we discovered B is
                           physically CCW-flipped, E bends 90° forward at
                           180, WV bends 90° forward at 180).
  collect_seq.txt        — Sequence-editor script for Scenario 4 (11 saved
                           waypoints × 6 cycles). See Data_Collection_Guide.pdf.

ToF_RR_Sensing/arduino/vl5_tca_4x4/
  vl5_tca_4x4.ino        — Teensy firmware (ToF + IR + IMU serial output at
                           100 Hz for IMU)
braccio_joint_test.ino   — Arduino Mega firmware (Braccio V2 servo driver,
                           SET ALL / SET DELTA / GET POS / solveIK protocol)
```

**Removed in safety-stack refactor (do not re-add):**
`auto_sweep.py`, `obstacle_map.py`, `obstacle_memory.py` — legacy modules
replaced by `safety/`. They were deleted in commit 2a11d00.

---

## Key Constants (constants.py)

```python
# Link lengths (standard Braccio V2 kinematics)
L1, L2, L3 = 125.0, 125.0, 60.0   # mm

# Polar defaults — chosen to match physical HOME pose
DEFAULT_THETA = 90.0    # base pointing +Y
DEFAULT_R     = 0.0     # tip directly over base axis
DEFAULT_Z     = 310.0   # = L1 + L2 + L3 (arm fully vertical)
DEFAULT_WRIST_OFFSET = -90.0  # gripper vertical up at startup

# Workspace bounds (manual-control clamps)
R_MIN, R_MAX = 0.0, 310.0      # radial reach
Z_MIN, Z_MAX = -250.0, 310.0   # height above/below shoulder pivot

# ToF per-channel thresholds [CH0, CH1, CH2, CH3]
TOF_THRESHOLDS_MM = [250.0, 250.0, 50.0, 50.0]

# Sensor authority
SENSOR_REPLAN_CHANNELS   = [0, 1]   # trigger REPLAN
SENSOR_ADVISORY_CHANNELS = [2]      # log only
SENSOR_IGNORE_CHANNELS   = [3]      # skip entirely

# Sweep
SWEEP_Z_DEFAULT           = 60.0
SWEEP_Z_CANDIDATES        = [0.0, 20.0, 55.0, 70.0, 95.0]
SWEEP_COLLISION_RADIUS_MM = 80.0

# Obstacle memory (voxel grid)
OBS_MEM_CELL_MM           = 40.0
OBS_MEM_INC               = 0.30
OBS_MEM_DECAY_PER_SEC     = 0.15
OBS_MEM_OCCUPIED_THRESHOLD = 0.20

# Absolute log paths (anchored at braccio_main_runner/, not cwd-relative)
LOG_DIR = <braccio_main_runner>/logs
SESSION_LOG_DIR = <braccio_main_runner>/session_logs
```

---

## Serial Protocols

**Arduino (arm):**
- Send: `SET ALL B<deg> S<deg> E<deg> WV<deg> WR<deg> G<deg>\n`
- Send: `SET DELTA <n>\n` (1–5, slew rate)
- Send: `SET <TOKEN> <deg>\n` (single joint; TOKEN ∈ {B,S,E,WV,WR,G})
- Receive: `POS B<deg> S<deg> ...` (actual positions)

**Teensy (ToF/IR/IMU):**
- Receive: `TF,<seq>,<ms>,S<ch>,<ch>,<joint>,<status>,<rows>,<cols>,<d0..dN>,<v0..vN>`
- Receive: `IR,<0-3>` (2-bit severity)
- Receive: `IMU,<seq>,<ms>,<ax_g>,<ay_g>,<az_g>,<gx_dps>,<gy_dps>,<gz_dps>,<temp>,<status>` at 100 Hz
- Send: `ACT 4\n` (enable all 4 channels), `MUX\n`, `CH0\n`–`CH3\n`

---

## Joint Convention (CRITICAL — empirically probed)

**Physical servo convention** (confirmed via `probe_joint_conventions.py`):
- `B=45` rotates base **counter-clockwise** from operator's view → IK emits `B_cmd = 180 − θ_world`
- `S=45` tilts upper arm forward (toward +r)
- `E=180` bends forearm **90° forward** from aligned (aligned at E=90)
- `WV=180` bends gripper **90° forward** from aligned (aligned at WV=90)
- `HOME_POS = [90, 90, 90, 90, 90, 73]` → arm fully vertical, gripper up

**IK convention** — we ported [cgxeiji/CGx-InverseK](https://github.com/cgxeiji/CGx-InverseK). The IK uses an internal frame rotated by −π/2 (so internal 0 = Braccio 90° = arm up). Elbow-down branch is preferred (shoulder forward, elbow outward); elbow-up is the fallback. The `_solve_planar_free_phi` helper sweeps outward from the caller's requested phi if the exact angle is unreachable.

**Important discrepancy** (self-consistent but worth knowing): The CGx IK convention matches the sim but not the physical arm's elbow-sign convention. The *same joint commands* executed on the hardware land the tip at a different position than CGx's FK predicts. This means **state.r and state.z are IK parameters, not physical tip coordinates**. The physical tip is recovered via `ik_solver.fk_tip_physical(joints)`. Session logger logs both.

**Consequence for training**: sim and hardware both operate in the same "IK parameter" space (both use `solve_ik` from the same module), so policy training is self-consistent. The state.z ≠ physical z discrepancy only matters if you need absolute physical positions (e.g., tabletop manipulation). Obstacle-avoidance sweeping works fine because ToF/IR readings are physical.

---

## Coordinate System

- Origin: arm base / shoulder pivot
- +X: arm forward at theta = 0°
- +Y: arm left at theta = 90°
- +Z: upward
- Theta: base rotation angle in degrees (0–180)
- r: radial reach in mm (IK parameter, not guaranteed physical)
- z: height above shoulder pivot in mm (IK parameter)
- Physical tip: use `fk_tip_physical(joints)` for true world coordinates

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

# Automated Scenario 2 data collection
python braccio_main_runner/run_scenario_2.py /dev/ttyACM0 \
    --teensy-port /dev/ttyACM1 --no-ir

# Probe physical joint conventions (sanity check on servo mounting)
python braccio_main_runner/probe_joint_conventions.py /dev/ttyACM0

# Calibrate FK against ruler measurements
python braccio_main_runner/validate_fk.py /dev/ttyACM0 \
    --teensy-port /dev/ttyACM1 --auto-probe --pivot-mm 76
```

**Key keyboard bindings:**
- `Z` — toggle autonomous sweep (safety BT, or RLSweeper if --rl-policy)
- `C` — calibrate IMU yaw reference
- `A/D` — theta (base rotation)
- `W/S` — r (reach)
- `Q/E` — z (height)
- `I/K` — wrist-vertical tilt (DIRECT single-joint command — does NOT run IK;
         rotates only the wrist servo, keeps shoulder/elbow fixed)
- `J/L` — wrist rotation (direct)
- `O/P` — gripper open/close (direct)
- `H` — go home (commands HOME_POS directly if equilibrium matches defaults;
         avoids the old IK-for-home bug that bent the wrist 90° forward)
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
- Logging: `__main__._configure_logging` sets up `logs/controller.log` via
  `basicConfig(force=True)` because curses owns the terminal
- The RLRecorder is always-on while the controller runs and writes
  `logs/rl_transitions_*.npz` at shutdown (or every 50 MB)
- `arm_state.polar_synced = True` by default (disables fk_polar writeback
  on first POS — that sync caused polar drift when IK quantised integer
  servo angles round-tripped through FK)
- `_send_safety_command` does NOT write `fk_polar(joints)` into polar
  state; state.θ/r/z remain user-authoritative (intent), not a physical
  readback. Physical tip is derived on demand via `fk_tip_physical`.

---

## Stage 0 Data Collection

`DATA_COLLECTION_PROTOCOL.md` and `Data_Collection_Guide.pdf` cover the full
workflow. Summary:

1. Launch controller with `--no-ir` if IR is flaky.
2. RLRecorder is auto-started. NPZ files go to `logs/rl_transitions_*.npz`.
3. For each scenario:
   - **S1 (free sweep)**: press Z, walk away 15 min.
   - **S2 (hand obstacle)**: run `run_scenario_2.py` which cycles through
     saved poses while you do palm motions.
   - **S3 (both-side obstacles)**: cylinder + hand, start sweep.
   - **S4 (sequence-editor run)**: load `collect_seq.txt` via X key, let
     it run 6 cycles (~9 min). Place cylinder at θ≈60° after cycle 1.
   - **S5 (manual joystick)**: manual A/D/W/S/Q/E to workspace edges.
   - **S6 (controlled near-misses)**: cylinder at decreasing distances.
4. Press ESC to flush NPZ.
5. Run `python calibrate_noise.py` → writes `sim/noise_params.json`.

Session logs also write JSONL telemetry to `session_logs/session_*.jsonl`
including IMU roll/pitch/accel, ToF grids, safety-stack state, and
**both** commanded polar AND physical tip position (`phys_tip_r/z`).

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
never called during free-sweep hardware collection (the safety BT does not
expose its internal target waypoint), so the goal-delta labels in
collected NPZ files are all zero — useless for behavioral cloning.

**Sim ↔ real consistency (verified in session audits):**
- URDF joint limits match firmware `JOINT_MIN/MAX` exactly.
- `BraccioSimEnv` uses `solve_ik` from the same `ik_solver` module as the
  hardware, so joint commands are identical.
- Sim physics are **analytical** — tip position is derived from
  `(theta, r, z)` directly, not from PyBullet's URDF FK. URDF axis
  directions thus don't affect training.
- Obs layout is identical between `_encode_obs` (recorder/sweeper) and
  `_get_obs` (sim). Slice indices verified in `rl_reward.py`.
- Action denormalisation is the same function (`BraccioBaseEnv.denormalize_action`).

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

---

## Session History (April 2026 IK+lifecycle refactor)

Multi-session effort to stabilize Stage 0 data collection and prepare for
one-shot RL training. Key changes committed over the session:

1. **NPZ recorder hardened.** Auto-wired into `controller.run()` and
   `_graceful_shutdown()`. Tick-error cap (`_MAX_CONSECUTIVE_ERRORS=10`).
   Flush failures raise instead of silently losing the session. `LOG_DIR`
   absolutized so cwd drift can't misplace NPZ files. Zero-shortcircuit
   bug in action-delta computation fixed (`prev_z or z` was dropping
   deltas off zero-z poses). Status surface (`last_save_path`,
   `tick_fatal`) printed at shutdown.

2. **Logging configured.** `__main__._configure_logging()` sets up
   `logs/controller.log` via `basicConfig(force=True)`. Previously, every
   `log.info/exception` went to `/dev/null` because no root handler was
   attached.

3. **Absolute log paths.** `LOG_DIR`, `SESSION_LOG_DIR`, `SCREENSHOT_DIR`
   in `constants.py` are now anchored at `_PROJECT_ROOT` rather than cwd.

4. **IK convention fixed.** Ported `cgxeiji/CGx-InverseK` verbatim to
   Python. The `_solve_planar` picks elbow-down with elbow-up fallback;
   `_solve_planar_free_phi` sweeps outward from the caller's requested
   phi (not always vertical). Base flip `B_cmd = 180 − θ_world` verified
   against physical probe (B=45 rotates CCW). Added
   `fk_tip_physical(joints)` for accurate tip-position computation under
   the physical elbow/wrist convention, alongside the CGx `fk_polar`
   which remains the inverse of `solve_ik`.

5. **HOME defaults match physical pose.** `DEFAULT_R = 0`,
   `DEFAULT_Z = 310`, `DEFAULT_WRIST_OFFSET = -90`. Previously
   `DEFAULT_R/Z = 152/-50` caused a 391 mm "crazy drop" on the first
   keypress because IK would snap to a far target. `Z_MAX = 310`
   (was 200) so `z_inc` from default no longer silently clips state.z
   down by 50 mm.

6. **`_send_home` fast-path.** When equilibrium polar matches defaults,
   `H` commands `HOME_POS` byte-for-byte instead of re-running IK; this
   guarantees the arm returns to the exact physical rest pose (gripper
   vertical up) rather than the IK's auto-level interpretation.

7. **Wrist decoupled from full IK.** Action handler routes I/K to
   `_send_wrist_vert`, a direct single-joint command (mirrors
   `_send_wrist_rot` for J/L). `state.wrist_offset` is re-derived from
   the new joints via `gripper_world = S + E + WV − 180` so a later
   A/D/W/S/Q/E preserves the just-set gripper pitch through IK.
   Eliminated the "press I and the whole arm moves" class of bugs.

8. **State display shows physical tip.** `display._draw_ik_state`
   renders both commanded `(r, z)` and physical `(r, z)` derived via
   `fk_tip_physical`. Session logger records both `phys_tip_r` and
   `phys_tip_z` per sample, so post-session analysis can reconstruct
   real tip positions.

9. **IMU logged to session files.** `session_logger.set_sources` now
   accepts `imu_state`; each sample gets `{roll_deg, pitch_deg,
   yaw_deg, ax, ay, az, calibrated, last_rx}`. Teensy streams IMU at
   100 Hz but prior sessions had zero IMU data in logs.

10. **IK-failure rollback.** `_send_ik_move(prev_polar=…)` — when IK
    rejects the new target, `state.theta/r/z` is rolled back to the
    pre-keypress value so the user sees "no motion + UNREACHABLE" rather
    than silently drifting state into unreachable coordinates.

11. **Empirical FK calibration tools.** `validate_fk.py`
    drives the arm through an 8-pose auto-probe grid and accepts
    ruler-measured tip heights, writing
    `logs/fk_calibration_*.json`. The first fit attempt on the initial
    4 poses had 77 mm RMS residual (hardware-vs-model
    discrepancy); more data + metric tape will tighten it.

12. **Scenario 2 automation.** `run_scenario_2.py` cycles saved poses
    for scripted palm-obstacle data collection without manual keypress
    choreography. Supports `--auto-probe`, `--dry-run`, `--list-ports`,
    `--no-ir`.

13. **`collect_seq.txt` created.** 11-waypoint × 6-cycle sequence for
    Data Collection Guide Scenario 4, matching the seq_*/corner_* saved
    states. Parses cleanly via `dsl/parser.parse_auto` (66 total MOVE
    executions, all references resolve).

14. **DSL accepts lowercase state names.** Patched `_RE_MOVE`,
    `_RE_UPPER_IDENT`, and the Lark `IDENT` token to allow
    `[A-Za-z_][A-Za-z0-9_]*` — saved states can mix case now.

### Known residual issues (audit, still open)

- **Dead `wrist_offset` parameter** in `arm_state.update_joints_from_ik`
  (line 92) — `del wrist_offset` inside the function. Callers still pass
  it. Cosmetic; low priority.
- **Unused constants** in `constants.py`:
  `SIMPLE_SWEEP_MODE`, `COLLISION_CHECK_RADIUS_MM`,
  `SENSOR_PRIMARY_CHANNELS` — zero external references.
  `TOF_THRESHOLD_MM` has exactly one external ref (in session_logger),
  so it's borderline.
- **`RLRecorder.set_goal` / `.set_sweeper`** — defined but never called
  (confirmed via grep). `RLSweeper.set_goal` IS used (by
  `rl_sequence_runner`). The RLRecorder methods appear to be a
  leftover from an earlier goal-conditioned recording design.
- **Sim noise-params silent fallback**: `_load_noise_params` swallows
  FileNotFoundError / JSONDecodeError and returns defaults without
  warning. If `calibrate_noise.py` wasn't run, training uses synthetic
  defaults and the user is unaware. Worth adding a warning.
- **URDF joint axis directions** don't match the physical Braccio's
  probed convention, but this is inconsequential for training because
  the sim's physics are analytical (tip derived from θ/r/z, not from
  PyBullet's URDF FK). Only the GUI visualisation would render "wrong."

### For next-session continuity

- The IK `fk_polar` returns CGx-convention coordinates; `fk_tip_physical`
  returns the physical tip. Know which you want before calling.
- If the RL policy misbehaves at deployment, first check
  `logs/controller.log` for RLSweeper's "policy predict error" lines
  (added in session). A crashed policy silently returns zeros per
  `AtomicPolicyRef.predict`, which looks like the arm "stalling."
- The physical servo convention was probed once (see item 4). If the
  arm is ever re-assembled or the servo horns are re-indexed, re-run
  `probe_joint_conventions.py` to confirm the convention still holds.
