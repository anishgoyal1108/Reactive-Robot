# Session Memory — Reactive Robot

## What Was Built / Changed in This Session

### 1. Per-Channel ToF Thresholds

**Problem:** All 4 sensors used a single threshold, causing CH3 (bottom, near the floor) to constantly fire false REPLAN alerts with no actual obstacle.

**Solution:** Replaced `TOF_THRESHOLD_MM` (single float) with `TOF_THRESHOLDS_MM = [250.0, 250.0, 50.0, 50.0]` indexed by channel. Added sensor authority classification:
- `SENSOR_REPLAN_CHANNELS = [0, 1]` — CH0/CH1 (sides) trigger REPLAN
- `SENSOR_ADVISORY_CHANNELS = [2]` — CH2 (top) logged only, no arm reaction
- `SENSOR_IGNORE_CHANNELS = [3]` — CH3 (bottom) skipped entirely

Changed files: `constants.py`, `tof_sensor.py`, `obstacle_map.py`, `controller.py`

---

### 2. Persistent Obstacle Memory (`obstacle_memory.py`)

**Problem:** The rolling point cloud (2 s age cap) forgot obstacles the moment they left sensor FOV. The arm would swing back into recently-seen obstacles.

**Solution:** Created `PersistentObstacleMemory` — a sparse 3D voxel confidence grid:
- Voxel key: `(ix, iy, iz)` = floor(xyz / OBS_MEM_CELL_MM)
- Observation boosts confidence by `OBS_MEM_INC = 0.30`
- Decayed by `OBS_MEM_DECAY_PER_SEC = 0.15` per second via `tick_decay(dt)`
- Voxels pruned below `OBS_MEM_KEEP_THRESHOLD = 0.05`
- Considered occupied above `OBS_MEM_OCCUPIED_THRESHOLD = 0.20`
- Budget-capped at `OBS_MEM_MAX_CELLS = 2000` (evicts lowest-confidence first)

**API:**
```python
memory.ingest(points_np)          # mark observed points (boosts confidence)
memory.tick_decay(dt_s)            # decay + prune (called every update)
memory.query_radius(pos_xyz, r_mm) # O(k) bool — is this sphere occupied?
memory.occupied_thetas(r_mm, z_mm) # list of theta values for replanning
memory.snapshot_stats()            # dict for display
memory.clear()                     # call on IMU recalibration
```

Integrated into `obstacle_map.py`: memory is decayed and ingested on every `update()` call. Cleared when `clear_memory()` is called (triggered by IMU recalibration in `controller.py`).

---

### 3. Z-Axis Replanning (`auto_sweep.py`)

**Problem:** When both side sensors (CH0 + CH1) were blocked, the arm only tried horizontal (theta) bypass. It never attempted going over or under the obstacle.

**Solution:** Added `_find_clear_z(theta)` which iterates `SWEEP_Z_CANDIDATES = [0.0, 20.0, 55.0, 70.0, 95.0]` (above current Z first, then below), checking `reachability()` and `_is_position_clear()` for each. If a clear Z is found, `sweep_z` is updated and the arm moves there.

`_do_replan()` now has two steps tried every tick:
1. Theta bypass (horizontal arc around obstacle)
2. Z-axis avoidance (go over or under)
3. Hold position if both fail (rare — obstacle spans all options)

---

### 4. Fixed Replanning Speed / Timeout

**Problem:** `_do_replan()` had a 10-second early-return that blocked Z-axis attempts: `if waited < 10.0: return`. This meant Z replanning was never tried for the first 10 seconds.

**Fix:** Removed the timeout entirely. Both theta bypass and Z bypass are retried on every tick (~100 ms). The arm reacts within one loop cycle.

---

### 5. Pre-Command Collision Check (`auto_sweep.py`)

Added `_is_position_clear(theta, z)` which:
1. Fast path: checks `obstacle_map.memory_occupied_near()` (O(k) voxel lookup)
2. Fallback: checks rolling cloud via `obstacle_map.all_points()` (O(N))

Called in `_send_move()` before any SET ALL command. Returns `False` (blocks the move) if the end-effector sphere would intersect a known obstacle.

---

### 6. IMU Rotation Matrix Caching (`imu_state.py`)

**Problem:** `rotation_matrix()` recomputed 6 trig calls every main-loop tick even when the IMU hadn't moved.

**Fix:** Added `_cached_R`, `_cached_roll`, `_cached_pitch`, `_cached_yaw` fields. Cache is invalidated when any angle changes. The matrix is computed outside the lock (pure math) then stored under the lock.

---

### 7. Four IR Sensors — Firmware Update

**Problem:** Original firmware used 2 digital pins (bits) for IR. EC-Buying active-LOW sensors were wired to pins 2–5 on the Teensy.

**Fix:** Updated `vl5_tca_4x4.ino`:
```cpp
const int IR_PINS[] = {2, 3, 4, 5};
const int IR_NUM    = 4;
// setup: INPUT_PULLUP for all 4
// readAndSendIR(): count LOW pins → severity
// 0 fired → ir_val=0 (CLEAR), 1→1 (FAR), 2→2 (CLOSE), 3+→3 (DANGER)
```

The 2-bit encoding on the software side (`tof_sensor.py`, `constants.py`) was unchanged.

---

### 8. Sweep Z Default Raised

Changed `SWEEP_Z_DEFAULT` from `-50.0` mm to `60.0` mm. The old value kept the arm near the ground, causing constant CH3 interference. At 60 mm the arm has comfortable floor clearance.

---

## Key Bug Fixes

| Bug | Fix |
|-----|-----|
| `NameError: TOF_THRESHOLD_MM` in `controller.py` | Removed stale `self._tof_state.tof_threshold_mm = TOF_THRESHOLD_MM` line (field initialized in `ToFState.__init__`) |
| Duplicate `self._memory = PersistentObstacleMemory()` in `obstacle_map.py` | Removed second assignment (merge artifact) |
| Duplicate `return result.copy()` block in `imu_state.py` | Removed dead unreachable duplicate |
| Stray `return None` after `_compute_bypass_theta()` in `auto_sweep.py` | Removed orphaned line outside function scope |
| `SENSOR_PRIMARY_CHANNELS` NameError in `tof_sensor.py` | Renamed to `SENSOR_REPLAN_CHANNELS`; `SENSOR_PRIMARY_CHANNELS` kept as alias in `constants.py` |

---

## Files Changed

| File | Summary of changes |
|------|--------------------|
| `constants.py` | `TOF_THRESHOLDS_MM` list; sensor authority lists; `SWEEP_Z_DEFAULT=60`; `SWEEP_Z_CANDIDATES`; `SWEEP_COLLISION_RADIUS_MM`; `OBS_MEM_*` constants |
| `tof_sensor.py` | Per-channel threshold logic; `SENSOR_REPLAN_CHANNELS` authority; keep legacy `tof_threshold_mm` field |
| `obstacle_memory.py` | New file — `PersistentObstacleMemory` voxel grid |
| `obstacle_map.py` | Integrates `PersistentObstacleMemory`; per-channel thresholds in projection; skip `SENSOR_IGNORE_CHANNELS` |
| `auto_sweep.py` | `_is_position_clear()`; `_find_clear_z()`; `_send_move()` pre-check; `_do_replan()` Z fallback; removed 10s timeout |
| `imu_state.py` | Rotation matrix cache (`_cached_R`, `_cached_roll`, `_cached_pitch`, `_cached_yaw`) |
| `controller.py` | Updated imports; removed stale threshold assignment; `clear_memory()` on IMU recalibration; `ObstacleMap(thresholds_mm=...)` |
| `display.py` | Per-channel threshold display; memory stats panel |
| `vl5_tca_4x4.ino` | 4-sensor IR count-to-severity firmware |

---

## Branch State

- Development branch: `claude/robotic-arm-sensor-plan-60YZ6`
- Last commit: `8241380` — "Merge origin/main: per-sensor thresholds, persistent obstacle memory, Z-axis replanning"
- Status: ahead of `main` by 3 commits; PR not yet merged
- To delete branch after merge: `git branch -d claude/robotic-arm-sensor-plan-60YZ6 && git push origin --delete claude/robotic-arm-sensor-plan-60YZ6`

---

## Known Remaining Work / Ideas

- Hardware test: verify CH3 no longer triggers false REPLAN at floor
- Hardware test: cover CH0+CH1 simultaneously to verify Z-axis replanning fires
- Tune `OBS_MEM_DECAY_PER_SEC` — 0.15/s means a confident voxel (1.0) takes ~7 s to fully decay; may want faster/slower depending on environment dynamics
- Tune `SWEEP_Z_CANDIDATES` — current values `[0.0, 20.0, 55.0, 70.0, 95.0]` are derived from saved states; update if `states.json` changes
- Consider magnetometer fusion for yaw (currently yaw = 0 from MPU-6050 accelerometer-only)
- `SENSOR_ADVISORY_CHANNELS` (CH2 top) detection is logged but not yet surfaced in replanning — could be used to lower Z when the arm is about to hit something above it

---

## RL System Plan (Next Major Feature)

A full RL plan has been designed to replace `AutoSweeper` with a learned policy. The complete plan is in `/root/.claude/plans/buzzing-booping-metcalfe.md`. Summary:

### Why RL

The current state machine is brittle: fixed margins, 5 discrete Z levels, single sweep radius. It cannot generalize to novel obstacle positions, moving obstacles, or goal-directed motion from the sequence editor. An RL policy encodes all of this in a single ~85K-parameter MLP whose inference costs ~0.1 ms on CPU — cheaper than the current Kalman + voxel lookup pipeline.

### Observation Space (73 floats)

| Group | Features | Notes |
|-------|----------|-------|
| Arm pose | 5 | theta, r, z, direction, delta — normalized |
| CH0 grid | 16 | Full 4×4 ToF grid / 250.0 (NaN → 1.0) |
| CH1 grid | 16 | Same for back sensor |
| CH2 grid | 16 | Advisory channel / 50.0 |
| IR | 1 | ir_bits / 3.0 |
| Obstacle map | 8 | has_active, age, centroid xyz, Kalman xyz |
| Z-level mask | 5 | Which SWEEP_Z_CANDIDATES are clear at current theta |
| Goal delta | 3 | (goal − current) for theta, r, z — normalized |
| Forbidden band | 4 | min, max, span of blocked thetas + memory_ahead flag |

Full grids (not just `nanmin`) are passed so the network learns partial-occlusion patterns — real sensors often see only a cluster of cells from a hand, not the whole obstacle.

### Reward Function (8 terms)

1. **Goal progress** `+0.1 × (dist_before − dist_after)` — reward closing distance to waypoint
2. **Waypoint reached** `+2.0` — bonus when within tolerance
3. **IR collision** `−5.0` DANGER / `−2.0` CLOSE / `−3.0` BACK_AWAY
4. **Proximity shaping** `−0.5 × (1 − dist/thr)²` per primary channel — soft quadratic ramp
5. **Velocity cost** `−0.02 × (Δtheta² + Δz²)` — penalizes fast motion always
6. **Proximity speed limit** `−1.0 × excess_speed` when closer than 150 mm — arm slows near obstacles
7. **Jerk penalty** `−0.02 × |Δaction_theta|` — smooth trajectory changes
8. **Holding penalty** `−0.05` per tick when action ≈ 0 but goal is far

### Training Pipeline

1. **Data collection** — 60–90 min hardware runs across 6 scenarios; new `rl_recorder.py` saves `(obs, action, reward=0, next_obs, done)` NPZ at 10 Hz alongside existing CSV logs
2. **Behavioral cloning** — pre-train policy on clean demonstrations (~100 epochs, 1 hr compute); warm-starts SAC so it doesn't explore randomly
3. **Noise calibration** — `calibrate_noise.py` fits servo lag + ToF noise + cell dropout rate from session log statistics; writes `sim/noise_params.json`
4. **SAC sim training** — PyBullet env with domain randomization; 1M timesteps; ~4–6 hrs on a laptop GPU
5. **Hardware fine-tune** — constrained exploration (50% action clip) at LR=3e-5; SafeHardwareEnv hard-overrides policy on IR DANGER

### Path Annotator Tool (`path_annotator.py`)

Standalone matplotlib app for annotating session logs with ideal paths:
- **Canvas**: theta (x) vs z (y) — the arm's primary motion plane; r encoded as color on actual-path line
- **Waypoint mode** (default): click to place control points; live Catmull-Rom spline (chord-length parameterized, inherently smooth)
- **Freehand mode**: drag mouse; on release → Savitzky-Golay filter (window=11, polyorder=3) → Douglas-Peucker simplification → Catmull-Rom spline
- **r annotation**: secondary subplot (r vs theta) with draggable spline; collapses to flat line if r is constant
- **DTW reward**: `fastdtw` computes warping path between actual and drawn trajectories; per-timestep reward proportional to distance from ideal path (range [−2, 0])
- **Export**: updates NPZ `rewards` array in-place; saves `_annotated.npz` copy

### Online Retraining

Three concurrent threads during live operation:
- **Hardware thread**: collects transitions at 10 Hz, pushes to `SharedReplayBuffer`
- **Feedback thread**: `+` key adds +0.5 / `−` key adds −0.5 to last 20 transitions' rewards
- **Training thread**: 200 gradient steps every 500 new transitions; KL divergence guard prevents destabilization; zero-downtime policy hot-swap via `AtomicPolicyRef` every 5000 steps

### Sequence Editor Integration

`SequenceRLRunner` feeds each named state from `states.json` as goal into the observation (`goal_dtheta`, `goal_dr`, `goal_dz` features). Same sequence file format, no changes to `states.json`. Policy reaches each waypoint while avoiding obstacles, then advances automatically when within tolerance.

### New Files Required

| File | Purpose |
|------|---------|
| `braccio_ctrl/rl_recorder.py` | 10 Hz NPZ recorder (obs, action, reward placeholder) |
| `braccio_ctrl/rl_env.py` | Gymnasium Env wrapping existing state objects |
| `braccio_ctrl/rl_sweeper.py` | Drop-in AutoSweeper replacement |
| `braccio_ctrl/rl_online_trainer.py` | Background SAC trainer + SharedReplayBuffer |
| `braccio_ctrl/rl_sequence_runner.py` | Goal-conditioned SequenceRunner |
| `braccio_ctrl/sim/braccio_env.py` | PyBullet sim env with domain randomization |
| `braccio_ctrl/sim/noise_params.json` | Calibrated noise params from session logs |
| `path_annotator.py` | Standalone annotation tool |
| `calibrate_noise.py` | One-shot noise model fitting script |
| `train_bc.py` | BC pre-training script |
| `train_rl.py` | SAC sim training script |

**New pip dependencies**: `stable-baselines3`, `gymnasium`, `imitation`, `fastdtw`, `pybullet`, `torch`

---

## Stage 4 — Training Pipeline (Session 1 complete)

### IR Emergency Stop Fast Path (added before Stage 4)

- `constants.py`: `IR_MIN = 1`, `IR_MONITOR_HZ = 50`
- `auto_sweep.py`: `SweepState.EMERGENCY_STOP`; `_ir_monitor_loop()` daemon at 50 Hz.
  Sets `_ir_estop` Event within ~20 ms of IR trigger. `_send_move()` hard-blocks
  while estop is set. Arm stops in <20 ms vs the 100 ms RL tick.
- `rl_reward.py`: removed dead `ir==2` branch (NOR-gate hardware outputs 0 or 3 only).

### Stage 4 Session 1 — Implemented

**`braccio_ctrl/rl_env.py`** (4.0 foundation)
- Abstract Gymnasium base class `BraccioBaseEnv`
- `observation_space = Box(-2, +2, shape=(74,))`
- `action_space = Box(-1, +1, shape=(3,))` — normalised [Δθ/THETA_STEP, Δz/Z_STEP, Δδ/DELTA_SPAN]
- Abstract methods: `_reset_state()`, `_get_obs()`, `_apply_action(action_n) → info dict`
- `step()`: obs_before → apply → obs_after → `compute_reward()` → return
- `reset()`: calls `_reset_state()`, tracks `_prev_action_n` for jerk term
- `_compute_done()`: terminates on `ir_bits >= 3`; truncates at `MAX_EPISODE_STEPS=500`
- Static helpers: `denormalize_action()`, `obs_to_arm_pose()`

**`braccio_ctrl/sim/__init__.py`** (sim package)
**`braccio_ctrl/sim/noise_params.json`** (default + DR ranges)
- Default values: servo_lag=0.92, tof_noise=10mm, dropout=0.15
- Domain randomisation ranges per plan (lag [0.8,1.2], noise [5,15] mm, dropout [0.05,0.25])
- `calibrated: false` until real data is fitted

**`calibrate_noise.py`** (4.1 noise calibration)
- Reads `rl_transitions_*.npz` from `logs/`
- Estimates servo_lag_factor via Δtheta_actual / Δtheta_commanded in moving ticks
- Estimates tof_noise_sigma_mm via temporal std of close-range cells in static segments
- Estimates cell_dropout_rate via CLEAR_SENTINEL fraction when obstacle map is active
- Writes `braccio_ctrl/sim/noise_params.json` with `calibrated: true`
- CLI: `python calibrate_noise.py` (auto-finds NPZs) or `python calibrate_noise.py logs/file.npz`

**`requirements_rl.txt`** (new pip dependencies for Stage 4+)
- `torch>=2.0`, `stable-baselines3>=2.3`, `gymnasium>=0.29`, `imitation>=1.0`
- `pybullet>=3.2`, `scipy>=1.12`, `fastdtw>=0.3.4`

**`DATA_COLLECTION_PROTOCOL.md`** (data collection guide — Stage 0)
**`collect_seq.txt`** (sequence editor program covering all workspace corners)

### Stage 4 Remaining (next sessions)

| Session | Files | Content |
|---------|-------|---------|
| 4 Session 2 | `braccio_ctrl/sim/braccio_env.py` Part A | URDF load + PyBullet step/reset + joint control |
| 4 Session 3 | `braccio_ctrl/sim/braccio_env.py` Part B | Synthetic ToF raycast + obstacle spawn + DR |
| 4 Session 4 | `train_bc.py` | BC pre-train from session NPZs |
| 4 Session 5 | `train_rl.py` | SAC sim training (warm-start BC) |
| 4 Session 6 | `SafeHardwareEnv` in `rl_sweeper.py` | Hardware fine-tune wrapper |

### Prompt for Session 2

> "Continue Stage 4 Session 2. Implement braccio_ctrl/sim/braccio_env.py Part A:
> PyBullet environment skeleton. Load URDF from assets/braccio_description/urdf/braccio_model.urdf.
> Implement step/reset/close with joint velocity control. Verify IK matches ik_solver.py.
> BraccioSimEnv inherits BraccioBaseEnv from rl_env.py — _reset_state, _get_obs (stub returning zeros),
> _apply_action (advance sim one tick). No raycasting yet. Write to MEMORY while implementing."
