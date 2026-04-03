# Robotic Arm Sensor Plan, Merge, and Remaining TODOs

## Branch Merge Checklist

1. Fetch and compare `main` vs `origin/claude/robotic-arm-sensor-plan-60YZ6`.
2. If Claude branch has unique commits, merge into `main` and resolve conflicts.
3. If Claude branch is behind `main`, keep `main` as source of truth.
4. Run tests and validation before pushing.

## Phase A - Context Parity

- Port missing items from Context history:
  - Per-channel ToF thresholds and authority rules.
  - Sweep replanning with Z-axis candidates and higher sweep Z default.
  - Pre-command collision guard in sweep path.
  - IMU rotation-matrix caching.
  - FK sync (`fk_polar`) on `POS` responses to avoid first-keypress drop.
  - Display updates for per-channel thresholds and channel roles.

## Phase B - Persistent Obstacle Memory

1. Add memory tuning constants in `constants.py`.
2. Add `obstacle_memory.py` with sparse voxel confidence map.
3. Feed memory from `obstacle_map.update()` world-frame observations.
4. Use memory fast-path in `auto_sweep._is_position_clear()`.
5. Show memory stats in curses display.

## Validation and Release

- Run project tests and targeted smoke checks.
- Verify serial diagnostics and plotter data path.
- Commit in logical chunks, then push `main`.
