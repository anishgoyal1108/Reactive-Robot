# CSTONE_TESTGROUND

Minimal two-MCU Braccio control stack:

- **Arduino Mega** runs the Braccio actuator interface and authoritative inverse kinematics through `SET IKP`.
- **Teensy** streams up to 4 VL53L5CX ToF channels plus direct-I2C MPU6050 IMU telemetry.
- **Host PC** runs the curses UI, detects obstacles, commits simple sector waypoint avoidance, records sessions, and renders playback.

## Entry Points

- Main runtime: `braccio_main_runner/run_braccio.py`
- Host control package: `braccio_main_runner/braccio_ctrl`
- Session playback/export: `braccio_main_runner/session_plotter.py`
- Mega firmware: `braccio_main_runner/braccio_joint_test/braccio_joint_test.ino`
- Hyperion Teensy firmware: `ToF_RR_Sensing/arduino/vl5_tca_hyperion_4x4/vl5_tca_hyperion_4x4/vl5_tca_hyperion_4x4.ino`
- Legacy 2-ToF Teensy firmware: `ToF_RR_Sensing/arduino/vl5_tca_RR/vl5_tca_RR/vl5_tca_RR.ino`

## Sensor Layout

- `CH0`: west/left side of wrist, authoritative detector
- `CH1`: east/right side of wrist, authoritative detector
- `CH2`: north/top support channel
- `CH3`: south/bottom support channel
- IMU: MPU6050 on direct I2C, independent of the ToF MUX

The host defaults to 4 installed ToF channels when a Teensy is connected. During normal sweep detection it commands the Teensy to `ACT 2`; during confirmation and avoidance tracking it commands `ACT 4`.

## Runtime Behavior

The controller is a small finite-state machine:

1. `detect_sweep`: sweep normally and detect only with `CH0/CH1` at 250 mm by default.
2. `confirm_stop`: stop the arm, switch to `ACT 4`, and confirm the obstacle with a 3-of-5 filtered sample window.
3. `vertical_probe`: keep `ACT 4`, sweep vertically within the configured Z scan limits, require the source channel to clear, and require `CH2/CH3` support clearance.
4. `planning_hold`: store the vertical opening and choose a deterministic waypoint bypass, allowing safe egress from an already inflated detected sector.
5. `avoid_execute`: execute the committed bypass while all ToF channels track the stored obstacle.
6. `tracking_sweep`: finish the current sweep leg with all channels active.
7. Return to `detect_sweep`, clear temporary memory, and switch back to `ACT 2`.

There is no QP, OSQP, relaxed optimizer, feasible-cloud generator, or startup baseline scan in the runtime.

## Running

List ports:

```powershell
python .\braccio_main_runner\run_braccio.py --list-ports
```

Run with Mega and Teensy:

```powershell
python .\braccio_main_runner\run_braccio.py COM5 --teensy-port COM3
```

Run Mega-only:

```powershell
python .\braccio_main_runner\run_braccio.py COM5 --no-tof
```

Useful runtime tuning:

```powershell
python .\braccio_main_runner\run_braccio.py COM5 --teensy-port COM3 --detect-threshold-mm 250 --confirm-frames 5 --confirm-hits 3
```

## UI Controls

- `A/D`: theta +/-
- `W/S`: reach +/-
- `Q/E`: height +/-
- `I/K`: wrist vertical offset +/-
- `J/L`: wrist rotation +/-
- `O` or `[`: gripper open
- `P` or `]`: gripper close
- `+/-`: slew delta
- `H`: go equilibrium/home
- `Shift+H`: set equilibrium from current pose
- `R`: side-to-side sweep profile
- `B`: start/stop recording
- `C`: IMU calibration
- `F` / `Shift+F`: ToF detection threshold +/- 50 mm
- `V`: ToF viewer toggle when enabled
- `N`: ToF screenshot when enabled
- `ESC`: quit

## Sweep Profile

The old side-to-side profile is preserved:

- theta sweep: `5 deg <-> 175 deg`
- `r = 112.0 mm`
- `z = 20.0 mm`
- wrist offset `0 deg`
- wrist rotation `90 deg`
- gripper `73 deg`
- step `1.5 deg`
- dwell `0.8 s`

## Recording And Playback

Press `B` to start/stop recording. Sessions are saved under:

- `logs/sessions/session_YYYYMMDD_HHMMSS/session.jsonl`
- `logs/sessions/session_YYYYMMDD_HHMMSS/meta.json`

The recording schema is `braccio_session_v2`. Each row stores compact pose, joint, ToF minima/freshness, IMU health, obstacle memory, committed plan, mode transitions, planner candidate debug, and control timing.

Playback:

```powershell
python .\braccio_main_runner\session_plotter.py
python .\braccio_main_runner\session_plotter.py --session .\logs\sessions\session_YYYYMMDD_HHMMSS --export-mp4
```

The session plotter no longer loads or renders feasible workspace clouds. It keeps robot playback, obstacle spheres, committed plan preview, joint angle traces, and planner-state timelines.

## Tests

```powershell
cd .\braccio_main_runner
python -m pytest tests\test_protocols.py tests\test_tof_sensor.py tests\test_projection.py tests\test_sector_planner.py tests\test_minimal_controller.py tests\test_session_plotter.py
python -m pytest
```

## Notes

- `CH0/CH1` are the only normal obstacle initiators.
- `CH2/CH3` are support/tracking channels for this rebuild.
- IMU correction is used for wrist-frame ToF projection after calibration; the host falls back to kinematic pose when IMU data is stale or uncalibrated.
- The Mega watchdog still owns low-level stale-command hold behavior.
