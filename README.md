# CSTONE_TESTGROUND

Two-MCU Braccio control stack with host-side obstacle-aware replanning:
- **Arduino Mega**: low-level actuator interface + MCU IK (`SET IKP`) + watchdog
- **Teensy**: ToF sensing via I2C MUX + MPU6050 on direct I2C (100 Hz stream)
- **Host PC**: UI, sensing ingest, obstacle modeling, linear-relaxed replanning, session tools

## 1. Repository Entry Points
- Main runtime: `braccio_main_runner/run_braccio.py`
- Host control package: `braccio_main_runner/braccio_ctrl`
- Session playback/export: `braccio_main_runner/session_plotter.py`
- Mega firmware: `braccio_main_runner/braccio_joint_test/braccio_joint_test.ino`
- Teensy firmware: `ToF_RR_Sensing/arduino/vl5_tca_RR/vl5_tca_RR/vl5_tca_RR.ino`

## 2. Hardware Setup (USB + Sensor Wiring)

### 2.1 USB links to host PC
Connect both MCUs directly to the host:
1. **Mega USB -> Host PC** (actuator/command link)
2. **Teensy USB -> Host PC** (sensor telemetry link)

Find COM ports:
```powershell
python .\braccio_main_runner\run_braccio.py --list-ports
```

### 2.2 Sensor channel map (current)
- **CH0**: ToF (side sensor)
- **CH1**: ToF (side sensor)
- **CH2 (logical stream id)**: MPU6050 IMU on direct I2C bus (top of wrist_pitch body)

ToF modeling uses a **63 deg x 63 deg square FOV** for CH0/CH1.

### 2.3 Home-pose directional reference used by controller
In home pose, in base XY plane:
- CH0 points toward **+Y**
- CH1 points toward **-Y**

Mount model note:
- ToF sensors are modeled at the **mid-body of `wrist_pitch`**, not exactly at the joint origin.
- Runtime transforms move sensor origin/orientation with arm FK each cycle.

## 3. Flash Firmware

### 3.1 Arduino Mega firmware
1. Open `braccio_main_runner/braccio_joint_test/braccio_joint_test.ino` in Arduino IDE.
2. Select board/port (Arduino Mega).
3. Upload.
4. Verify serial prints `READY`.

### 3.2 Teensy firmware
1. Open `ToF_RR_Sensing/arduino/vl5_tca_RR/vl5_tca_RR/vl5_tca_RR.ino`.
2. Select Teensy board/port.
3. Upload.
4. Verify stream contains `TF,...`, `IMU,...`, and `IR,...` lines.

Teensy command support:
- `MUX`, `CH0`, `CH1`
- `ACT <n>` where `n` is `1..2` (active ToF channels)

## 4. Python Environment Setup
From repo root:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r .\braccio_main_runner\requirements.txt
```

On Windows, also install curses support:
```powershell
pip install windows-curses
```

## 5. Run the Main Controller

### 5.1 Two-MCU mode (recommended)
```powershell
python .\braccio_main_runner\run_braccio.py <MEGA_PORT> --teensy-port <TEENSY_PORT> --tof-channels 2 --imu-lpf-alpha 0.25
```
Example:
```powershell
python .\braccio_main_runner\run_braccio.py COM5 --teensy-port COM3 --tof-channels 2 --imu-lpf-alpha 0.25
```

### 5.2 Mega-only mode
```powershell
python .\braccio_main_runner\run_braccio.py <MEGA_PORT> --no-tof
```

### 5.3 Optional visualization windows (disabled by default)
```powershell
python .\braccio_main_runner\run_braccio.py <MEGA_PORT> --teensy-port <TEENSY_PORT> --enable-plot --enable-tof-plot
```

## 6. Main UI Controls
- `A/D`: theta +/-
- `W/S`: r +/-
- `Q/E`: z +/-
- `I/K`: wrist offset +/-
- `J/L`: wrist rotation +/-
- `O` or `[`: gripper open
- `P` or `]`: gripper close
- `+/-`: slew delta
- `H`: go equilibrium/home
- `Shift+H`: set equilibrium from current
- `M`: states menu
- `X`: sequence editor
- `T`: tuning menu
- `C`: **IMU calibration sequence**
- `F` / `Shift+F`: ToF threshold +/- 50 mm
- `V`: ToF viewer toggle (if enabled)
- `N`: ToF screenshot (if enabled)
- `R`: side-to-side sweep test profile
- `B`: start/stop recording
- `ESC`: quit

### 6.1 Tuning menu parameters
- Weight parameters: `w_track_*`, `w_smooth_*`, `w_avoid_*`
- Distance threshold: `tof_threshold_mm`
- Decay time: `obstacle_decay_s`
- Override: `Persistent Memory` (`ACTIVE/INACTIVE`)
- IMU filter: `imu_lpf_alpha` (also available at startup via `--imu-lpf-alpha`)

## 7. IMU Calibration Workflow
At startup, IMU status is shown as:
- `[UNCALIBRATED]`

Press **`C`** to calibrate:
1. Host runs a short arm motion sequence.
2. Teensy IMU acceleration samples are paired with wrist orientation from FK.
3. Host estimates fixed IMU-to-wrist rotation (Wahba/Kabsch fit).
4. Status becomes:
- `[CALIBRATED]`

After calibration, host applies IMU tilt correction to wrist orientation used by ToF projection (improves obstacle localization robustness).

## 8. Integrated Side-to-Side Obstacle Test (Main Script)
Toggle with **`R`**.

Profile target:
- theta sweep: `0 <-> 180`
- `r = 112.0 mm`
- `z = 20.0 mm`
- wrist offset `0 deg`
- wrist rotation `90 deg`
- gripper `73 deg`

The profile remains endpoint-committed per leg (no mid-leg direction flip).

## 9. Recording Sessions
Toggle with **`B`**.

When stopping a recording:
1. Side profile is disabled.
2. Arm is commanded home/equilibrium.
3. UI shows `[SAVING RECORDING...]`.
4. Session is saved under timestamped folder.

Saved outputs:
- `logs/sessions/session_YYYYMMDD_HHMMSS/session.jsonl`
- `logs/sessions/session_YYYYMMDD_HHMMSS/meta.json`

## 10. Session Playback / Export
Run:
```powershell
python .\braccio_main_runner\session_plotter.py
```

Features:
- Dropdown session selector
- URDF arm playback
- Feasible workspace cloud overlay
- Obstacle bubbles (decay opacity in decay mode; miss-count fade in persistent mode)
- 5-second end-effector breadcrumb
- ToF sensor coverage cones attached to wrist_pitch
- Optional MP4 export (720p)

Playback keys:
- `Space`: pause/resume
- `Left/Right`: step frame
- `S`: save MP4 (720p)

Direct export:
```powershell
python .\braccio_main_runner\session_plotter.py --session .\logs\sessions\session_YYYYMMDD_HHMMSS --export-mp4
```

## 11. Safety/Fallback Behavior
- Mega watchdog enters hold/comms-fault mode if command stream times out.
- Host enters hold/fallback behavior on stale sensing or communication issues.
- Obstacle memory supports decay-time mode or persistent-remap mode (tuning menu override).

## 12. Useful Docs
- `docs/two_mcu_control_architecture.md`
- `docs/repo_architecture_two_mcu.md`
- `docs/message_protocols.md`
- `docs/optimizer_integration.md`
- `docs/obstacle_detection_pipeline.md`
- `docs/end_to_end_timing_budget.md`

## 13. Troubleshooting
- `Mega did not respond to PING/PONG`:
  - Verify Mega COM port.
  - Close other serial monitors.
  - Reflash `braccio_main_runner/braccio_joint_test/braccio_joint_test.ino`.
- No Teensy data:
  - Verify Teensy COM port and firmware.
  - Verify I2C MUX/sensor power/wiring.
- `_curses` import error (Windows):
  - `pip install windows-curses`
- MP4 export fails:
  - Install ffmpeg and add to PATH.



