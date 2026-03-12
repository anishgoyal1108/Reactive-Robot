# Repository Architecture (Two-MCU)

## Repo Tree Summary
- `braccio_main_runner/`
  - `run_braccio.py` host launcher
  - `braccio_ctrl/` host UI/control loop + comm bridges
  - `planning/` host obstacle modeling + replanning
  - `sensing/` host packet/frame utilities
  - `braccio_joint_test/braccio_joint_test.ino` Mega firmware (actuation + MCU IK)
  - `tests/` host unit tests
- `ToF_RR_Sensing/`
  - `arduino/vl5_tca_RR/vl5_tca_RR/vl5_tca_RR.ino` Teensy firmware

## Host-Side Entry Points
- `braccio_main_runner/run_braccio.py`
- `braccio_main_runner/braccio_ctrl/__main__.py`

## Teensy-Side Source Files
- `ToF_RR_Sensing/arduino/vl5_tca_RR/vl5_tca_RR/vl5_tca_RR.ino`

## Mega-Side Source Files
- `braccio_main_runner/braccio_joint_test/braccio_joint_test.ino`

## Current Sensor/Control Data Path
1. Teensy polls ToF channels CH0/CH1 via TCA9548A and streams `TF,...` frames.
2. Teensy reads MPU6050 on direct I2C and streams `IMU,...` telemetry (logical CH2 stream id).
3. Host ingests `TF/IMU/IR` in `braccio_ctrl/tof_sensor.py`.
4. Host computes wrist-frame obstacle points (FK + IMU tilt correction after calibration).
5. Host replanner computes safe polar command when obstacle risk is active.
6. Host sends **MCU-IK command** `SET IKP ...` (or `CMD,...` compatible path) to Mega.
7. Mega solves IK, applies joints, enforces timeout hold watchdog.

## Communication Paths
- Host <-> Teensy: USB serial ASCII (`TF`, `IMU`, `IR`, `MODE`, `CFG`).
- Host <-> Mega: USB serial ASCII (legacy + v1 packets).
- Teensy <-> Mega: none.

## Timing Assumptions in Code
- Mega servo update tick: 10 ms.
- Mega command timeout watchdog: 750 ms.
- Teensy target ToF frequency: ~15 Hz per active ToF sensor.
- Teensy IMU stream interval: 10 ms (100 Hz target).
- Host freshness gate: 0.4 s for ToF/IMU use.

## Responsibility Table
| Function | Current Execution | Desired Execution | Notes / Required Modifications |
|---|---|---|---|
| ToF polling | Teensy | Teensy | CH0/CH1 round-robin over MUX |
| MUX channel switching | Teensy | Teensy | Implemented in firmware |
| IMU polling (MPU6050) | Teensy | Teensy | Direct I2C read, streamed as IMU logical CH2 telemetry |
| Sensor frame parsing | Host | Host | Robust parser for `TF` + `IMU` + `IR` |
| Obstacle modeling | Host | Host | Base-frame points + feasible-region filtering |
| Base-frame transforms | Host | Host | FK + optional IMU tilt correction |
| IK solve | Mega MCU | Mega MCU | `SET IKP` path preserved as authoritative |
| Command generation | Host | Host | Host generates IK target / mode-aware commands |
| Actuator command streaming | Mega | Mega | Existing interface + watchdog |
| Trajectory execution | Mega | Mega | Servo updates at 100 Hz |
| Low-level safety fallback | Mega | Mega | Hold on stale/invalid command stream |

## Final Proposed Architecture
- **Teensy**: fast sensing node (ToF + IMU + lightweight validation + packetization).
- **Mega**: actuator and authoritative IK execution node.
- **Host**: obstacle interpretation, replanning, mode management, diagnostics, recording/playback.

## Data Flow Diagram
`ToF/IMU sensors -> Teensy (MUX poll + packetize) -> Host (parse + model + replan) -> Mega (IK + apply + watchdog) -> Arm`

## Known Gaps
- IMU calibration is runtime-estimated and should still be validated against fixture measurements.
- Full HIL latency benchmark should be re-run after firmware flashing on target hardware.
- Exact mechanical extrinsics can still be refined with measured mount offsets.

