# Repository Architecture (Two-MCU)

## Current Shape

- `braccio_main_runner/run_braccio.py`: host launcher
- `braccio_main_runner/braccio_ctrl/`: curses UI, serial bridges, finite-state controller
- `braccio_main_runner/planning/sector_planner.py`: deterministic sector waypoint planner
- `braccio_main_runner/planning/sensor_config.py`: ToF mount geometry source
- `braccio_main_runner/planning/tof_projection.py`: ToF cell projection into base frame
- `braccio_main_runner/planning/urdf_kinematics.py`: URDF forward kinematics for projection and playback
- `braccio_main_runner/sensing/`: ToF frame parsing and stream helpers
- `braccio_main_runner/session_plotter.py`: session playback/export
- `braccio_main_runner/braccio_joint_test/braccio_joint_test.ino`: Mega firmware with MCU IK
- `ToF_RR_Sensing/arduino/vl5_tca_hyperion_4x4/...ino`: 4x ToF + IMU Teensy firmware

## Runtime Data Path

1. Teensy polls the active ToF channels and streams `TF,...` frames.
2. Teensy streams MPU6050 telemetry as `IMU,...`.
3. Host parses ToF/IMU frames in `braccio_ctrl/tof_sensor.py`.
4. Host projects valid ToF cells into base frame using URDF FK and IMU correction when calibrated.
5. Host detects obstacles with `CH0/CH1`, confirms with filtered samples, and stores one temporary obstacle sector.
6. Host probes vertically with all ToF channels active, requiring the source channel to clear and `CH2/CH3` support clearance.
7. Host chooses a deterministic waypoint bypass and sends `SET IKP ...` polar targets to the Mega.
8. Mega solves IK, applies servo commands, and enforces its command watchdog.

## Responsibility Table

| Function | Owner |
|---|---|
| ToF polling and MUX switching | Teensy |
| IMU polling | Teensy |
| ToF/IMU parsing | Host |
| Obstacle detection and confirmation | Host |
| Sector waypoint planning | Host |
| Inverse kinematics | Mega |
| Servo actuation and watchdog | Mega |
| Recording and playback | Host |

## Notes

- Normal detection uses `ACT 2` and only `CH0/CH1` can initiate obstacle events.
- Confirmation, vertical probing, and avoidance tracking use `ACT 4`.
- The optimizer/QP/feasible-cloud stack has been removed from the host runtime.
