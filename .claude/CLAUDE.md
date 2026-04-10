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
- CH2 (top/side): primary obstacle authority — promoted from advisory on 2026-04-10 after a hardware test showed it was silently ignoring a hand held close to the sensor
- CH3 (bottom): faces ground — completely ignored (floor false positives)

**IR wiring:** All 4 sensors share 5V and GND. Signal pins → Teensy 2, 3, 4, 5. Active LOW with `INPUT_PULLUP`. A count of firing sensors maps to severity: 0=CLEAR, 1=FAR, 2=CLOSE, 3=DANGER.

---

## Codebase Layout

```
braccio_main_runner/
  braccio_ctrl/
    __main__.py          — entry point (argparse, wires all objects together)
    controller.py        — main loop: keyboard input, obstacle checks, display refresh
    arm_state.py         — thread-safe 6-DOF joint state + IK helpers
    serial_bridge.py     — serial comms to Arduino (SET ALL / SET DELTA / GET POS)
    tof_sensor.py        — ToFBridge serial reader + ToFState (grids, IR, obstacle response)
    imu_state.py         — thread-safe IMU state with cached rotation matrix
    obstacle_map.py      — world-frame point cloud + Kalman tracker (6-DOF const-vel)
    obstacle_memory.py   — persistent voxel confidence grid (O(1) occupancy checks)
    auto_sweep.py        — autonomous sweep state machine + replanning logic
    ik_solver.py         — 2-link planar IK (law of cosines), reachability check
    constants.py         — ALL tunable parameters in one place
    display.py           — curses TUI
    protocol.py          — cmd_set_all(), cmd_set_delta() string builders
    state_library.py     — saved named arm poses (states.json)
    data_publisher.py    — UDP publisher for arm/ToF telemetry

ToF_RR_Sensing/arduino/vl5_tca_4x4/
  vl5_tca_4x4.ino        — Teensy firmware (ToF + IR + IMU serial output)
```

---

## Key Constants (constants.py)

```python
# Link lengths
L1, L2, L3 = 125.0, 125.0, 60.0   # mm

# ToF per-channel thresholds [CH0, CH1, CH2, CH3]
TOF_THRESHOLDS_MM = [250.0, 250.0, 250.0, 50.0]

# Sensor authority
SENSOR_REPLAN_CHANNELS   = [0, 1, 2]   # trigger REPLAN + gate manual input
SENSOR_ADVISORY_CHANNELS = []          # reserved; currently empty
SENSOR_IGNORE_CHANNELS   = [3]         # skip entirely (floor)

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
# Full controller (arm + ToF)
python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1

# Arm only (no ToF)
python -m braccio_ctrl /dev/ttyACM0

# ToF plotter only
python braccio_main_runner/tof_plotter_app.py --port /dev/ttyACM1
```

**Key keyboard bindings:**
- `Z` — toggle autonomous sweep
- `C` — calibrate IMU yaw reference
- `A/D` — theta (base rotation)
- `W/S` — r (reach)
- `Q/E` — z (height)
- `H` — go home
- `M` — saved states menu
- `ESC` — quit

---

## Architecture Notes

- All shared state objects use `threading.RLock()` for thread safety
- The sweep daemon thread runs at `SWEEP_TICK_HZ = 10 Hz`
- `_do_replan()` retries every tick — no blocking timeouts
- `_send_move()` validates against obstacle memory before issuing any command
- IMU rotation matrix is cached; invalidated only when roll/pitch/yaw changes
- Obstacle memory persists across sensor FOV gaps; cleared on IMU recalibration
