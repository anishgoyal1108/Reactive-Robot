# Reactive Robot: An Autonomous Obstacle-Avoiding Robotic Arm with Real-Time Trajectory Replanning

<p align="center">
  <img src="https://github.com/anishgoyal1108/Reactive-Robot/blob/main/assets/braccio-render.png?raw=true" alt="Braccio V2 digital twin render" width="600">
</p>

<p align="center">
  A 6-DOF Braccio V2 servo arm with synchronous obstacle avoidance, a PyBullet digital twin, and a browser-based block programming editor.
</p>

<p align="center">
  <a href="https://anishgoyal1108.github.io/Reactive-Robot/">Wiki</a>
  ·
  <a href="https://anishgoyal1108.github.io/Reactive-Robot/editor/">Interactive Block Editor</a>
</p>

## Authors

- Anish Goyal [@anishgoyal1108](https://github.com/anishgoyal1108) ([ag29340@georgiasouthern.edu](mailto:ag29340@georgiasouthern.edu))
- Sebastian Oviedo ([so05798@georgiasouthern.edu](mailto:so05798@georgiasouthern.edu))
- Ariana Story ([as51527@georgiasouthern.edu](mailto:as51527@georgiasouthern.edu))
- Elizabeth McGlone ([em18976@georgiasouthern.edu](mailto:em18976@georgiasouthern.edu))

## Overview

Reactive Robot drives a 6-DOF Braccio V2 servo arm through a continuous base sweep from 0° to 180° and back while detecting and avoiding obstacles in real time. The system fuses four VL53L5CX 8x8 time-of-flight grids, four IR proximity sensors, and an MPU-6050 IMU into a world-frame point cloud, then validates every motion against the live cloud before the command reaches the servos.

We built the system around three guarantees:

1. Every motion request routes through the SafetyAPI chokepoint before the servos receive a signal.
2. Collision checks run synchronously against the live point cloud, not a cached snapshot.
3. Each working ToF channel holds replan authority by default, so a pixel with valid target returns gates motion even when sweep geometry suggests otherwise.

## Key Features

- __Synchronous safety chokepoint__: The SafetyAPI validates every manual keypress, sweep tick, DSL sequence step, and web command against the live point cloud.
- __BiRRT planner with capsule collision checks__: `safety.planner` wraps `pybullet_planning.birrt` and smooths the resulting path. `safety.collision` computes analytical capsule-to-point distance across all 6 arm links.
- __Cascading replanner__: Six recovery strategies fire in order when a segment fails, from micro-retreat to a full sweep-side flip.
- __Digital twin__: `braccio_twin` loads the same URDF into a headless PyBullet session, runs virtual ToF, IR, and IMU sensors, and exposes the same `BraccioBackend` Protocol as the hardware path.
- __Web application__: A FastAPI backend streams telemetry over WebSocket to a React + Three.js viewer. A Blockly editor compiles blocks into the Lark-based DSL and runs sequences through the same safety chokepoint.
- __Session telemetry__: `session_logger` writes 10 Hz JSONL traces of joint state, sensor frames, and safety-stack decisions for post-run analysis.

## Hardware

| Component | Detail |
|-----------|--------|
| Arm | Braccio V2, 6-DOF: base, shoulder, elbow, wrist-vertical, wrist-rotation, gripper |
| Arm controller | Arduino Mega running stock Braccio firmware at `/dev/ttyACM0` |
| Sensor controller | Teensy 4.1 at `/dev/ttyACM1` |
| ToF | 4x VL53L5CX on a TCA9548A I2C mux, 8x8 distance grid per sensor |
| IR | 4x EC-Buying IR obstacle sensors, active-LOW, 5 V, through a TI SN74LVC02A NOR-inverter stage to 3.3 V active-HIGH on Teensy pins 23, 22, 21, 20 |
| IMU | MPU-6050 on the Teensy, accelerometer-derived roll and pitch |

### Sensor Mounting

- __CH0__ (front side) faces the sweep direction as a primary obstacle authority.
- __CH1__ (back side) covers the opposite side as a primary obstacle authority.
- __CH2__ (top side) holds replan authority after a hardware test showed advisory classification let the arm sweep through a hand held close to the sensor.
- __CH3__ (bottom) is ignored to suppress floor false positives.

### IR Wiring

All four IR sensors share 5 V power and ground. Each active-LOW output feeds one input of a TI SN74LVC02A quad NOR gate running at VCC of 3.3 V, with the other input tied to ground so the gate inverts and clamps the signal to a Teensy-safe 3.3 V level. The firmware keeps `INPUT_PULLUP` enabled as a fail-safe: if the NOR chip loses power or a trace opens, the pin floats HIGH and the firmware treats the channel as an obstacle. A count of firing sensors maps to severity: 0 is CLEAR, 1 is FAR, 2 is CLOSE, 3 is DANGER.

## System Architecture

1. __Sensor ingestion__: `tof_sensor.py` parses Teensy serial frames (`TF,...`, `IR,...`, `IMU,...`) on a daemon thread.
2. __World model__: `safety.world_model.WorldModel` holds a timestamped point cloud, a KDTree for nearest-neighbor queries, and a coarse log-odds occupancy grid.
3. __Behavior tree__: `safety.behavior` routes requests by mode (manual, sweep, sequence) via a py-trees root. Each mode hits the same planner and collision checker.
4. __Planner__: `safety.planner.SafetyPlanner` runs BiRRT through `pybullet_planning` and smooths the result. `safety.collision` validates every interpolated waypoint against the live cloud.
5. __Replanner__: `safety.replanner.CascadingReplanner` applies six recovery strategies in order when a segment fails.
6. __Actuation__: `serial_bridge.py` writes `SET ALL B<deg> S<deg>...\n` commands to the Arduino and reads back `POS` responses.
7. __Telemetry__: `session_logger.py` emits JSONL frames and `data_publisher.py` streams UDP packets to `arm_plotter_app.py` and `tof_plotter_app.py`.

## Directory Structure

```
Reactive-Robot/
├── braccio_main_runner/
│   ├── braccio_ctrl/                 # Main Python control package
│   │   ├── __main__.py               # CLI entry point
│   │   ├── controller.py             # Curses TUI and main loop
│   │   ├── arm_state.py              # Thread-safe joint state + IK
│   │   ├── serial_bridge.py          # Arduino serial driver
│   │   ├── tof_sensor.py             # ToF + IR + IMU serial reader
│   │   ├── imu_state.py              # IMU state + cached rotation matrix
│   │   ├── constants.py              # Tunable parameters
│   │   ├── backends.py               # BraccioBackend Protocol
│   │   ├── session_logger.py         # 10 Hz JSONL trace writer
│   │   ├── data_publisher.py         # UDP telemetry publisher
│   │   ├── dsl/                      # Lark-based sequence DSL
│   │   └── safety/                   # Obstacle-aware motion planning
│   │       ├── api.py                # SafetyAPI chokepoint
│   │       ├── behavior.py           # py-trees behavior tree
│   │       ├── world_model.py        # Timestamped point cloud + KDTree
│   │       ├── fk.py                 # Headless PyBullet forward kinematics
│   │       ├── collision.py          # Capsule-to-point-cloud distance
│   │       ├── planner.py            # BiRRT + smoothing
│   │       ├── replanner.py          # Six-strategy recovery ladder
│   │       ├── polar_map.py          # 1D angular map for sweep skip
│   │       ├── pre_scan.py           # Wrist-sweep workspace scan
│   │       └── hysteresis.py         # Schmitt trigger, EMA, rate clamp
│   ├── braccio_twin/                 # PyBullet digital twin
│   │   ├── sim_backend.py            # SimBackend implementing BraccioBackend
│   │   ├── virtual_tof.py            # Raycast-based ToF sensor
│   │   ├── virtual_ir.py             # Raycast-based IR sensor
│   │   ├── virtual_imu.py            # Simulated accelerometer + gyroscope
│   │   ├── obstacle_world.py         # Loadable virtual obstacle scenes
│   │   ├── worlds/                   # JSON scene definitions
│   │   └── urdf/braccio.urdf         # Arm kinematic description
│   ├── run_braccio.py                # Standalone launcher
│   ├── arm_plotter_app.py            # Live matplotlib plot of joint angles
│   ├── tof_plotter_app.py            # Live ToF grid plotter
│   └── tests/                        # Pytest suites for safety + sensors
├── ToF_RR_Sensing/
│   ├── arduino/vl5_tca_4x4/          # Teensy firmware (ToF + IR + IMU)
│   └── python/                       # Teensy-side Python prototyping
├── web/
│   ├── backend/                      # FastAPI server + WebBridge
│   │   ├── app.py                    # FastAPI routes and WebSocket endpoint
│   │   ├── bridge.py                 # Session owner, sequence runner
│   │   └── models.py                 # Pydantic request and response schemas
│   └── frontend/                     # React + Three.js viewer
│       ├── src/viewer/               # Three.js scene, arm mesh, sensor rays
│       ├── src/editor/               # Blockly editor + DSL compiler
│       ├── src/recorder/             # Canvas recording
│       ├── src/state/                # Zustand stores for telemetry + URDF
│       └── src/mode/                 # Digital-twin vs. hardware mode toggle
├── requirements-dev.txt              # Development dependencies
└── pyproject.toml                    # Project metadata
```

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ for the web frontend
- Arduino IDE or `arduino-cli` for firmware flashing

### Python Setup

```bash
git clone https://github.com/anishgoyal1108/Reactive-Robot.git
cd Reactive-Robot
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r braccio_main_runner/requirements.txt
pip install -r requirements-dev.txt
```

### Firmware

1. Flash the Braccio firmware to the Arduino Mega via the Arduino IDE.
2. Open `ToF_RR_Sensing/arduino/vl5_tca_4x4/vl5_tca_4x4/vl5_tca_4x4.ino` and flash to the Teensy 4.1.

### Web Frontend

```bash
cd web/frontend
npm install
```

## Running the System

### Full Controller (Curses TUI)

```bash
python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1
```

Keybindings:

- `Z` toggles autonomous sweep.
- `C` calibrates the IMU yaw reference.
- `A` and `D` rotate the base.
- `W` and `S` adjust radial reach.
- `Q` and `E` adjust height.
- `H` returns the arm to home.
- `M` opens the saved-states menu.
- `ESC` quits.

### Arm Only (No ToF)

```bash
python -m braccio_ctrl /dev/ttyACM0
```

### Digital Twin

The headless PyBullet backend shares the BraccioBackend Protocol with the hardware path, so the same safety stack runs against virtual sensors.

```bash
python -m braccio_twin
```

### Web Stack

Terminal 1 runs the FastAPI backend on port 8011:

```bash
uvicorn web.backend.app:app --reload --port 8011
```

Terminal 2 runs the Vite dev server on port 5175:

```bash
cd web/frontend
npm run dev
```

Open `http://localhost:5175` in your browser.

### Standalone Plotters

Both plotters listen on UDP ports published by `data_publisher`:

```bash
python braccio_main_runner/arm_plotter_app.py
python braccio_main_runner/tof_plotter_app.py --port /dev/ttyACM1
```

### Tests

```bash
pytest braccio_main_runner/tests
cd web/frontend && npm test
```

## Module Reference

### `braccio_ctrl`

| Module | Purpose |
|--------|---------|
| `__main__.py` | Argparse CLI. Wires together the serial bridge, sensor reader, safety stack, and controller. |
| `controller.py` | Curses TUI and main input loop. Dispatches manual keys to SafetyAPI. |
| `arm_state.py` | Thread-safe 6-DOF joint state with IK helpers. |
| `serial_bridge.py` | Arduino serial driver for `SET ALL`, `SET DELTA`, `GET POS`. |
| `tof_sensor.py` | ToFBridge serial reader with per-frame status tracking. |
| `imu_state.py` | Thread-safe IMU state, cached rotation matrix, `wait_for_frame()` primitive. |
| `constants.py` | All tunable parameters: link lengths, ToF thresholds, sweep cadence, voxel size. |
| `backends.py` | `BraccioBackend` Protocol with hardware, sim, and in-memory implementations. |
| `ik_solver.py` | 2-link planar IK via law of cosines and reachability check. |
| `session_logger.py` | JSONL trace writer at 10 Hz for post-run analysis. |
| `data_publisher.py` | UDP broadcaster for live plot subscribers. |
| `dsl/` | Lark grammar, interpreter, and sequence runner for the block-compiled DSL. |

### `braccio_ctrl.safety`

| Module | Purpose |
|--------|---------|
| `api.py` | `SafetyAPI.plan_and_validate`, `queue_manual_intent`, `queue_sequence`. The single chokepoint for every motion request. |
| `behavior.py` | py-trees root selecting between manual, sweep, and sequence branches. |
| `world_model.py` | Timestamped point cloud, KDTree, coarse log-odds grid. |
| `fk.py` | Headless PyBullet forward kinematics returning link capsules for the full arm. |
| `collision.py` | Analytical capsule-to-point-cloud distance, including capsule-to-capsule for self-collision. |
| `planner.py` | BiRRT wrapper over `pybullet_planning.birrt` plus path smoothing. |
| `replanner.py` | Six-strategy cascading replanner for sequence segment failures. |
| `polar_map.py` | 1D angular occupancy map for sweep skip-ahead. |
| `pre_scan.py` | Wrist-sweep workspace observation routine. |
| `hysteresis.py` | Schmitt trigger, EMA, and rate clamp primitives used across the stack. |
| `dialog.py` | Curses overlay for manual-mode collision refusal. |

### `braccio_twin`

| Module | Purpose |
|--------|---------|
| `sim_backend.py` | `SimBackend` implementing the `BraccioBackend` Protocol against a headless PyBullet session. |
| `sim_arm.py` | PyBullet arm loader, joint setters, FK lookups. |
| `virtual_tof.py` | Raycast-based VL53L5CX emulator producing 8x8 distance grids. |
| `virtual_ir.py` | Raycast-based IR emulator producing 2-bit severity. |
| `virtual_imu.py` | Simulated accelerometer and gyroscope with configurable noise. |
| `obstacle_world.py` | Loadable virtual obstacle scenes from JSON. |

### `web`

| Module | Purpose |
|--------|---------|
| `backend/app.py` | FastAPI routes: state library, sequence CRUD, DSL run control, WebSocket telemetry. |
| `backend/bridge.py` | `WebBridge` ties backend, state library, sequence store, and subscribers together. |
| `backend/models.py` | Pydantic schemas for REST requests and responses. |
| `frontend/src/viewer/` | Three.js scene, URDF-driven arm mesh, sensor ray visualization. |
| `frontend/src/editor/` | Blockly editor, block definitions, DSL compiler. |
| `frontend/src/recorder/` | Canvas recording for live demo capture. |
| `frontend/src/state/` | Zustand stores for telemetry, URDF, and API wrappers. |
| `frontend/src/mode/` | Hardware vs. digital-twin mode toggle. |

## Coordinate System

- Origin at the arm base and shoulder pivot.
- +X axis points forward when theta is 0°.
- +Y axis points left when theta is 90°.
- +Z axis points upward.
- Theta is the base rotation angle in degrees, from 0 to 180.
- `r` is the radial reach in millimeters.
- `z` is the height above the shoulder pivot in millimeters.

## Serial Protocols

__Arduino (arm)__

- Send: `SET ALL B<deg> S<deg> E<deg> WV<deg> WR<deg> G<deg>\n`
- Send: `SET DELTA <n>\n` where `n` ranges from 1 to 5 for slew rate.
- Receive: `POS B<deg> S<deg> ...` with actual joint positions.

__Teensy (ToF + IR + IMU)__

- Receive: `TF,<seq>,<ms>,S<ch>,<ch>,<joint>,<status>,<rows>,<cols>,<d0..dN>,<v0..vN>`
- Receive: `IR,<0-3>` for 2-bit severity.
- Receive: `IMU,<seq>,<ms>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<temp>,<status>`
- Send: `ACT 4\n` to enable all 4 channels, `MUX\n`, `CH0\n` through `CH3\n`.

## License

MIT License. See `LICENSE` for the full text.
