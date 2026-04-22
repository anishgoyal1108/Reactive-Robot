# Plan — Braccio Digital Twin + Block-Based Programming Environment

This is a multi-phase plan. Each phase leaves the system in a working, demoable state. You can stop after any phase.

---

## What already exists (verified by exploration)

| Component | Status | Lives in |
|---|---|---|
| State library (17 named poses) | Full | `braccio_ctrl/state_library.py` + `states.json` |
| Sequence editor + runner | Full | `braccio_ctrl/sequence_editor.py` |
| Mini sequence DSL (`STATE_NAME WAIT_MS` + `REPEAT N`) | Minimal regex parser | `sequence_editor.py:398-439` |
| Serial protocol → arm | Full | `protocol.py`, `serial_bridge.py` |
| ToF/IR/IMU bridge | Full | `tof_sensor.py`, `imu_state.py` |
| Obstacle map + memory + MotionGuard | Full | `obstacle_map.py`, `obstacle_memory.py`, `motion_guard.py` |
| UDP telemetry publisher | Full | `data_publisher.py` |
| 2D matplotlib plotters | Full | `plotter.py`, `tof_plotter.py` |
| Curses TUI | Full | `display.py`, `states_menu.py` |
| **3D arm rendering** | **Missing** | — |
| **Web frontend / Blockly editor** | **Missing** | — |
| **MP4 / video export** | **Missing** | — |
| **Physics-sim digital twin** | **Missing** | — |
| **Formal AST/grammar for the DSL** | **Missing** | — |

**Important constraint already in memory:** safety checks must be synchronous and routed through `MotionGuard.plan_clear_pose()` — the digital twin must keep this contract intact when it stands in for the real arm.

---

## Recommended architecture (high level)

```
┌──────────────────────────────────────────────────────────────────────┐
│                          BROWSER (kid's view)                        │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐  │
│  │  Blockly     │   │  Three.js    │   │  States / Obstacles /    │  │
│  │  block       │──▶│  3D viewer   │   │  Run / Record / Deploy   │  │
│  │  editor      │   │  (URDF)      │   │  side panel              │  │
│  └──────┬───────┘   └──────▲───────┘   └────────────┬─────────────┘  │
│         │ DSL text         │ joint+sensor stream    │ REST           │
└─────────┼──────────────────┼────────────────────────┼────────────────┘
          │ WebSocket        │                        │
┌─────────▼──────────────────┴────────────────────────▼────────────────┐
│              FastAPI server (web/backend/app.py)                     │
│  - REST: states, sequences, obstacles, deploy mode                   │
│  - WebSocket: telemetry stream (joints, ToF, IR, events)             │
│  - Owns one BraccioBackend instance:                                 │
│      ┌─────────────────────────────┐    ┌────────────────────────┐   │
│      │ SimBackend (PyBullet twin)  │ OR │ HardwareBackend        │   │
│      │ - virtual servos            │    │ - existing SerialBridge│   │
│      │ - virtual ToF/IR raycasts   │    │ - existing ToFBridge   │   │
│      │ - Obstacle() world          │    │ - real /dev/ttyACM*    │   │
│      └─────────────────────────────┘    └────────────────────────┘   │
│                          ▲                                           │
│                          │ Both speak the same internal interface    │
│                          │ (BraccioBackend ABC)                      │
│  - Reuses obstacle_map, obstacle_memory, motion_guard, auto_sweep    │
│  - Reuses sequence runner & DSL interpreter                          │
└──────────────────────────────────────────────────────────────────────┘
```

**Key idea:** introduce a `BraccioBackend` abstract base class that both `SimBackend` (PyBullet) and `HardwareBackend` (existing SerialBridge + ToFBridge) implement. The controller, MotionGuard, sweep logic, and DSL interpreter all talk to a `BraccioBackend` — they don't care if it's a real arm or a sim. The "Run on sim / Run on hardware" toggle is one line of switching at the FastAPI layer.

---

## Recommended technology choices (with rationale)

| Decision | Recommendation | Why |
|---|---|---|
| Sim engine (Python side) | **PyBullet** | Pure pip install, loads URDF directly, real-time physics, fast raycasts for ToF/IR, optional GUI window. MuJoCo is overkill; Gazebo is too heavy. |
| URDF source | **Existing Braccio URDF** (`lots-of-things/braccio_arduino_ros_rviz` or `arduino-braccio` description) | Joints, axes, limits, meshes already wired. Building one from a raw STL would burn days before any pixels move. We add virtual ToF/IR sensor frames on top. |
| 3D viewer (browser) | **Three.js + react-three-fiber** with **urdf-loaders** | Industry standard, free, MIT, runs anywhere a browser does. Same URDF file as PyBullet so the two views stay consistent. |
| Web stack | **FastAPI + Vite + React + TypeScript** | Python backend matches the existing codebase. React/TS is the most common stack for Blockly + Three.js. Vite for fast dev iteration. |
| Block editor | **Google Blockly** | Industry-standard for kid-facing block programming. Used by Scratch, MakeCode, etc. Has a built-in code generator framework — perfect for compiling blocks → DSL text. |
| DSL parser | **Lark** (Python) | Lightweight, EBNF-style grammar, produces a clean AST. Backwards compatible with the existing line-based mini-DSL. |
| MP4 export | **Browser-side MediaRecorder** capturing the live Three.js canvas, output as WebM, optional ffmpeg.wasm transcode to MP4 | Matches your "real-time render, not playback" requirement. The recording IS the live animation, captured frame-by-frame from the canvas as the sequence runs. |
| Virtual serial bridge | **PTY pair** (`pty.openpty()`) | Optional drop-in mode where the sim looks exactly like `/dev/ttyACM0` + `/dev/ttyACM1`. Lets the existing `python -m braccio_ctrl` CLI run unchanged against the twin. |

---

## Phase 0 — Repo prep & shared abstractions

**Goal:** make the existing code aware that there can be multiple backends, without changing any current behavior.

**New files:**
- `braccio_main_runner/braccio_ctrl/backend.py`
  - `class BraccioBackend(ABC)` with methods:
    - `connect()` / `close()` / `is_connected()`
    - `send_joints(positions: list[int]) -> None`
    - `set_delta(delta: int) -> None`
    - `get_positions() -> list[int]`
    - `tof_snapshot() -> dict` (per-channel grids, IR bits, IMU)
    - `subscribe_telemetry(callback)` (push model)
- `braccio_main_runner/braccio_ctrl/hardware_backend.py`
  - `class HardwareBackend(BraccioBackend)` — thin adapter that wraps the existing `SerialBridge` + `ToFBridge` + `IMUState`. **Pure adapter** — no behavior change.

**Modified files:**
- `controller.py` — call sites that currently reach into `serial_bridge` / `tof_state` directly are routed through `self._backend` (a `BraccioBackend`). Default `--port /dev/ttyACM0 --teensy-port /dev/ttyACM1` constructs a `HardwareBackend`. Behavior is identical.

**Done when:** real-arm runs work exactly as today, but the controller no longer mentions `SerialBridge` directly.

---

## Phase 1 — Python digital twin (PyBullet)

**Goal:** a self-contained Python sim of the Braccio + sensors + obstacles. Runs the existing controller and sweep logic against simulated hardware.

**New package:** `braccio_twin/`
- `__init__.py`
- `__main__.py` — CLI: `python -m braccio_twin [--gui] [--pty] [--port-arm /tmp/...] [--port-teensy /tmp/...]`
- `urdf/braccio.urdf` + `urdf/meshes/*.stl` — vendored from an existing Braccio URDF repo (with attribution)
- `world.py` — PyBullet world setup (gravity, ground plane, lighting if `--gui`)
- `arm.py` — loads URDF, exposes `set_joint_targets(deg: list[int])`, `get_joint_positions() -> list[int]`, `step(dt)`
  - Maps the 6 joint names from `JOINT_TOKENS` to PyBullet joint indices
  - Applies a position controller per joint with a configurable max velocity matching the real Braccio's slew rate (so `SET DELTA` translates correctly)
- `tof_sim.py` — `class VirtualToF`
  - 4 sensors mounted at the wrist link in 4 directions: **+Z (up), −Z (down), +Y (left), −Y (right)** — matches your spec
  - Each sensor casts an 8×8 ray bundle in a configurable FOV (matches VL53L5CX field of view), reads the closest hit per ray, returns mm distances
  - Uses `pybullet.rayTestBatch()` for speed
  - Output format identical to the Teensy `TF,...` line so the existing `_parse_tf` can ingest it without changes
- `ir_sim.py` — `class VirtualIR`
  - 4 sensors mounted on the base, two per side, in your "circular pockets" (configurable XY offsets)
  - Each casts a single short ray (matches EC-Buying IR module's ~30 cm range)
  - Returns active-LOW count → 2-bit severity (0/1/2/3) matching the existing firmware
- `imu_sim.py` — synthesizes accelerometer + gyro from PyBullet base orientation; output matches the existing `IMU,...` line format
- `obstacles.py`
  - `class Obstacle` — simple shapes (box / sphere / cylinder), placed by `(theta, r, z)` polar OR `(x, y, z)` cartesian
  - `class ObstacleWorld` — owns the list, spawns/despawns rigid bodies in PyBullet, exposes JSON (de)serialization for save/load
- `sim_backend.py`
  - `class SimBackend(BraccioBackend)` — wraps `Arm + VirtualToF + VirtualIR + VirtualIMU + ObstacleWorld` and exposes the same interface as `HardwareBackend`
  - Spawns a daemon thread that steps PyBullet at 240 Hz and pushes telemetry callbacks at 30 Hz
- `pty_bridge.py` — *optional drop-in mode*
  - Opens two PTY pairs via `pty.openpty()`
  - One mimics `/dev/ttyACM0`: parses `SET ALL ...`, `SET DELTA ...`, `GET POS\n`, `PING\n`, replies with `OK ALL=...`, `POS=...`, `PONG`
  - One mimics `/dev/ttyACM1`: emits `TF,...`, `IR,...`, `IMU,...`, `MODE,...`, `CFG,...` lines from the sim
  - When the user does `python -m braccio_twin --pty`, the sim prints two `/dev/pts/N` paths the user can pass to `python -m braccio_ctrl`. **Drop-in: existing controller runs against the twin with zero changes.**

**Validation:**
- Unit test: feed a known sequence of joint commands, assert the sim arm reaches them within tolerance
- Integration test: run `python -m braccio_ctrl` against the PTY-bridged sim, press `Z` (autonomous sweep), verify the sweep + obstacle avoidance loop works end-to-end against an `Obstacle` placed in the path
- Verifies the safety chokepoint memory (`MotionGuard.plan_clear_pose`) still gates every move — the sim feeds the same ToF format the real Teensy does

**Done when:**
- `python -m braccio_twin --gui` opens a window showing the arm
- `python -m braccio_twin --pty` prints two PTY paths
- `python -m braccio_ctrl /dev/pts/3 --teensy-port /dev/pts/4` runs the curses TUI against the sim and the sweep / obstacle avoidance behaves correctly

---

## Phase 2 — Extended DSL with formal grammar

**Goal:** evolve the regex parser into a real (but small) language so blocks have something concrete to compile to. Stay backwards compatible with existing `.txt` sequences.

**New package:** `braccio_main_runner/braccio_ctrl/dsl/`
- `grammar.lark` — Lark EBNF
- `parser.py` — `parse(text) -> Program`
- `ast_nodes.py` — `Program`, `MoveStmt`, `SetJointStmt`, `WaitStmt`, `RepeatBlock`, `IfBlock`, `WhileBlock`, `DefineState`, `DefineObstacle`, `Comment`
- `interpreter.py` — `class Interpreter` walks the AST against a `BraccioBackend`, with hooks for live status updates (so the editor can highlight the currently-executing line/block)
- `compat.py` — accepts the original line-based format and emits the same `Program` AST so existing `TEST_INPUT_SEQUENCE.txt` files still run

**Grammar sketch:**
```
program     := statement*
statement   := move_stmt | set_stmt | wait_stmt | repeat_blk
             | if_blk | while_blk | def_state | def_obs | comment
move_stmt   := "MOVE" IDENT ("WAIT" INT)?
set_stmt    := "SET" JOINT INT ("WAIT" INT)?
wait_stmt   := "WAIT" INT
repeat_blk  := "REPEAT" INT "{" statement* "}"
if_blk      := "IF" cond "{" statement* "}" ("ELSE" "{" statement* "}")?
while_blk   := "WHILE" cond "{" statement* "}"
cond        := sensor_expr CMP INT
sensor_expr := "TOF" "[" INT "]" | "IR" | "JOINT" "[" JOINT "]"
def_state   := "STATE" IDENT "{" "JOINTS" INT INT INT INT INT INT "}"
def_obs     := "OBSTACLE" IDENT "{" obs_field+ "}"
obs_field   := "POS" FLOAT FLOAT FLOAT | "RADIUS" FLOAT | "SHAPE" SHAPE
JOINT       := "B" | "S" | "E" | "WV" | "WR" | "G"
SHAPE       := "BOX" | "SPHERE" | "CYLINDER"
```

**Modified files:**
- `sequence_editor.py` — replace `_parse_sequence` with the new parser; keep the curses overlay as-is. The runner becomes a thin shim around `Interpreter`. Existing `.txt` programs continue to work via `compat.py`.

**Done when:** every state in `states.json` and the existing `TEST_INPUT_SEQUENCE.txt` runs unchanged, AND a new sequence with `IF TOF[0] < 200 { MOVE STOW_COMPACT }` runs against the sim.

---

## Phase 3 — FastAPI backend + WebSocket telemetry

**Goal:** put the digital twin behind a web API so the browser can talk to it.

**New package:** `web/backend/`
- `app.py` — FastAPI app
- `models.py` — pydantic schemas for joint state, ToF grid, IR, obstacles, sequence, state library entry
- `bridge.py` — owns one `BraccioBackend` instance (sim by default, hardware on toggle); subscribes to telemetry callbacks and fans them out to all connected WebSocket clients
- REST endpoints:
  - `GET /states` / `POST /states` / `DELETE /states/{name}` — wraps `StateLibrary`
  - `GET /sequences` / `POST /sequences` / `GET /sequences/{name}/run` / `POST /sequences/stop`
  - `GET /obstacles` / `POST /obstacles` / `DELETE /obstacles/{id}` — only valid in sim mode; rejects in hardware mode
  - `POST /mode` `{"mode": "sim" | "hardware"}` — swap backends at runtime
  - `GET /urdf` / `GET /meshes/{file}` — serves the URDF + meshes so the browser viewer loads the same model the sim uses
- WebSocket endpoint `/ws/telemetry` — pushes joint angles, sensor snapshots, sequence status, sweep state, error events at ~30 Hz

**Done when:** `uvicorn web.backend.app:app` runs, `curl /states` returns the 17 saved states, and `wscat -c ws://localhost:8000/ws/telemetry` streams joint updates while the sim is running.

---

## Phase 4 — Three.js viewer

**Goal:** render the arm + sensors + obstacles live in the browser, fed by the WebSocket.

**New package:** `web/frontend/`
- `package.json`, `vite.config.ts`, `tsconfig.json`
- `src/`
  - `App.tsx` — top-level layout: editor pane (left), viewer pane (center), state/obstacle panel (right)
  - `viewer/Scene.tsx` — react-three-fiber `<Canvas>`, lighting, ground, camera controls
  - `viewer/Arm.tsx` — loads `/urdf` via `urdf-loaders`, exposes `setJoints(angles[])`; subscribes to the WebSocket and updates joints every frame
  - `viewer/SensorRays.tsx` — draws ToF cones (4 per wrist) and IR rays (4 at base) with color coded by detection state (green/yellow/red)
  - `viewer/Obstacles.tsx` — draws spawned `Obstacle()` objects as semi-transparent meshes
  - `state/telemetry.ts` — WebSocket client + zustand store
  - `state/api.ts` — REST client wrappers
- The viewer uses **the same URDF** PyBullet uses (served from `/urdf`), so the on-screen arm and the sim arm cannot drift.

**Done when:** the browser shows the arm moving in real time as the sim runs the autonomous sweep, with ToF cones turning red when an obstacle blocks them and IR rays lighting up at the base.

---

## Phase 5 — Blockly block-based programming editor

**Goal:** kids drag blocks together; blocks compile to the Phase 2 DSL; the sequence runs on the sim and animates in the viewer; "deploy" runs the same sequence on the real arm.

**New files in `web/frontend/src/editor/`:**
- `BlocklyEditor.tsx` — embeds Blockly workspace, listens for changes, compiles to DSL text
- `blocks.ts` — custom block definitions corresponding 1:1 to Phase 2 DSL nodes:
  - **Movement blocks:**
    - "Move to state [dropdown of saved states]" → `MOVE state_name`
    - "Wait [number] ms" → `WAIT n`
    - "Set joint [B/S/E/WV/WR/G] to [number]°" → `SET JOINT n`
  - **Control flow blocks:**
    - "Repeat [n] times { ... }" → `REPEAT n { ... }`
    - "If sensor [ToF CH0/CH1/CH2/CH3 / IR] reads less than [mm] { ... }" → `IF`
    - "While sensor [...] [...] [...] { ... }" → `WHILE`
  - **Definition blocks (abstraction):**
    - "Define state [name] with joints [B][S][E][WV][WR][G]" → `STATE { JOINTS ... }`
    - "Define obstacle [name] at theta=[..] r=[..] z=[..] radius=[..]" → `OBSTACLE { ... }`
  - **Math/expression blocks** for "on-the-fly calculations": Blockly's built-in math/variable/text blocks, generating intermediate values that feed into Move / Set / Wait inputs
- `dslGenerator.ts` — Blockly code generator; emits the Phase 2 DSL grammar exactly. Round-trips: DSL text can be loaded back into the workspace via `dslImporter.ts`.
- `Toolbox.tsx` — categorized block palette: Movement, Sensing, Control, States, Obstacles, Math, Variables
- `RunControls.tsx` — Run / Stop / Step / Reset buttons, mode toggle (Sim / Hardware), running line highlight via the interpreter's status callback over WebSocket
- `ManualJointPanel.tsx` — six sliders bound to `set_joint` REST calls so a kid can manually pose the arm and "Save as state" with a name. Becomes a new entry in `states.json` (server-side via `POST /states`).

**Compatibility loop:**
1. Kid builds blocks → workspace serializes → DSL text
2. DSL text → POST `/sequences` → saved on disk (compatible with `sequence_editor.py`!)
3. Run → server interprets DSL → telemetry streams → viewer animates
4. Same DSL file can be opened in the existing curses `sequence_editor.py` for power users

**Done when:** a kid can build "MOVE LIFT_HIGH_CARRY → IF TOF[0] < 200 { MOVE STOW_COMPACT } → REPEAT 3 { ... }" entirely with blocks, hit Run, and watch the simulated arm execute it with obstacle reactions.

---

## Phase 6 — Live MP4 / video export from the live render

**Goal:** "render to video" that is **NOT** a saved-file playback — the recorder captures the live Three.js canvas as the sequence executes.

**Implementation:**
- Use `canvas.captureStream(30)` on the Three.js renderer's canvas to get a `MediaStream`
- Pipe through `MediaRecorder` with `video/webm; codecs=vp9` (universally supported in modern browsers)
- "Record" button starts a fresh sequence run AND begins recording in lockstep
- "Stop" finalizes the WebM blob; user can download it directly
- Optional MP4 transcoding via `@ffmpeg/ffmpeg` (ffmpeg.wasm) — happens client-side after recording, no server dependency
- Output filename includes sequence name + timestamp, e.g. `META_FULL_DEMO_2026-04-10_15-22-08.mp4`

**Why this matches your spec:** the video is produced *by* the renderer, frame-by-frame, while the simulation is actually running. There is no pre-rendered file being played back. If a kid edits the sequence, the next recording is a fresh re-render.

**Done when:** a kid clicks Record, the sequence runs, the file downloads, and the downloaded video shows the same animation that just played in the viewer.

---

## Phase 7 — Deploy-to-hardware bridge

**Goal:** one click to switch the same sequence from running on the sim to running on the real arm.

**Implementation:**
- The `BraccioBackend` abstraction from Phase 0 makes this nearly free
- `POST /mode {"mode": "hardware"}` tears down `SimBackend`, instantiates `HardwareBackend`, reconnects the existing daemon threads, and broadcasts the mode change over WebSocket
- The frontend shows a clear "🔧 Hardware mode — connected to /dev/ttyACM0" banner with a confirmation modal before switching (high blast-radius action)
- Obstacle definition blocks are disabled in hardware mode (real obstacles come from real ToF — defining one in code is meaningless)
- Sequence runs against the real arm, real telemetry streams back, the 3D viewer continues to animate but now the source of truth is the actual joint positions reported over serial

**Safety:** all the existing reactive safety (synchronous ToF gating, MotionGuard chokepoint, IR back-away) is preserved because `HardwareBackend` is the same code path the current controller uses today. The synchronous safety memory item is honored.

**Done when:** the same Blockly program that ran in sim runs on the real Braccio with one toggle, and the autonomous sweep + obstacle avoidance behavior is identical to the current `python -m braccio_ctrl` experience.

---

## Files added / modified summary

**New (Phase 0–7):**
```
braccio_main_runner/braccio_ctrl/
  backend.py                          # ABC
  hardware_backend.py                 # adapter for existing serial code
  dsl/
    __init__.py
    grammar.lark
    parser.py
    ast_nodes.py
    interpreter.py
    compat.py

braccio_twin/                         # new top-level package
  __init__.py
  __main__.py
  urdf/braccio.urdf
  urdf/meshes/*.stl                   # vendored
  world.py
  arm.py
  tof_sim.py
  ir_sim.py
  imu_sim.py
  obstacles.py
  sim_backend.py
  pty_bridge.py

web/                                  # new top-level package
  backend/
    __init__.py
    app.py
    bridge.py
    models.py
  frontend/
    package.json
    vite.config.ts
    tsconfig.json
    index.html
    src/
      App.tsx
      main.tsx
      viewer/
        Scene.tsx
        Arm.tsx
        SensorRays.tsx
        Obstacles.tsx
      editor/
        BlocklyEditor.tsx
        blocks.ts
        dslGenerator.ts
        dslImporter.ts
        Toolbox.tsx
        RunControls.tsx
        ManualJointPanel.tsx
      state/
        telemetry.ts
        api.ts
        store.ts
```

**Modified:**
```
braccio_main_runner/braccio_ctrl/controller.py     # routes through BraccioBackend
braccio_main_runner/braccio_ctrl/sequence_editor.py # uses dsl/ module
braccio_main_runner/braccio_ctrl/__main__.py       # --backend hardware|sim flag
braccio_main_runner/requirements.txt               # add pybullet, lark, fastapi, uvicorn, pydantic, websockets
pyproject.toml                                     # add new packages
```

---

## Decision points I locked in (override any of these and I'll re-plan)

1. **PyBullet** for the Python sim engine — not MuJoCo, Gazebo, or Webots
2. **Existing Braccio URDF** vendored into the repo — not hand-built from raw STL (would burn days)
3. **Three.js + react-three-fiber** in the browser — not Babylon.js or raw WebGL
4. **FastAPI + Vite + React + TypeScript** — not Flask + vanilla JS, not Svelte
5. **Google Blockly** for the block editor — not Scratch Blocks fork or custom
6. **Lark** for the DSL grammar — not ANTLR or PLY
7. **MediaRecorder + canvas.captureStream** in the browser for MP4 — not a Python ffmpeg pipeline
8. **PTY virtual serial** as a *drop-in* mode for the sim — so legacy `python -m braccio_ctrl` keeps working unchanged
9. **`BraccioBackend` ABC** as the seam between control logic and hardware/sim — instead of forking the controller into two branches
10. **Blocks compile to the extended DSL text** (not directly to Python) — so block programs round-trip with the existing curses sequence editor and so the DSL stays the single source of truth

---

## Phasing recap

| Phase | Deliverable | Standalone value |
|---|---|---|
| 0 | `BraccioBackend` ABC + `HardwareBackend` adapter | Refactor only — no behavior change |
| 1 | PyBullet digital twin + `--pty` mode | Run existing controller against a virtual arm |
| 2 | Lark-based extended DSL with conditionals/loops/state-defs/obstacle-defs | Power users can write richer sequences in the curses editor |
| 3 | FastAPI server + WebSocket telemetry | Programmatic web API for the arm/sim |
| 4 | Three.js arm viewer | Watch the arm in 3D in any browser |
| 5 | Blockly block-based programming editor | Kids program the arm with drag-and-drop |
| 6 | Live MP4 export from the renderer canvas | Kids share their algorithms as videos |
| 7 | One-toggle deploy to physical hardware | Kids see their program run on the real arm |

You can stop after any phase and the system is in a working, demoable state.

---

## What I am NOT doing (out of scope unless you say otherwise)

- Hosting the web app online — assume `localhost` for now
- User accounts / multi-user editing
- Cloud storage for sequences — local filesystem only
- Mobile / tablet UI optimization (block editor on a phone is a separate UX problem)
- ROS / MoveIt integration
- Photorealistic rendering — Three.js with PBR materials is "good enough", not Isaac Sim quality
- Replacing the curses TUI — it stays as the power-user interface
- Magnetometer fusion / true yaw on the IMU (separate item already in the project notes)

---

## Open questions I'd like your call on before I start

1. **URDF source preference?** I plan to vendor `lots-of-things/braccio_arduino_ros_rviz` (MIT-licensed, looks accurate). If you'd rather I pull from a specific repo, or build from a specific STL you already have, name it.
2. **Where do the kids run this?** Localhost-only on the lab machine, or do you want it deployable to a school network? (Affects auth + hosting story.)
3. **Curses sequence editor — keep, retire, or read-only?** I plan to keep it working in parallel with the web editor via the shared DSL. Confirm.
4. **Phase 1 first, then re-evaluate?** This is a large plan. I recommend I ship Phase 0 + Phase 1 first (the digital twin alone is genuinely useful) and we re-scope phases 2–7 once you've actually used the sim. Confirm or push back.
