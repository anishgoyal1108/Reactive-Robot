"""
constants.py — All configuration values in one place.

Edit link lengths here if your Braccio build differs from the standard:
  L1 = shoulder pivot (M2) → elbow pivot (M3)
  L2 = elbow pivot   (M3) → wrist pivot (M4)
  L3 = wrist pivot   (M4) → gripper tip
"""

import curses

# ── Link lengths (mm) ─────────────────────────────────────────────────────
L1 = 125.0
L2 = 125.0
L3 = 60.0

# ── Joint indices ─────────────────────────────────────────────────────────
JOINT_BASE       = 0
JOINT_SHOULDER   = 1
JOINT_ELBOW      = 2
JOINT_WRIST_VERT = 3
JOINT_WRIST_ROT  = 4
JOINT_GRIPPER    = 5

# ── Joint limits: (min_deg, max_deg) ─────────────────────────────────────
JOINT_LIMITS = [
    (0,   180),   # Base
    (15,  165),   # Shoulder
    (0,   180),   # Elbow
    (0,   180),   # Wrist Vertical
    (0,   180),   # Wrist Rotation
    (10,   73),   # Gripper
]

# ── Protocol tokens (must match Arduino's JOINT_TOKEN[]) ──────────────────
JOINT_TOKENS = ['B', 'S', 'E', 'WV', 'WR', 'G']

# ── Joint display names ───────────────────────────────────────────────────
JOINT_NAMES = ['Base  ', 'Shoulder', 'Elbow   ', 'WristV  ', 'WristR  ', 'Gripper ']

# ── Home / equilibrium position (deg) ────────────────────────────────────
HOME_POS = [90, 90, 90, 90, 90, 73]

# ── Serial config ─────────────────────────────────────────────────────────
DEFAULT_PORT    = '/dev/ttyACM0'
BAUD_RATE       = 115200
SERIAL_TIMEOUT  = 0.1   # seconds for readline timeout

# ── Default IK state ──────────────────────────────────────────────────────
DEFAULT_THETA = 90.0    # degrees (arm pointing straight forward)
DEFAULT_R     = 152.0   # mm (~6 inches)
DEFAULT_Z     = -50.0   # mm below shoulder pivot

# ── IK reach limits ───────────────────────────────────────────────────────
R_MIN = 10.0
R_MAX = 240.0   # L1 + L2 - L3 = 190mm effective, but allow up to full extension
Z_MIN = -250.0
Z_MAX = 200.0

# ── Keyboard step sizes ───────────────────────────────────────────────────
THETA_STEP   =  5.0   # deg per keypress
R_STEP       = 10.0   # mm per keypress
Z_STEP       = 10.0   # mm per keypress
WRIST_V_STEP =  5.0   # deg per keypress (wrist offset)
WRIST_R_STEP =  5.0   # deg per keypress (wrist rotation)

# ── Gripper positions ─────────────────────────────────────────────────────
GRIPPER_OPEN  = 10    # fully open
GRIPPER_CLOSE = 73    # fully closed
GRIPPER_GENTLE = 55   # gentle grip (won't crush a pen)

# ── Slew rate constraints (deg/tick at 100 Hz) ────────────────────────────
DELTA_MIN     = 1
DELTA_MAX     = 5
DELTA_DEFAULT = 1

# ── Plotter ───────────────────────────────────────────────────────────────
JOINT_COLORS = [
    'tab:blue',   # Base
    'tab:orange', # Shoulder
    'tab:green',  # Elbow
    'tab:red',    # Wrist Vertical
    'tab:purple', # Wrist Rotation
    'tab:brown',  # Gripper
]
PLOT_WINDOW_S  = 60.0   # seconds of history visible at once (scrolling window)
PLOT_SAMPLE_HZ = 20.0   # sampler thread rate (Hz)
PLOT_UPDATE_HZ = 10.0   # FuncAnimation redraw rate (Hz)
PLOT_KEY_RESET      = 'u'   # reset plot & restart t=0
PLOT_KEY_SCREENSHOT = 'y'   # save PNG screenshot
PLOT_KEY_LOG        = 't'   # toggle CSV logging
PLOT_KEY_MAIN_TOGGLE = '0'  # hide/show main combined window
# keys '1'–'6' toggle individual per-joint pop-out windows (hardcoded in plotter)
PLOT_Y_MARGIN_DEG   = 10    # extra degrees above/below joint limits on y-axis
LOG_DIR        = 'logs'
SCREENSHOT_DIR = 'screenshots'
SESSION_LOG_DIR = 'session_logs'   # JSONL telemetry for offline review
SESSION_LOG_HZ  = 10               # telemetry sample rate (Hz)

# ── Data streaming (UDP to standalone plotter apps) ──────────────────────
ARM_DATA_PORT  = 9870      # joint-angle packets → arm_plotter_app.py
TOF_DATA_PORT  = 9871      # ToF/IR packets      → tof_plotter_app.py

# ── ToF sensor settings ──────────────────────────────────────────────────
TOF_NUM_CHANNELS    = 4        # number of VL53L5CX sensors on MUX
TOF_DEFAULT_PORT    = '/dev/ttyACM1'   # Teensy port (separate from Braccio)
TOF_BAUD_RATE       = 115200
# Per-channel obstacle detection thresholds [CH0, CH1, CH2, CH3] (mm)
# CH0/CH1/CH2 are primary authority sensors — any reading below the channel's
# threshold triggers REPLAN and gates manual commands via MotionGuard.
# CH3 (bottom, ground-facing) is ignored entirely — it always reads the floor.
# A hardware test on 2026-04-10 showed CH2's old 50mm threshold was too tight:
# the user's hand held next to CH2 produced readings in the 41-100mm band,
# most of which were silently filtered out. Matching CH0/CH1 at 250mm catches
# the full detection range and still leaves ~98% of idle frames above threshold.
# 2026-04-17: CH1/CH2 bumped 100→250 (accuracy over speed) — at 100 mm the
# sides were reacting too late and producing the escape_collision jitter in
# session_20260417_014307.
TOF_THRESHOLDS_MM   = [100.0, 250.0, 250.0, 50.0]
TOF_THRESHOLD_MM    = TOF_THRESHOLDS_MM[0]   # legacy fallback alias
TOF_UPSAMPLE_N     = 40       # bilinear upsample resolution for plotting
TOF_SURFACE_EVERY  = 5        # redraw 3D surface every N frames (performance)
TOF_PLOT_INTERVAL_MS = 100    # animation timer interval
TOF_MAX_RANGE_MM   = 3000.0   # max display range for heatmap colorbar

# ── IR sensor settings (OUT1D, 2-bit) ───────────────────────────────────
# Bits: 00=CLEAR  01=FAR  10=CLOSE  11=DANGER
IR_LABEL_MAP = {0: 'CLEAR', 1: 'FAR', 2: 'CLOSE', 3: 'DANGER'}

# Debounce counts for IR state transitions. The IR sensors are noisy when
# wires are loose and can flap every frame — without debouncing each flap
# becomes a new BACK_AWAY command, causing the arm to jerk in place.
IR_DEBOUNCE_ON_SAMPLES     = 3        # need N consecutive high samples to trigger
IR_DEBOUNCE_OFF_SAMPLES    = 5        # need N consecutive clear samples to release

# Minimum hold time (seconds) for an obstacle_response after it fires. Once
# the arm commits to REPLAN or BACK_AWAY, it holds that decision for at least
# this long even if the sensor reading flickers clear momentarily. Keeps the
# arm from oscillating when a ToF reading sits right on the threshold.
OBSTACLE_HOLD_S            = 0.5

# ── Safety stack tuning (see braccio_ctrl/safety/) ────────────────────────
# The obstacle-avoidance architecture is point-cloud + KDTree + capsule
# collision checking + BiRRT, orchestrated by a py_trees behavior tree.
# See research.md / research_output.pdf + ~/.claude/plans/calm-sprouting-shell.md
# for the design rationale; every constant here maps to a section of that
# document.
#
# ToF hysteresis — Schmitt trigger engage / disengage (§7). Per-channel
# deadband. CH0 (front) uses a 100/180 window tight enough to avoid
# false-positives on the arm's own structure. CH1/CH2 (side-facing) were
# bumped 100/180 → 250/350 on 2026-04-17 after session_20260417_014307 showed
# the sides reacting too late, producing escape_collision jitter. CH3 is
# ignored (ground).
TOF_ENGAGE_MM            = [100.0, 250.0, 250.0,  50.0]
TOF_DISENGAGE_MM         = [180.0, 350.0, 350.0, 100.0]

# Joint command smoothing — EMA α (§7, DeXtreme default).
# Lowered 0.2 → 0.15 on 2026-04-17 for a smoother spline (accuracy over
# speed); more lag, less per-tick chatter.
JOINT_COMMAND_EMA_ALPHA  = 0.15
# Backstop rate clamp applied after EMA (deg per BT tick). Set to match
# SWEEP_STEP_DEG so normal sweep motion runs at full speed while big replan
# jumps (escape_collision hit 49° in a single 100 ms tick in
# session_20260417_014307) get spread over many ticks — this is the
# "replanning window" that makes corrective motion smooth.
JOINT_RATE_MAX_DEG       = 5.0

# Capsule clearance — margin added to each link's capsule radius before the
# collision check. Swallows FK rounding, commanded-vs-actual servo drift,
# and ToF noise.
CAPSULE_CLEARANCE_MM     = 30.0

# Projected ToF points closer than (capsule_radius + this) to any arm link
# are dropped at ingest — prevents the arm's own structure from being logged
# as a permanent obstacle that deadlocks the planner. Kept small (5 mm) so
# real obstacles just outside the arm surface (e.g. a finger 20–30 mm from
# the wrist) aren't filtered as self-reflections; the old 40 mm margin was
# eating every close reading the ToF sensors produced.
TOF_SELF_FILTER_MARGIN_MM = 5.0

# Per-channel self-filter margin overrides. CH3 (bottom sensor) sits
# above the TCA9548A MUX breakout board; when the wrist tilts forward
# past ~30° the bottom sensor starts seeing the MUX/PCB/wiring at
# ~40-80 mm and mis-reads those returns as obstacles, which drives the
# BT into permanent escape_collision + edge_reverse ping-pong. Widen
# CH3's margin so the mount-block volume gets filtered as "self", not
# "obstacle". CH0/CH1/CH2 keep the tight default margin.
TOF_SELF_FILTER_MARGIN_PER_CHANNEL_MM: tuple[float, float, float, float] = (
    5.0,   # CH0 top    — open sky
    5.0,   # CH1 right  — side
    5.0,   # CH2 left   — side
    90.0,  # CH3 bottom — sees MUX board + PCB + wiring on tilt
)

# Physical MUX channel → software channel remap. The Teensy firmware
# selects TCA9548A channels 0-3 in hardware order and labels them S0-S3
# on serial, but the wiring actually pairs them:
#   MUX 0 ↔ LEFT sensor   → software CH2
#   MUX 1 ↔ RIGHT sensor  → software CH1
#   MUX 2 ↔ TOP sensor    → software CH0
#   MUX 3 ↔ BOTTOM sensor → software CH3
# Applied once in tof_sensor._parse_tf so every downstream consumer
# (display, obstacle decision, safety ingest) sees a consistent
# "CH0=Top, CH1=Right, CH2=Left, CH3=Bottom" convention.
TOF_MUX_TO_CHANNEL = {0: 2, 1: 1, 2: 0, 3: 3}

# Synthetic distance (mm) injected when a ToF frame is fully masked
# (every zone invalid but frames are arriving). Sits just below the
# engage threshold so the Schmitt trigger fires immediately.
TOF_MASKED_SYNTHETIC_MM = 60.0

# VL53L5CX target_status codes that unambiguously indicate a close-
# obstacle / saturation scenario. IMPORTANT: codes 5 and 9 are the
# *normal valid* codes per the SparkFun / ST datasheet ("5 & 9 means
# ranging OK") — do NOT include them here. We only promote on codes
# that clearly mean "sensor is saturated or seeing something too close
# to measure". Conservatively empty by default; the masked-frame
# detector (all zones invalid) covers the hand-over-sensor case without
# needing per-cell status heuristics.
TOF_CLOSE_STATUS_CODES: tuple[int, ...] = ()

# BiRRT fallback budget for the "go around the obstacle" sweep escape hatch
# (safety/behavior.py). Fires only when the z-ladder is fully exhausted; the
# resulting path is cached in the blackboard so follow-up ticks stream
# waypoints at ~3 ms each instead of replanning.
SWEEP_BIRRT_FALLBACK_TIMEOUT_S = 0.8

# BiRRT tuning (§2).
BIRRT_TIMEOUT_S          = 2.0
BIRRT_EXTEND_DEG         = 5.0

# WorldModel — point-cloud aging and KDTree rebuild cadence (§3).
WORLD_MAX_AGE_S          = 15.0
WORLD_KDTREE_REBUILD_S   = 0.5

# ── Simple replan sweep (demo-safe mode) ─────────────────────────────────
# When True, the sweep branch uses a minimal reactive state machine: on a
# ToF hit, parametrise a smooth radial-pullback detour from current θ to
# the first clear θ past the obstacle (same z-line, same direction) and
# stream it one waypoint per BT tick. Only reverses direction when the
# clear-side θ would fall outside the [0, 180] sweep domain. Set False to
# re-enable the full escalation ladder (polar skip → z-ladder → BiRRT →
# direction flip).
SIMPLE_REPLAN_MODE       = True
# Back-compat alias — older tests and session logs reference the prior
# name. Kept as an alias so a flip of SIMPLE_REPLAN_MODE also flips the
# legacy reads; monkeypatching either one in tests still works because the
# sweep branch reads SIMPLE_REPLAN_MODE via ``getattr(_c, ...)`` at runtime.
SIMPLE_SWEEP_MODE        = SIMPLE_REPLAN_MODE
# Cool-down after a reactive reverse before the sweep allows another
# direction flip. Prevents jitter when a sensor straddles its threshold.
SIMPLE_SWEEP_REVERSE_COOLDOWN_S = 0.4

# Angular-detour shape knobs (see _SweepTick._build_detour_path). The
# detour is a smoothstep in θ with a sine-shaped radial dip at midpoint
# so the arm curves AROUND an obstacle while keeping z on the sweep's
# z-line.
SWEEP_DETOUR_STEPS       = 6        # waypoints per detour (rate-clamp fills in)
SWEEP_DETOUR_R_DIP_MM    = 40.0     # nominal radial pull-in at midpoint
SWEEP_DETOUR_R_MIN_MM    = 60.0     # IK inner-cone guard; pull-in cannot go below this
SWEEP_DETOUR_CHAIN_MAX   = 3        # adjacent-block chain-skip iterations

# Sweep configuration (§7).
SWEEP_R_DEFAULT_MM       = 152.0
SWEEP_Z_DEFAULT_MM       = 35.0
SWEEP_STEP_DEG           = 5.0     # full-speed sweep — smoothness comes from the RateClamp in SafetyAPI._handle_command, not from throttling the step
SWEEP_PRE_SCAN_WAYPOINTS = 5
SWEEP_SKIP_MARGIN_DEG    = 10.0
SWEEP_FOV_HALF_DEG       = 22.5    # VL53L5CX per-axis FoV half-angle
# Alternate Z levels tried when the current slice is entirely blocked.
# The sweep branch walks these in order, sticking with the first that has
# a reachable unblocked angle in the direction of travel.
SWEEP_Z_LADDER_MM        = [35.0, 60.0, 90.0, 10.0, -20.0]

# Sequence replanner (§6).
SEQUENCE_MAX_RETRIES     = 3

# Behavior tree tick rate. 20 Hz keeps sweep + manual-hold responsive; each
# plan is still <50 ms on commodity CPUs so the tick budget is comfortable.
# Per-tick smoothness is enforced by the RateClamp in the command path, not
# by lowering the tick rate (lower tick rate made the sweep feel sluggish).
BT_TICK_HZ               = 20.0

# ── User experience toggles ───────────────────────────────────────────────
# Strict mode: when False (default) every safety refusal is handled
# silently — the manual branch just doesn't advance the arm and the user
# can press the key again. When True, a curses refusal dialog opens with
# Retreat / Axis / Step / Override / Cancel choices bound to F1–F5.
STRICT_MODE_DEFAULT      = False
# Absolute path to the sound file played on every obstacle-triggered
# refusal / replan. Blank string disables the sound.
import os as _os
OBSTACLE_SOUND_PATH      = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "completion-fail.oga"
)
OBSTACLE_SOUND_PATH      = _os.path.abspath(OBSTACLE_SOUND_PATH)
# Minimum seconds between successive sound plays (rate-limit).
OBSTACLE_SOUND_COOLDOWN_S = 1.5

# ── Key bindings: curses key code → action string ─────────────────────────
# fmt: off
KEY_BINDINGS = {
    ord('a'): 'theta_dec',    ord('A'): 'theta_dec',
    ord('d'): 'theta_inc',    ord('D'): 'theta_inc',
    ord('w'): 'r_inc',        ord('W'): 'r_inc',
    ord('s'): 'r_dec',        ord('S'): 'r_dec',
    ord('q'): 'z_inc',        ord('Q'): 'z_inc',
    ord('e'): 'z_dec',        ord('E'): 'z_dec',
    ord('i'): 'wv_inc',       ord('I'): 'wv_inc',
    ord('k'): 'wv_dec',       ord('K'): 'wv_dec',
    ord('j'): 'wr_dec',       ord('J'): 'wr_dec',
    ord('l'): 'wr_inc',       ord('L'): 'wr_inc',
    ord('o'): 'grip_open',    ord('O'): 'grip_open',
    ord('['): 'grip_open',
    ord('p'): 'grip_close',   ord('P'): 'grip_close',
    ord(']'): 'grip_close',
    ord('+'): 'delta_inc',    ord('='): 'delta_inc',
    ord('-'): 'delta_dec',    ord('_'): 'delta_dec',
    ord('h'): 'go_home',
    ord('H'): 'set_equil',    # Shift+H
    ord('m'): 'states_menu',  # States menu
    ord('M'): 'states_menu',
    ord('x'): 'seq_editor',   # Sequence editor
    ord('X'): 'seq_editor',
    # ── Sweep / IMU controls ──────────────────────────────────────────────
    ord('z'): 'sweep_toggle',
    ord('Z'): 'sweep_toggle',
    ord('c'): 'imu_calibrate',
    ord('C'): 'imu_calibrate',
    # ── Strict-mode toggle (safety dialog on/off) ────────────────────────
    # Pressing 't' flips between silent auto-replan (default) and the
    # strict refusal-dialog mode. Bound away from the legacy IK keys so
    # nothing conflicts.
    ord('t'): 'strict_toggle',
    ord('T'): 'strict_toggle',
    # ── Arrow-key manual-IK alternatives ─────────────────────────────────
    # Sit alongside WASD so the user can hold keys without those WASD
    # positions conflicting with the strict-mode refusal-dialog choices.
    # Terminals usually emit KEY_LEFT/RIGHT etc. for numpad with NumLock
    # off, so the numpad still works via these bindings.
    curses.KEY_LEFT:  'theta_dec',
    curses.KEY_RIGHT: 'theta_inc',
    curses.KEY_UP:    'r_inc',
    curses.KEY_DOWN:  'r_dec',
    curses.KEY_PPAGE: 'z_inc',      # PgUp / numpad 9 (NumLock off)
    curses.KEY_NPAGE: 'z_dec',      # PgDn / numpad 3 (NumLock off)
    # ── Refusal-dialog function keys (only consumed in strict mode) ──────
    # Using F1–F5 means these never collide with any manual key.
    curses.KEY_F1: 'dialog_retreat',
    curses.KEY_F2: 'dialog_axis',
    curses.KEY_F3: 'dialog_step',
    curses.KEY_F4: 'dialog_override',
    curses.KEY_F5: 'dialog_cancel',
    27:       'quit',          # ESC
}
# fmt: on
