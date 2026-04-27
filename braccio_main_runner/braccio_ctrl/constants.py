"""Configuration constants for Braccio host control."""

from __future__ import annotations

# Link lengths (mm)
L1 = 125.0
L2 = 125.0
L3 = 60.0

# Joint indices
JOINT_BASE = 0
JOINT_SHOULDER = 1
JOINT_ELBOW = 2
JOINT_WRIST_VERT = 3
JOINT_WRIST_ROT = 4
JOINT_GRIPPER = 5

# Joint limits (deg)
JOINT_LIMITS = [
    (0, 180),
    (15, 165),
    (0, 180),
    (0, 180),
    (0, 180),
    (10, 73),
]

JOINT_TOKENS = ["B", "S", "E", "WV", "WR", "G"]
JOINT_NAMES = ["Base  ", "Shoulder", "Elbow   ", "WristV  ", "WristR  ", "Gripper "]

HOME_POS = [90, 90, 90, 90, 90, 73]

# Serial config
DEFAULT_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.1
MEGA_PROTOCOL_V1 = True
MEGA_CMD_TIMEOUT_MS = 750
MEGA_HEARTBEAT_MS = 250

# Default IK/polar state
DEFAULT_THETA = 90.0
DEFAULT_R = 152.0
DEFAULT_Z = 20.0

# Reach limits
R_MIN = 10.0
R_MAX = 240.0
Z_MIN = 0.0
Z_MAX = 200.0

# Keyboard step sizes
THETA_STEP = 5.0
R_STEP = 10.0
Z_STEP = 10.0
WRIST_V_STEP = 5.0
WRIST_R_STEP = 5.0

# Gripper
GRIPPER_OPEN = 10
GRIPPER_CLOSE = 73
GRIPPER_GENTLE = 55

# Slew rate
DELTA_MIN = 1
DELTA_MAX = 5
DELTA_DEFAULT = 1

# Runtime efficiency targets
CONTROL_LOOP_HZ = 25.0
DISPLAY_LOOP_HZ = 8.0

# Side-to-side obstacle test profile defaults (integrated into main UI)
PROFILE_SWEEP_R_MM = 112.0
PROFILE_SWEEP_Z_MM = 20.0
PROFILE_SWEEP_WRIST_OFFSET_DEG = 0.0
PROFILE_SWEEP_WRIST_ROT_DEG = 90
PROFILE_SWEEP_GRIPPER_DEG = 73
PROFILE_SWEEP_STEP_DEG = 1.5
PROFILE_SWEEP_DWELL_S = 0.8
PROFILE_SWEEP_THETA_MIN_DEG = 5.0
PROFILE_SWEEP_THETA_MAX_DEG = 175.0

# Plotter (kept for optional use)
JOINT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
]
PLOT_WINDOW_S = 60.0
PLOT_SAMPLE_HZ = 20.0
PLOT_UPDATE_HZ = 10.0
PLOT_KEY_RESET = "u"
PLOT_KEY_SCREENSHOT = "y"
PLOT_KEY_LOG = "t"
PLOT_KEY_MAIN_TOGGLE = "0"
PLOT_Y_MARGIN_DEG = 10
LOG_DIR = "logs"
SCREENSHOT_DIR = "screenshots"

# ToF settings
# Runtime defaults to the 4-sensor Hyperion sketch. The host switches the
# Teensy between ACT 2 during normal detection and ACT 4 during confirmation
# / avoidance tracking.
TOF_NUM_CHANNELS = 4
TOF_CHANNELS_DEFAULT = 4
TOF_CHANNELS_STANDARD_MAX = 2
TOF_CHANNELS_HYPERION_DEFAULT = 4
TOF_CHANNELS_MAX = 4
TOF_DEFAULT_PORT = "/dev/ttyACM1"
TOF_BAUD_RATE = 115200
TOF_THRESHOLD_MM = 250.0
TOF_FRESHNESS_S = 0.4
OBSTACLE_DECAY_S_DEFAULT = 5.0
TOF_UPSAMPLE_N = 40
TOF_SURFACE_EVERY = 5
TOF_PLOT_INTERVAL_MS = 100
TOF_MAX_RANGE_MM = 3000.0

OBSTACLE_CONFIRMATION_HITS = 3
OBSTACLE_CONFIRMATION_FRAMES = 5
OBSTACLE_CANDIDATE_TIMEOUT_S = 1.0
OBSTACLE_TRACK_GATE_MM = 90.0
OBSTACLE_TRACK_MAX_MISSES = 5
OBSTACLE_TRACK_ALPHA = 0.45
OBSTACLE_TRACK_DISAGREE_MM = 140.0
Z_SCAN_STEP_MM = 20.0
Z_SCAN_MAX_STEPS = 8
Z_SCAN_CLEARANCE_SCALE = 1.2
OPTIMIZER_STUCK_CYCLES = 6
OPTIMIZER_PROGRESS_EPS = 0.35

# IR mapping
IR_LABEL_MAP = {0: "CLEAR", 1: "FAR", 2: "CLOSE", 3: "DANGER"}

# Key bindings
KEY_BINDINGS = {
    ord("a"): "theta_dec", ord("A"): "theta_dec",
    ord("d"): "theta_inc", ord("D"): "theta_inc",
    ord("w"): "r_inc", ord("W"): "r_inc",
    ord("s"): "r_dec", ord("S"): "r_dec",
    ord("q"): "z_inc", ord("Q"): "z_inc",
    ord("e"): "z_dec", ord("E"): "z_dec",
    ord("i"): "wv_inc", ord("I"): "wv_inc",
    ord("k"): "wv_dec", ord("K"): "wv_dec",
    ord("j"): "wr_dec", ord("J"): "wr_dec",
    ord("l"): "wr_inc", ord("L"): "wr_inc",
    ord("o"): "grip_open", ord("O"): "grip_open",
    ord("["): "grip_open",
    ord("p"): "grip_close", ord("P"): "grip_close",
    ord("]"): "grip_close",
    ord("+"): "delta_inc", ord("="): "delta_inc",
    ord("-"): "delta_dec", ord("_"): "delta_dec",
    ord("h"): "go_home",
    ord("H"): "set_equil",
    # ToF/monitoring controls
    ord("v"): "tof_view_toggle", ord("V"): "tof_view_toggle",
    ord("n"): "tof_screenshot", ord("N"): "tof_screenshot",
    ord("f"): "tof_threshold_inc",
    ord("F"): "tof_threshold_dec",

    # Main runtime controls
    ord("b"): "record_toggle", ord("B"): "record_toggle",
    ord("r"): "profile_side_toggle", ord("R"): "profile_side_toggle",
    ord("c"): "imu_calibrate", ord("C"): "imu_calibrate",

    27: "quit",  # ESC
}

# IMU filtering
# Exponential LPF: y_k = alpha*x_k + (1-alpha)*y_{k-1}
# Lower alpha => stronger smoothing.
IMU_LPF_ALPHA_DEFAULT = 0.25

# Obstacle memory behavior
PERSISTENT_MEMORY_DEFAULT = False
PERSISTENT_CLEAR_REQUIRED = 4

# Future planning preview for committed waypoint visualization
PLANNING_HORIZON_S = 5.0
PLANNING_PREVIEW_DT_S = 0.2

TIMING_LOG_PATH = "logs/end_to_end_timing.jsonl"


