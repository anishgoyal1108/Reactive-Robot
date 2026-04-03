"""
constants.py — All configuration values in one place.

Edit link lengths here if your Braccio build differs from the standard:
  L1 = shoulder pivot (M2) → elbow pivot (M3)
  L2 = elbow pivot   (M3) → wrist pivot (M4)
  L3 = wrist pivot   (M4) → gripper tip
"""

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

# ── ToF sensor settings ──────────────────────────────────────────────────
TOF_NUM_CHANNELS    = 4        # number of VL53L5CX sensors on MUX
TOF_DEFAULT_PORT    = '/dev/ttyACM1'   # Teensy port (separate from Braccio)
TOF_BAUD_RATE       = 115200
# Per-channel obstacle detection thresholds [CH0, CH1, CH2, CH3] (mm)
# CH0 (front/side) and CH1 (back/side) are primary authority sensors.
# CH2 (top) and CH3 (bottom) use a tighter threshold but have lower authority.
TOF_THRESHOLDS_MM   = [250.0, 250.0, 50.0, 50.0]
TOF_UPSAMPLE_N     = 40       # bilinear upsample resolution for plotting
TOF_SURFACE_EVERY  = 5        # redraw 3D surface every N frames (performance)
TOF_PLOT_INTERVAL_MS = 100    # animation timer interval
TOF_MAX_RANGE_MM   = 3000.0   # max display range for heatmap colorbar

# ── IR sensor settings (OUT1D, 2-bit) ───────────────────────────────────
# Bits: 00=CLEAR  01=FAR  10=CLOSE  11=DANGER
IR_LABEL_MAP = {0: 'CLEAR', 1: 'FAR', 2: 'CLOSE', 3: 'DANGER'}

# ── Autonomous sweep config ───────────────────────────────────────────────
SWEEP_THETA_MIN            = 0.0      # degrees
SWEEP_THETA_MAX            = 180.0    # degrees
SWEEP_R_DEFAULT            = 152.0    # mm  (same as DEFAULT_R)
SWEEP_Z_DEFAULT            = 60.0    # mm above shoulder — high enough to clear table
SWEEP_STEP_DEG             = 2.0     # degrees advanced per tick
SWEEP_TICK_HZ              = 10.0    # loop rate of the sweep thread (Hz)
SWEEP_DELTA_NORMAL         = 2       # SET DELTA during normal sweep
SWEEP_DELTA_REPLAN         = 4       # SET DELTA during replanning (smoother)
SWEEP_BACK_STEPS           = 2       # steps to retreat on BACK_AWAY
SWEEP_OBSTACLE_MARGIN_DEG  = 10.0    # safety margin beyond obstacle edge (deg)
# Discrete Z levels (mm) tried during Z-axis replanning, derived from saved states.
# AutoSweeper searches above current Z first (go over), then below (go under).
SWEEP_Z_CANDIDATES         = [0.0, 20.0, 55.0, 70.0, 95.0]

# ── Obstacle map config ───────────────────────────────────────────────────
OBS_MAP_MAX_AGE_S          = 2.0     # seconds before stale cloud points are discarded
OBS_MAP_TOF_FOV_DEG        = 45.0    # VL53L5CX full horizontal/vertical FoV
OBS_MAP_GRID_SIZE          = 8       # 8×8 sensor grid

# Sensor mount offsets from end-effector, in mm [x, y, z]
# Tune these values to match the physical sensor positions on the arm
SENSOR_MOUNT_FRONT_OFFSET  = [60.0,  0.0,  0.0]
SENSOR_MOUNT_BACK_OFFSET   = [-60.0, 0.0,  0.0]
SENSOR_MOUNT_TOP_OFFSET    = [0.0,   0.0,  30.0]
SENSOR_MOUNT_BOTTOM_OFFSET = [0.0,   0.0, -30.0]

# Sensor channel authority classification
# CH0 (front/side) and CH1 (back/side): trigger REPLAN — primary authority
SENSOR_REPLAN_CHANNELS   = [0, 1]
# CH2 (top): advisory only — fires below its own threshold but does not trigger REPLAN
SENSOR_ADVISORY_CHANNELS = [2]
# CH3 (bottom): ignored — very close to ground, assume sufficient clearance
SENSOR_IGNORE_CHANNELS   = [3]

# Pre-command end-effector safety sphere radius (mm).
# A planned move is blocked if any obstacle cloud point is closer than this.
COLLISION_CHECK_RADIUS_MM = 80.0

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
    ord('0'): 'plot_main_toggle',
    ord('1'): 'plot_joint_1_toggle',
    ord('2'): 'plot_joint_2_toggle',
    ord('3'): 'plot_joint_3_toggle',
    ord('4'): 'plot_joint_4_toggle',
    ord('5'): 'plot_joint_5_toggle',
    ord('6'): 'plot_joint_6_toggle',
    ord('7'): 'plot_reset',
    ord('8'): 'plot_screenshot',
    ord('9'): 'plot_log_toggle',
    # ── ToF / IR sensor controls ─────────────────────────────────────────
    ord('v'): 'tof_view_toggle',       # V: Toggle ToF 4-channel surface viewer
    ord('V'): 'tof_view_toggle',
    ord('b'): 'tof_export_csv',        # B: Export current ToF grids to CSV
    ord('B'): 'tof_export_csv',
    ord('n'): 'tof_screenshot',        # N: Screenshot ToF plots
    ord('N'): 'tof_screenshot',
    ord('g'): 'tof_log_toggle',        # G: Toggle ToF CSV streaming log
    ord('G'): 'tof_log_toggle',
    ord('f'): 'tof_threshold_inc',     # F: Increase ToF threshold +50mm
    ord('F'): 'tof_threshold_dec',     # Shift+F: Decrease ToF threshold -50mm
    # ── Sweep / IMU controls ──────────────────────────────────────────────
    ord('z'): 'sweep_toggle',          # Z: Start/stop autonomous theta sweep
    ord('Z'): 'sweep_toggle',
    ord('c'): 'imu_calibrate',         # C: Record IMU calibration at current pose
    ord('C'): 'imu_calibrate',
    27:       'quit',          # ESC
}
# fmt: on
