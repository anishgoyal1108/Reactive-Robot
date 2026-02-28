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
    27:       'quit',          # ESC
}
# fmt: on
