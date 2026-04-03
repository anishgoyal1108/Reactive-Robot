"""
braccio_ctrl — Modular keyboard controller for the Tinkerkit Braccio arm.

Usage
-----
  python -m braccio_ctrl [port]                            # default port: /dev/ttyACM0
  python -m braccio_ctrl --list-ports                      # list available serial ports
  python -m braccio_ctrl [port] --teensy-port /dev/ttyACM1 # with ToF/IR sensing

Live matplotlib plotters are separate processes (no threading):
  python arm_plotter_app.py          # joint-angle time series
  python tof_plotter_app.py          # ToF heatmaps + 3-D surfaces

ToF serial troubleshooting (close other programs using the port):
  python tof_serial_diagnose.py /dev/ttyACM0 --seconds 5

Modules
-------
  constants       — all configuration values and key bindings
  protocol        — serial command builders and response parser
  ik_solver       — Python IK solver (mirrors the Arduino C++ solveIK)
  arm_state       — thread-safe shared state
  serial_bridge   — serial I/O with a daemon reader thread
  keyboard_handler— curses non-blocking key → action mapping
  display         — curses terminal UI
  controller      — main control loop, integrates all modules
  tof_sensor      — Teensy serial bridge for VL53L5CX ToF + IR (OUT1D)
  data_publisher  — fire-and-forget UDP publisher for plotter apps
"""

from .controller import BraccioController
from .tof_sensor import ToFState, ToFBridge, ObstacleResponse

__all__ = ['BraccioController', 'ToFState', 'ToFBridge', 'ObstacleResponse']
