"""
braccio_ctrl — Modular keyboard controller for the Tinkerkit Braccio arm.

Usage
-----
  python -m braccio_ctrl [port]                            # default port: /dev/ttyACM0
  python -m braccio_ctrl --list-ports                      # list available serial ports
  python run_braccio.py [port]                             # standalone launcher
  python -m braccio_ctrl [port] --teensy-port /dev/ttyACM1 # with ToF/IR sensing

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
  tof_plotter     — real-time 4-channel ToF heatmap + 3D surface viewer
"""

from .controller import BraccioController
from .tof_sensor import ToFState, ToFBridge, ObstacleResponse

__all__ = ['BraccioController', 'ToFState', 'ToFBridge', 'ObstacleResponse']
