#!/usr/bin/env python3
"""
run_braccio.py - Standalone launcher (no install needed).

Usage:
  python run_braccio.py                   # default port /dev/ttyACM0
  python run_braccio.py /dev/ttyUSB0
  python run_braccio.py --list-ports
  python run_braccio.py /dev/ttyACM0 --baud 115200
"""
import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from braccio_ctrl.__main__ import main

main()

