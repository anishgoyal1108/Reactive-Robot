#!/usr/bin/env python3
"""Compatibility launcher.

The side-to-side obstacle test is now integrated in the main controller.
Use `R` in the UI to toggle the profile.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from braccio_ctrl.__main__ import main


if __name__ == "__main__":
    print("[INFO] Side-to-side profile is integrated in run_braccio. Press R to toggle it.")
    main()
