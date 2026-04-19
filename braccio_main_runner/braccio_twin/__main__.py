"""
__main__.py — braccio_twin CLI entry point.

    python -m braccio_twin [--gui] [--world world.json] [--no-ir]

Runs the same BraccioController the real arm uses, but plugged into a
PyBullet digital twin instead of a USB Arduino. Every existing keybinding
works: 'Z' toggles the sweep, 'A/D' drive theta, the states menu and
sequence editor all function unchanged.

Use `--gui` to open the PyBullet debug window so you can see the arm
move; otherwise the sim runs headless behind the curses TUI. The
`--world` flag loads an ObstacleWorld JSON file before the controller
starts, which is how Phase 5's web editor will stage scenes.
"""

from __future__ import annotations

import argparse
import sys

from braccio_ctrl.constants import (
    BAUD_RATE,
    TOF_BAUD_RATE,
    TOF_NUM_CHANNELS,
    TOF_THRESHOLDS_MM,
)
from braccio_ctrl.controller import BraccioController
from braccio_ctrl.imu_state import IMUState
from braccio_ctrl.tof_sensor import ToFState

from .sim_backend import SimBackend, SimSensorBridge


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m braccio_twin",
        description=(
            "Run the Braccio controller against a PyBullet digital twin "
            "instead of real hardware."
        ),
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a PyBullet GUI window (requires a graphical session)",
    )
    parser.add_argument(
        "--world",
        default=None,
        help="Path to an ObstacleWorld JSON file to load at startup",
    )
    parser.add_argument(
        "--no-ir",
        action="store_true",
        help="Disable the virtual IR sensors (debug aid)",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="Override the built-in Braccio URDF path",
    )
    args = parser.parse_args()

    # Build the sim backend first so we can attach sensor state before
    # we hand it to the controller.
    sim_backend = SimBackend(gui=args.gui, urdf_path=args.urdf)
    tof_state = ToFState(num_channels=TOF_NUM_CHANNELS)
    tof_state.tof_threshold_mm = TOF_THRESHOLDS_MM[0]
    tof_state.tof_thresholds_mm = list(TOF_THRESHOLDS_MM)
    imu_state = IMUState()
    sim_backend.install_sensor_state(tof_state, imu_state)

    sensor_bridge = SimSensorBridge(sim_backend)

    ctrl = BraccioController(
        port="pybullet://sim",
        baud=BAUD_RATE,
        teensy_port="pybullet://sim",
        teensy_baud=TOF_BAUD_RATE,
        enable_ir=not args.no_ir,
        arm_backend=sim_backend,
        sensor_backend=sensor_bridge,
        tof_state=tof_state,
        imu_state=imu_state,
    )

    # Pre-load obstacles if requested — has to happen after
    # sim_backend.connect() so the PyBullet world exists. The controller
    # calls backend.connect() inside curses_main, so we need to defer
    # world loading. Easiest: monkey-patch the sim_backend.connect to
    # also load the JSON after the world is up.
    if args.world:
        original_connect = sim_backend.connect
        world_path = args.world

        def _connect_and_load() -> bool:
            ok = original_connect()
            if ok:
                world = sim_backend.get_obstacle_world()
                if world is not None:
                    try:
                        n = world.load(world_path)
                        print(
                            f"[braccio_twin] loaded {n} obstacles from "
                            f"{world_path}",
                            file=sys.stderr,
                        )
                    except Exception as exc:
                        print(
                            f"[braccio_twin] failed to load {world_path}: "
                            f"{exc}",
                            file=sys.stderr,
                        )
            return ok

        sim_backend.connect = _connect_and_load  # type: ignore[method-assign]

    mode = "GUI" if args.gui else "headless"
    print(f"[braccio_twin] starting PyBullet ({mode})...", file=sys.stderr)
    print("[braccio_twin] press ESC in the curses UI to quit.", file=sys.stderr)

    try:
        ctrl.run()
    except KeyboardInterrupt:
        pass

    sim_backend.close()
    print("[braccio_twin] stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
