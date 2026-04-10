"""
__main__.py — CLI entry point.

  python -m braccio_ctrl [port] [--baud N] [--list-ports]
                         [--teensy-port PORT] [--no-tof]

Live matplotlib plots are provided by standalone companion scripts
(arm_plotter_app.py, tof_plotter_app.py) that receive data over UDP.
"""

import argparse
import sys

from .serial_bridge import SerialBridge, ensure_serial_device_path
from .controller import BraccioController
from .constants import (
    DEFAULT_PORT,
    BAUD_RATE,
    TOF_DEFAULT_PORT,
    TOF_BAUD_RATE,
    ARM_DATA_PORT,
    TOF_DATA_PORT,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m braccio_ctrl",
        description="Braccio Arm IK Keyboard Controller + ToF/IR Sensing",
    )
    parser.add_argument(
        "port",
        nargs="?",
        default=DEFAULT_PORT,
        help=(
            f"USB serial device for Braccio arm, e.g. /dev/ttyACM0 (default: "
            f"{DEFAULT_PORT}). Not a number — {TOF_DATA_PORT} is UDP for the "
            f"plotter, not this argument."
        ),
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help=f"Baud rate for Braccio (default: {BAUD_RATE})",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit",
    )
    # ── ToF / IR sensor arguments ────────────────────────────────────────
    parser.add_argument(
        "--teensy-port",
        default=None,
        help=f"Serial port for ToF/IR Teensy (default: {TOF_DEFAULT_PORT}). "
        f'Pass "auto" to use {TOF_DEFAULT_PORT}.',
    )
    parser.add_argument(
        "--teensy-baud",
        type=int,
        default=TOF_BAUD_RATE,
        help=f"Baud rate for Teensy (default: {TOF_BAUD_RATE})",
    )
    parser.add_argument(
        "--no-tof",
        action="store_true",
        help="Disable ToF/IR sensors entirely",
    )
    args = parser.parse_args()

    try:
        ensure_serial_device_path(args.port, "Braccio arm serial (first argument)")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    if args.list_ports:
        ports = SerialBridge.list_ports()
        if not ports:
            print("No serial ports found.")
        else:
            print("Available serial ports:")
            for device, desc in ports:
                print(f"  {device:<20s}  {desc}")
        sys.exit(0)

    # Resolve Teensy port
    teensy_port = None
    if not args.no_tof:
        if args.teensy_port == "auto":
            teensy_port = TOF_DEFAULT_PORT
        elif args.teensy_port:
            teensy_port = str(args.teensy_port)
            try:
                ensure_serial_device_path(teensy_port, "Teensy serial (--teensy-port)")
            except ValueError as exc:
                print(exc, file=sys.stderr)
                sys.exit(2)

    print(f"Connecting to Braccio on {args.port} at {args.baud} baud...")
    if teensy_port:
        print(
            f"Connecting to ToF/IR Teensy on {teensy_port} at {args.teensy_baud} baud..."
        )
    else:
        print("ToF/IR sensors: disabled (pass --teensy-port to enable)")
    print(
        f"Streaming data → UDP localhost:{ARM_DATA_PORT} (arm), "
        f":{TOF_DATA_PORT} (tof)"
    )
    print("Press ESC to quit.\n")

    ctrl = BraccioController(
        args.port,
        args.baud,
        teensy_port=str(teensy_port),
        teensy_baud=args.teensy_baud,
    )

    try:
        ctrl.run()
    except KeyboardInterrupt:
        pass

    print("\nController stopped.")


if __name__ == "__main__":
    main()
