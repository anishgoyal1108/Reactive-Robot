"""
__main__.py — CLI entry point.

  python -m braccio_ctrl [port] [--baud N] [--list-ports] [--no-plot]
                         [--teensy-port PORT] [--no-tof]
"""

import argparse
import sys

from .serial_bridge import SerialBridge
from .controller    import BraccioController
from .constants     import DEFAULT_PORT, BAUD_RATE, TOF_DEFAULT_PORT, TOF_BAUD_RATE


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m braccio_ctrl',
        description='Braccio Arm IK Keyboard Controller + ToF/IR Sensing',
    )
    parser.add_argument(
        'port',
        nargs='?',
        default=DEFAULT_PORT,
        help=f'Serial port for Braccio arm (default: {DEFAULT_PORT})',
    )
    parser.add_argument(
        '--baud',
        type=int,
        default=BAUD_RATE,
        help=f'Baud rate for Braccio (default: {BAUD_RATE})',
    )
    parser.add_argument(
        '--list-ports',
        action='store_true',
        help='List available serial ports and exit',
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable the real-time joint angle plotter window',
    )
    # ── ToF / IR sensor arguments ────────────────────────────────────────
    parser.add_argument(
        '--teensy-port',
        default=None,
        help=f'Serial port for ToF/IR Teensy (default: {TOF_DEFAULT_PORT}). '
             f'Pass "auto" to use {TOF_DEFAULT_PORT}.',
    )
    parser.add_argument(
        '--teensy-baud',
        type=int,
        default=TOF_BAUD_RATE,
        help=f'Baud rate for Teensy (default: {TOF_BAUD_RATE})',
    )
    parser.add_argument(
        '--no-tof',
        action='store_true',
        help='Disable ToF/IR sensors entirely',
    )
    parser.add_argument(
        '--no-tof-plot',
        action='store_true',
        help='Connect ToF sensors but disable the ToF plotter window',
    )
    args = parser.parse_args()

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
        if args.teensy_port == 'auto':
            teensy_port = TOF_DEFAULT_PORT
        elif args.teensy_port:
            teensy_port = args.teensy_port

    print(f"Connecting to Braccio on {args.port} at {args.baud} baud...")
    if teensy_port:
        print(f"Connecting to ToF/IR Teensy on {teensy_port} at {args.teensy_baud} baud...")
    else:
        print("ToF/IR sensors: disabled (pass --teensy-port to enable)")
    print("Press ESC to quit.\n")

    ctrl = BraccioController(
        args.port, args.baud,
        teensy_port=teensy_port,
        teensy_baud=args.teensy_baud,
    )

    if not args.no_plot:
        from .plotter import ArmPlotter
        ctrl.attach_plotter(ArmPlotter(ctrl._state))

    if teensy_port and not args.no_tof_plot:
        from .tof_plotter import ToFPlotter
        ctrl.attach_tof_plotter(ToFPlotter(ctrl.tof_state))

    try:
        ctrl.run()
    except KeyboardInterrupt:
        pass

    print("\nController stopped.")


if __name__ == '__main__':
    main()
