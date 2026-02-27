"""
__main__.py — CLI entry point.

  python -m braccio_ctrl [port] [--baud N] [--list-ports] [--no-plot]
"""

import argparse
import sys

from .serial_bridge import SerialBridge
from .controller    import BraccioController
from .constants     import DEFAULT_PORT, BAUD_RATE


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m braccio_ctrl',
        description='Braccio Arm IK Keyboard Controller',
    )
    parser.add_argument(
        'port',
        nargs='?',
        default=DEFAULT_PORT,
        help=f'Serial port device (default: {DEFAULT_PORT})',
    )
    parser.add_argument(
        '--baud',
        type=int,
        default=BAUD_RATE,
        help=f'Baud rate (default: {BAUD_RATE})',
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

    print(f"Connecting to Braccio on {args.port} at {args.baud} baud...")
    print("Press ESC to quit.\n")

    ctrl = BraccioController(args.port, args.baud)

    if not args.no_plot:
        from .plotter import ArmPlotter
        ctrl.attach_plotter(ArmPlotter(ctrl._state))

    try:
        ctrl.run()
    except KeyboardInterrupt:
        pass

    print("\nController stopped.")


if __name__ == '__main__':
    main()
