"""CLI entry point for Braccio controller."""

from __future__ import annotations

import argparse
import sys

from .constants import (
    BAUD_RATE,
    DEFAULT_PORT,
    TOF_BAUD_RATE,
    TOF_CHANNELS_DEFAULT,
    TOF_CHANNELS_MAX,
    TOF_DEFAULT_PORT,
    IMU_LPF_ALPHA_DEFAULT,
)
from .controller import BraccioController
from .serial_bridge import SerialBridge


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m braccio_ctrl",
        description="Braccio Arm Controller + ToF obstacle monitoring",
    )
    parser.add_argument(
        "port",
        nargs="?",
        default=DEFAULT_PORT,
        help=f"Serial port for Braccio arm (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help=f"Baud rate for Braccio (default: {BAUD_RATE})",
    )
    parser.add_argument("--list-ports", action="store_true", help="List serial ports and exit")

    parser.add_argument(
        "--teensy-port",
        default=None,
        help=f"Serial port for ToF Teensy (default: {TOF_DEFAULT_PORT}). Use 'auto' for default.",
    )
    parser.add_argument(
        "--teensy-baud",
        type=int,
        default=TOF_BAUD_RATE,
        help=f"Baud rate for Teensy (default: {TOF_BAUD_RATE})",
    )
    parser.add_argument("--tof-channels", type=int, default=TOF_CHANNELS_DEFAULT,
                        help=f"Active ToF channels (1-{TOF_CHANNELS_MAX}, default: {TOF_CHANNELS_DEFAULT})")
    parser.add_argument("--no-tof", action="store_true", help="Disable ToF/IR sensors")
    parser.add_argument("--imu-lpf-alpha", type=float, default=IMU_LPF_ALPHA_DEFAULT,
                        help=f"IMU LPF alpha in [0.02..1.0] (default: {IMU_LPF_ALPHA_DEFAULT})")

    parser.add_argument("--enable-plot", action="store_true", help="Enable the joint-angle plotter")
    parser.add_argument("--enable-tof-plot", action="store_true", help="Enable the ToF plotter")

    parser.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-tof-plot", action="store_true", help=argparse.SUPPRESS)

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

    teensy_port = None
    if not args.no_tof:
        if args.teensy_port == "auto":
            teensy_port = TOF_DEFAULT_PORT
        elif args.teensy_port:
            teensy_port = args.teensy_port

    tof_channels = max(1, min(int(args.tof_channels), TOF_CHANNELS_MAX))

    print(f"Connecting to Braccio on {args.port} at {args.baud} baud...")
    if teensy_port:
        print(f"Connecting to ToF Teensy on {teensy_port} at {args.teensy_baud} baud...")
        print(f"ToF channels enabled: {tof_channels}")
    else:
        print("ToF/IR sensors: disabled (pass --teensy-port to enable)")

    ctrl = BraccioController(
        args.port,
        args.baud,
        teensy_port=teensy_port,
        teensy_baud=args.teensy_baud,
        tof_channels_enabled=tof_channels,
        imu_lpf_alpha=float(args.imu_lpf_alpha),
    )

    plot_enabled = args.enable_plot and not args.no_plot
    tof_plot_enabled = args.enable_tof_plot and not args.no_tof_plot

    if plot_enabled:
        from .plotter import ArmPlotter

        ctrl.attach_plotter(ArmPlotter(ctrl._state))
    else:
        print("Joint plotter: disabled (use --enable-plot to enable)")

    if teensy_port and tof_plot_enabled:
        from .tof_plotter import ToFPlotter

        ctrl.attach_tof_plotter(ToFPlotter(ctrl.tof_state))
    elif teensy_port:
        print("ToF plotter: disabled (use --enable-tof-plot to enable)")

    print("Press ESC to quit. R=side profile, B=record, T=tuning, C=IMU calibrate.\n")

    try:
        ctrl.run()
    except KeyboardInterrupt:
        pass

    print("\nController stopped.")


if __name__ == "__main__":
    main()


