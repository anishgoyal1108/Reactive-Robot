#!/usr/bin/env python3
"""
validate_fk.py — Cross-validate commanded z against physical z.

Problem it solves
-----------------
``ik_solver.py``'s ``wrist_level_angle`` formula produces a wrist tilt that
doesn't match the physical Braccio's convention, so ``solve_ik(x, y, z)``
computes joint angles whose actual gripper tip lands several cm away from
the commanded z. RL policies trained in the sim read (θ, r, z) from obs
vectors; if those don't correspond to the physical tip position, sim-to-
real transfer fails.

This tool collects ground-truth ``(joints, measured_z)`` pairs so we can
fit a corrected FK model offline (next step). For each saved pose:

  1. Command ``SET ALL`` with the pose's joint angles.
  2. Wait for the arm to settle.
  3. Optionally read CH3 (bottom-facing) ToF for a raw ground distance.
  4. Prompt the user for a ruler-measured gripper-tip height above the
     table. This is the ground truth.
  5. Compute measured z in the shoulder-pivot frame (which is what
     ``ik_solver.py`` uses).
  6. Report the delta to the commanded z.
  7. Append a record to ``logs/fk_calibration_<timestamp>.json``.

Usage
-----
  python validate_fk.py /dev/ttyACM0 --teensy-port /dev/ttyACM1 \
      --states s2a_pose,s2b_pose,s2c_pose,s2d_start --pivot-mm 150

``--pivot-mm`` is the height of the shoulder-pivot axis above the table,
measured with a ruler once. If you've already done this measurement on
another run, the script can load it from the most-recent calibration
JSON via ``--reuse-pivot``.

The resulting JSON is consumed by ``fit_fk.py`` (planned for the next
iteration) to solve for the correct wrist-angle formula and any link-
length or joint-offset corrections.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import sys
import time
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
os.chdir(_SCRIPT_DIR)

from braccio_ctrl.constants import (  # noqa: E402
    BAUD_RATE, DEFAULT_PORT, LOG_DIR, TOF_BAUD_RATE, TOF_DEFAULT_PORT,
    TOF_NUM_CHANNELS, TOF_SELF_FILTER_MARGIN_PER_CHANNEL_MM, L1, L2, L3,
)
from braccio_ctrl.imu_state import IMUState  # noqa: E402
from braccio_ctrl.protocol import cmd_home, cmd_set_all, cmd_set_delta  # noqa: E402
from braccio_ctrl.serial_bridge import SerialBridge  # noqa: E402
from braccio_ctrl.state_library import StateLibrary  # noqa: E402
from braccio_ctrl.tof_sensor import ToFBridge, ToFState  # noqa: E402

log = logging.getLogger("validate_fk")

_CALIBRATION_FILE_PREFIX = "fk_calibration_"
_CH3_SELF_FILTER_MM = TOF_SELF_FILTER_MARGIN_PER_CHANNEL_MM[3]

# ── Auto-probe pose grid ──────────────────────────────────────────────────
# These poses are deliberately chosen to **isolate each joint's contribution**
# to the tip height. The trio cal_home / cal_wv0 / cal_wv150 changes only WV;
# cal_e45 / cal_e135 changes only E relative to cal_home; cal_s60 / cal_s120
# changes only S. That lets the FK fit uniquely identify each joint's scale +
# offset instead of the underdetermined mess we get when multiple joints
# change simultaneously between calibration points.
#
# All poses stay within the Arduino firmware's joint limits:
#   B  [0, 180], S [15, 165], E [0, 180], WV [0, 180], WR [0, 180], G [10, 73]
# The base (B) is held at 90° throughout because base rotation doesn't
# affect the tip's z in the shoulder-pivot frame.
#
# Move between poses always goes via HOME (90,90,90,90,90,73) at slow slew
# to guarantee safe transits — otherwise going from cal_s60 (arm reaching
# high) straight to cal_s120 (arm low) would whip through awkward
# intermediate angles.
_AUTO_PROBE_POSES = [
    ("cal_home",  [90,  90,  90,  90, 90, 73]),   # reference: all 90
    ("cal_wv0",   [90,  90,  90,   0, 90, 73]),   # WV swing low  (same S,E as home)
    ("cal_wv150", [90,  90,  90, 150, 90, 73]),   # WV swing high (same S,E as home)
    ("cal_e45",   [90,  90,  45,  90, 90, 73]),   # E folded (same S,WV as home)
    ("cal_e135",  [90,  90, 135,  90, 90, 73]),   # E extended
    ("cal_s60",   [90,  60,  90,  90, 90, 73]),   # S toward back (arm higher/behind)
    ("cal_s120",  [90, 120,  90,  90, 90, 73]),   # S toward forward (arm lower/forward)
    ("cal_mix",   [90, 110, 130,  60, 90, 73]),   # all three off-nominal — generalization check
]


def _configure_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "validate_fk.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def _connect_arm(port: str, baud: int) -> SerialBridge:
    """Same PONG-retry contract as run_scenario_2.py — the Arduino takes
    ~4 s to finish setup() before it services Serial."""
    bridge = SerialBridge(port, baud)
    if not bridge.connect():
        raise SystemExit(f"Cannot open {port}: {bridge.connect_error}")
    log.info("Waiting for Arduino PONG (up to 8 s)...")
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if bridge.ping(timeout=1.0):
            log.info("Arduino PONG received.")
            return bridge
    log.warning("No PONG in 8 s — proceeding; commands will queue in the "
                "UART buffer and run once the firmware is ready.")
    return bridge


def _connect_teensy(port: str, baud: int) -> tuple[Optional[ToFBridge], ToFState, IMUState]:
    tof = ToFState(num_channels=TOF_NUM_CHANNELS)
    imu = IMUState()
    bridge = ToFBridge(tof, imu)
    if not bridge.connect(port, baud):
        log.warning("Teensy connect failed — ToF-based z readings disabled.")
        return None, tof, imu
    log.info("Teensy connected — waiting for first ToF frame...")
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline:
        snap = tof.snapshot()
        if any(snap.get("frame_cnt", [])):
            log.info("ToF frames arriving.")
            return bridge, tof, imu
        time.sleep(0.05)
    log.warning("No ToF frames within 2.5 s — using it anyway.")
    return bridge, tof, imu


def _drain_serial(bridge: SerialBridge) -> None:
    """Empty the response queue so the reader thread doesn't block."""
    try:
        while True:
            bridge.response_queue.get_nowait()
    except queue.Empty:
        pass


def _sample_ch3_min(tof: ToFState, duration_s: float = 1.0,
                     self_filter_mm: float = _CH3_SELF_FILTER_MM) -> Optional[float]:
    """Return the minimum CH3 range (mm) observed over ``duration_s`` that
    is outside the self-filter margin. None if no frames or only self-
    returns were seen.

    CH3 faces the ground; readings <= self_filter_mm are the MUX/PCB and
    are discarded. The minimum of what remains is "closest non-self
    return", which in a clear workspace is the table (or whatever object
    is directly beneath the gripper).
    """
    import numpy as np
    deadline = time.monotonic() + duration_s
    best: Optional[float] = None
    while time.monotonic() < deadline:
        snap = tof.snapshot()
        grids = snap.get("grids", [])
        if len(grids) < 4:
            time.sleep(0.1); continue
        g = grids[3]
        if g is None or getattr(g, "size", 0) == 0:
            time.sleep(0.1); continue
        arr = np.asarray(g, dtype=np.float32)
        valid = arr[(arr > self_filter_mm) & np.isfinite(arr)]
        if valid.size:
            m = float(valid.min())
            if best is None or m < best:
                best = m
        time.sleep(0.1)
    return best


def _prompt_float(msg: str, default: Optional[float] = None) -> Optional[float]:
    """Read a float from stdin. Blank input → default (which may be None)."""
    default_str = f" [{default:.1f}]" if default is not None else ""
    while True:
        try:
            raw = input(f"{msg}{default_str}: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Not a number. Try again (or Ctrl+D to accept default).")


def _find_last_pivot_mm() -> Optional[float]:
    """Read --reuse-pivot: scan logs/ for the most recent calibration JSON
    and return its ``pivot_mm`` if present."""
    try:
        files = sorted(
            (f for f in os.listdir(LOG_DIR)
             if f.startswith(_CALIBRATION_FILE_PREFIX) and f.endswith(".json")),
            reverse=True,
        )
        for fname in files:
            with open(os.path.join(LOG_DIR, fname)) as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("pivot_mm") is not None:
                return float(data["pivot_mm"])
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arm_port", nargs="?", default=None,
                        help="Arduino serial port (positional; default: /dev/ttyACM0).")
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Arduino serial port (flag form). Default: {DEFAULT_PORT}")
    parser.add_argument("--baud", type=int, default=BAUD_RATE)
    parser.add_argument("--teensy-port", default=TOF_DEFAULT_PORT,
                        help="Teensy serial port for CH3 ToF (optional).")
    parser.add_argument("--teensy-baud", type=int, default=TOF_BAUD_RATE)
    parser.add_argument("--states", default="",
                        help="Comma-separated list of named states to probe "
                             "(must exist in states.json). Empty + no "
                             "--auto-probe aborts.")
    parser.add_argument("--auto-probe", action="store_true",
                        help="Probe the built-in 8-pose grid designed for FK "
                             "calibration. Can be combined with --states; "
                             "auto-probe poses run first.")
    parser.add_argument("--no-home-hop", action="store_true",
                        help="Don't return to HOME between poses. Saves ~3 s "
                             "per transition but risks awkward intermediate "
                             "trajectories between distant poses.")
    parser.add_argument("--pivot-mm", type=float, default=None,
                        help="Height of the shoulder-pivot axis above the "
                             "table, in mm. Measure once with a ruler. If "
                             "omitted, the script will ask.")
    parser.add_argument("--reuse-pivot", action="store_true",
                        help="Read pivot_mm from the most recent fk_calibration_*.json.")
    parser.add_argument("--settle-s", type=float, default=4.0,
                        help="Seconds to wait after SET ALL before reading sensors.")
    args = parser.parse_args()
    if args.arm_port is not None:
        args.port = args.arm_port

    _configure_logging()

    # Resolve pivot height.
    pivot_mm = args.pivot_mm
    if pivot_mm is None and args.reuse_pivot:
        pivot_mm = _find_last_pivot_mm()
        if pivot_mm is not None:
            print(f"Reusing pivot_mm={pivot_mm:.1f} from most recent calibration file.")
    if pivot_mm is None:
        pivot_mm = _prompt_float(
            "Measure the shoulder-pivot axis height above the table (mm)"
        )
        if pivot_mm is None:
            raise SystemExit("Need a pivot height. Aborting.")
    log.info("Using pivot_mm = %.1f", pivot_mm)

    # Build the pose list. Auto-probe poses have known joints + no polar
    # metadata (IK wasn't involved). State-backed poses come from
    # states.json and carry the main-controller's polar values (which may
    # be wrong — we compare them against the ruler measurement).
    state_lib = StateLibrary()
    poses: list[dict] = []

    if args.auto_probe:
        for name, joints in _AUTO_PROBE_POSES:
            poses.append({
                "name":   name,
                "source": "auto_probe",
                "joints": list(joints),
                "theta":  None,
                "r":      None,
                "z":      None,
            })

    wanted = [s.strip() for s in args.states.split(",") if s.strip()]
    missing = []
    for sn in wanted:
        st = state_lib.get_state(sn)
        if st is None:
            missing.append(sn)
        else:
            poses.append({
                "name":   sn,
                "source": "state_library",
                "joints": [int(j) for j in st["joints"]],
                "theta":  st.get("theta"),
                "r":      st.get("r"),
                "z":      st.get("z"),
            })
    if missing:
        raise SystemExit(
            f"Missing named states in states.json: {missing}. "
            f"Save them via the main controller first."
        )
    if not poses:
        raise SystemExit(
            "No poses to probe. Pass --auto-probe and/or --states.")

    # Connect hardware.
    arm = _connect_arm(args.port, args.baud)
    teensy, tof_state, _imu = _connect_teensy(args.teensy_port, args.teensy_baud)

    # Home at slow slew — known safe pose to start from.
    arm.send_cmd(cmd_set_delta(1))
    arm.send_cmd(cmd_home())
    log.info("Sent HOME; settling %.1fs...", args.settle_s)
    time.sleep(args.settle_s)

    records = []
    HOME_JOINTS = [90, 90, 90, 90, 90, 73]
    try:
        for idx, pose in enumerate(poses, start=1):
            name = pose["name"]
            joints = pose["joints"]
            src = pose["source"]
            print()
            print("=" * 70)
            print(f"[{idx}/{len(poses)}] {name} ({src})")
            print(f"  joints to command: {joints}")
            if pose["z"] is not None:
                print(f"  saved polar (from controller — may be inaccurate): "
                      f"θ={pose['theta']:.1f}°  r={pose['r']:.1f} mm  "
                      f"z={pose['z']:+.1f} mm")
            print("=" * 70)

            # Optional HOME hop between poses for safe trajectory transit.
            # Skip for the very first pose (we already went HOME on startup).
            if idx > 1 and not args.no_home_hop:
                print(f"  (hopping via HOME between poses)...")
                arm.send_cmd(cmd_set_all(HOME_JOINTS))
                _drain_serial(arm)
                time.sleep(args.settle_s)

            # Command the target joints verbatim (byte-for-byte, no IK).
            arm.send_cmd(cmd_set_all(joints))
            _drain_serial(arm)
            print(f"  Waiting {args.settle_s:.1f}s for the arm to settle...")
            time.sleep(args.settle_s)

            # CH3 ground distance (may be None if Teensy absent / only self-returns).
            ch3_mm = None
            if teensy is not None:
                ch3_mm = _sample_ch3_min(tof_state, duration_s=1.0)
                if ch3_mm is None:
                    print("  CH3 ToF: no non-self return (table may be "
                          "outside sensor range or blocked).")
                else:
                    print(f"  CH3 ToF closest non-self return: {ch3_mm:.1f} mm")

            # Derive a CH3-based "ground truth" tip-above-table estimate.
            # CH3 faces down from near the wrist/gripper; its reading is
            # distance from sensor to whatever is below. For a clean table
            # reading, tip_above_table ≈ CH3 distance - (sensor-to-tip
            # offset, tens of mm). We surface the raw number and let the
            # user override with a ruler measurement.
            suggested = ch3_mm if ch3_mm is not None else None
            ruler_tip_mm = _prompt_float(
                "  Ruler-measured tip height ABOVE TABLE (mm)",
                default=suggested,
            )
            if ruler_tip_mm is None:
                print("  (skipped)")
                continue

            measured_z_from_pivot = ruler_tip_mm - pivot_mm
            delta = (measured_z_from_pivot - pose["z"]) if pose["z"] is not None else None
            print(f"  → measured z (from pivot): {measured_z_from_pivot:+.1f} mm")
            if pose["z"] is not None:
                print(f"  → saved polar z (controller):  {pose['z']:+.1f} mm")
                print(f"  → delta (measured - saved):    {delta:+.1f} mm")
            else:
                print(f"  → (auto-probe pose — no saved polar to compare against)")

            records.append({
                "state_name":            name,
                "source":                src,
                "joints":                joints,
                "commanded_theta":       pose["theta"],
                "commanded_r":           pose["r"],
                "commanded_z":           pose["z"],
                "ruler_tip_above_table": ruler_tip_mm,
                "measured_z":            measured_z_from_pivot,
                "ch3_raw_mm":            ch3_mm,
                "delta":                 delta,
            })
    finally:
        # Park at HOME and close.
        try:
            arm.send_cmd(cmd_set_delta(1))
            arm.send_cmd(cmd_home())
        except Exception:
            pass
        try:
            arm.close()
        except Exception:
            pass
        if teensy is not None:
            try:
                teensy.close()
            except Exception:
                pass

    # Write calibration JSON.
    if records:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(LOG_DIR, f"{_CALIBRATION_FILE_PREFIX}{stamp}.json")
        with open(out_path, "w") as f:
            json.dump({
                "pivot_mm":    pivot_mm,
                "L1":          L1,
                "L2":          L2,
                "L3":          L3,
                "records":     records,
            }, f, indent=2)
        print()
        print("=" * 70)
        print(f"Calibration saved: {out_path}")
        print(f"  {len(records)} pose(s) measured.")
        print("  Summary:")
        for rec in records:
            cz = rec["commanded_z"]
            dz = rec["delta"]
            cz_str = f"{cz:+6.1f}" if cz is not None else "   —  "
            dz_str = f"{dz:+6.1f}" if dz is not None else "   —  "
            print(f"    {rec['state_name']:12s}  "
                  f"[{rec['source']:13s}]  "
                  f"saved z={cz_str}  measured={rec['measured_z']:+6.1f}  "
                  f"Δ={dz_str} mm")
        print()
        print("Next step: once you've collected 4+ poses spanning the "
              "workspace, we fit the FK model so commanded z matches "
              "measured z for any pose. That fix goes into ik_solver.py.")


if __name__ == "__main__":
    main()
