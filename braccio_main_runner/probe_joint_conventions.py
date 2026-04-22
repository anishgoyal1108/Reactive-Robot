#!/usr/bin/env python3
"""
probe_joint_conventions.py — Verify Braccio servo sign conventions.

Moves one joint at a time from HOME to a "forward" extreme and a
"backward" extreme, at slow slew. At each pose the script prints what
the IK code expects to happen physically. You compare to what the arm
actually does.

After the run, tell me for each test what you observed (forward/back,
left/right, up/down) and I'll encode any detected sign flips.

Usage:
    python probe_joint_conventions.py [/dev/ttyACM0]
"""
from __future__ import annotations
import sys, time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from braccio_ctrl.constants import DEFAULT_PORT, BAUD_RATE, HOME_POS
from braccio_ctrl.protocol import cmd_set_all, cmd_set_delta, cmd_home
from braccio_ctrl.serial_bridge import SerialBridge

PORT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
SETTLE = 4.5        # seconds to hold each test pose so you can observe
HOME_SETTLE = 3.0   # return-to-HOME dwell


def go_home(bridge):
    bridge.send_cmd(cmd_home())
    time.sleep(HOME_SETTLE)


def send_and_hold(bridge, joints, label, expectation):
    print(f"\n  → {label}")
    print(f"    expected behavior: {expectation}")
    print(f"    commanding joints {joints}...", flush=True)
    bridge.send_cmd(cmd_set_all(joints))
    time.sleep(SETTLE)


def main():
    print(f"Connecting to {PORT}...")
    bridge = SerialBridge(PORT, BAUD_RATE)
    if not bridge.connect():
        print(f"FAILED: {bridge.connect_error}", file=sys.stderr)
        sys.exit(1)

    # Wait through Arduino's long setup() boot, same pattern as the
    # run_scenario_2 script uses — PING with retries.
    print("Waiting for Arduino PONG (up to 8 s)...", flush=True)
    deadline = time.monotonic() + 8.0
    pong = False
    while time.monotonic() < deadline and not pong:
        pong = bridge.ping(timeout=1.0)
    print("PONG OK." if pong else "No PONG — proceeding anyway.")

    bridge.send_cmd(cmd_set_delta(1))
    go_home(bridge)

    try:
        print("\n=== PROBE 1: SHOULDER ===")
        print("    Expected: S=45 tilts the upper arm FORWARD (toward +X, i.e.,")
        print("    the direction the arm is pointing at HOME's base rotation).")
        print("    S=135 tilts the upper arm BACKWARD (toward -X).")

        send_and_hold(bridge, [90, 45, 90, 90, 90, 73],
                      "S=45", "upper arm tilted FORWARD from vertical")
        go_home(bridge)
        send_and_hold(bridge, [90, 135, 90, 90, 90, 73],
                      "S=135", "upper arm tilted BACKWARD from vertical")
        go_home(bridge)

        print("\n=== PROBE 2: ELBOW ===")
        print("    Expected: E=180 → forearm fully extended, aligned with upper arm.")
        print("    E=45   → forearm folded ~135° from upper arm (toward shoulder).")

        send_and_hold(bridge, [90, 90, 180, 90, 90, 73],
                      "E=180", "forearm aligned with upper arm (both pointing up)")
        go_home(bridge)
        send_and_hold(bridge, [90, 90, 45, 90, 90, 73],
                      "E=45", "forearm folded toward the shoulder (~135° from upper arm)")
        go_home(bridge)

        print("\n=== PROBE 3: WRIST_V ===")
        print("    Expected: WV=180 tilts gripper one way; WV=0 the other way.")
        print("    At HOME (S=E=90, forearm horizontal forward), expect:")
        print("      WV=180 → gripper points UP (roughly +Z)")
        print("      WV=0   → gripper points DOWN (roughly -Z)")

        send_and_hold(bridge, [90, 90, 90, 180, 90, 73],
                      "WV=180", "gripper pointing UP from forearm")
        go_home(bridge)
        send_and_hold(bridge, [90, 90, 90, 0, 90, 73],
                      "WV=0", "gripper pointing DOWN from forearm")
        go_home(bridge)

        print("\n=== PROBE 4: BASE ===")
        print("    Expected: B=45  → rotated RIGHT (from operator facing arm)")
        print("              B=135 → rotated LEFT")

        send_and_hold(bridge, [45, 90, 90, 90, 90, 73],
                      "B=45", "base rotated to the RIGHT")
        go_home(bridge)
        send_and_hold(bridge, [135, 90, 90, 90, 90, 73],
                      "B=135", "base rotated to the LEFT")
        go_home(bridge)

        print("\nDone. Parking at HOME.")

    finally:
        bridge.send_cmd(cmd_set_delta(1))
        bridge.send_cmd(cmd_home())
        time.sleep(3.0)
        bridge.close()


if __name__ == "__main__":
    main()
