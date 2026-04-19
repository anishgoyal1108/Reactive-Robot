"""
smoke_test.py — Quick end-to-end sanity check for the digital twin.

Run with:

    python -m braccio_twin.smoke_test

It boots the sim headless, spawns one box, issues a few SET ALL /
HOME / GET POS commands, lets the tick loop run for a second, and
prints the resulting joint positions + ToF / IR / IMU snapshots.

No curses, no display — purely a "does the plumbing work?" test you can
run in CI or after pulling changes.
"""

from __future__ import annotations

import sys
import time

from braccio_ctrl.constants import TOF_NUM_CHANNELS, TOF_THRESHOLDS_MM
from braccio_ctrl.imu_state import IMUState
from braccio_ctrl.tof_sensor import ToFState

from .obstacle_world import ObstacleSpec
from .sim_backend import SimBackend, SimSensorBridge


def main() -> int:
    tof_state = ToFState(num_channels=TOF_NUM_CHANNELS)
    tof_state.tof_threshold_mm = TOF_THRESHOLDS_MM[0]
    tof_state.tof_thresholds_mm = list(TOF_THRESHOLDS_MM)
    imu_state = IMUState()

    sim = SimBackend(gui=False)
    sim.install_sensor_state(tof_state, imu_state)

    if not sim.connect():
        print(f"connect failed: {sim.connect_error}", file=sys.stderr)
        return 1

    world = sim.get_obstacle_world()
    assert world is not None
    world.add(
        ObstacleSpec(
            name="test_box",
            shape="box",
            position_mm=(200.0, 0.0, 80.0),
            size_mm=(60.0, 60.0, 80.0),
        )
    )

    # Exercise the command parser
    sim.send_cmd("PING\n")
    sim.send_cmd("SET ALL 90 90 90 90 90 73\n")
    sim.send_cmd("SET DELTA 2\n")
    sim.send_cmd("GET POS\n")

    # Let the tick loop run and sensors stabilise
    time.sleep(1.0)

    # Drain responses
    print("─" * 60)
    print("Backend responses:")
    while not sim.response_queue.empty():
        print(" ", sim.response_queue.get_nowait())

    # Read sensor state
    tof_snap = tof_state.snapshot()
    imu_snap = imu_state.snapshot()
    print("─" * 60)
    print(f"ToF obstacle_response : {tof_snap['obstacle_response']}")
    print(f"ToF obstacle_source   : {tof_snap['obstacle_source']}")
    print(f"ToF obstacle_dist_mm  : {tof_snap['obstacle_dist_mm']:.1f}")
    print(f"IR bits (debounced)   : {tof_snap['ir_bits']} ({tof_snap['ir_label']})")
    for ch in range(tof_snap["num_channels"]):
        if tof_snap["active"][ch]:
            print(
                f"  ch{ch}: min={tof_snap['diag_raw_min_mm'][ch]:.1f} mm "
                f"max={tof_snap['diag_raw_max_mm'][ch]:.1f} mm "
                f"cells={tof_snap['diag_valid_cells'][ch]}"
            )
    print(f"IMU roll/pitch        : {imu_snap['roll_deg']:.1f}, {imu_snap['pitch_deg']:.1f}")

    # Command a reach toward +X and watch the front ToF see the box
    sim.send_cmd("SET ALL 90 60 120 90 90 73\n")
    time.sleep(1.0)
    tof_snap = tof_state.snapshot()
    print("─" * 60)
    print("After reach toward +X (arm extended):")
    print(f"ToF obstacle_response : {tof_snap['obstacle_response']}")
    print(f"ToF obstacle_source   : {tof_snap['obstacle_source']}")
    print(f"ToF obstacle_dist_mm  : {tof_snap['obstacle_dist_mm']:.1f}")
    for ch in range(tof_snap["num_channels"]):
        if tof_snap["active"][ch]:
            print(
                f"  ch{ch}: min={tof_snap['diag_raw_min_mm'][ch]:.1f} mm"
            )

    sim.close()
    print("─" * 60)
    print("smoke_test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
