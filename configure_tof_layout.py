#!/usr/bin/env python3
"""Interactive ToF wrist-layout viewer.

The runtime sensor layout now comes only from the URDF ToF links.
This tool remains useful as a live inspection aid, but saved edits no longer
change controller or plotter geometry.
"""

from __future__ import annotations

import argparse
import curses
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BRACCIO_ROOT = ROOT / "braccio_main_runner"
if str(BRACCIO_ROOT) not in sys.path:
    sys.path.insert(0, str(BRACCIO_ROOT))

from braccio_ctrl.constants import (  # noqa: E402
    BAUD_RATE,
    PROFILE_SWEEP_GRIPPER_DEG,
    PROFILE_SWEEP_R_MM,
    PROFILE_SWEEP_WRIST_OFFSET_DEG,
    PROFILE_SWEEP_WRIST_ROT_DEG,
    PROFILE_SWEEP_Z_MM,
    TOF_BAUD_RATE,
)
from braccio_ctrl.protocol import cmd_get_pos, cmd_set_ik_polar  # noqa: E402
from braccio_ctrl.serial_bridge import SerialBridge  # noqa: E402
from braccio_ctrl.tof_sensor import ToFBridge, ToFState  # noqa: E402
from planning.sensor_config import (  # noqa: E402
    DEFAULT_SENSOR_CONFIG,
    LAYOUT_NOTES_PATH,
    SENSOR_CONFIG,
    export_sensor_layout_notes,
    save_sensor_layout_notes,
    set_channel_layout,
)


POSE_THETA_DEG = 90.0
ANGLE_SNAP = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}
CHANNEL_TITLES = {
    0: "CH0 WEST  boresight -Y",
    1: "CH1 EAST  boresight +Y",
    2: "CH2 NORTH boresight -X",
    3: "CH3 SOUTH boresight +X",
}


def _wrap_deg(value: float) -> float:
    return float(value) % 360.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


class LayoutConfigurator:
    def __init__(self, mega_port: str | None, teensy_port: str, mega_baud: int, teensy_baud: int):
        self._mega_port = mega_port
        self._teensy_port = teensy_port
        self._mega_baud = mega_baud
        self._teensy_baud = teensy_baud

        self._mega = SerialBridge(mega_port, mega_baud) if mega_port else None
        self._tof_state = ToFState(num_channels=4)
        self._tof = ToFBridge(self._tof_state)
        self._selected_ch = 0
        self._status = "Starting..."
        self._running = True
        self._save_path = LAYOUT_NOTES_PATH
        self._channels = {}
        self._load_current_layout()

    def _load_current_layout(self) -> None:
        self._channels = {}
        for ch in range(4):
            cfg = SENSOR_CONFIG["channels"][ch]
            self._channels[ch] = {
                "label": str(cfg.get("label", f"ch{ch}")),
                "mount_angle_deg": float(cfg.get("mount_angle_deg", DEFAULT_SENSOR_CONFIG["channels"][ch].get("mount_angle_deg", 0.0))),
                "roll_deg": float(cfg.get("roll_deg", 0.0)),
                "radial_offset_m": float(cfg.get("radial_offset_m", 0.018)),
                "axial_offset_m": float(cfg.get("axial_offset_m", 0.0)),
            }

    def _connect(self) -> None:
        if self._mega is not None:
            if self._mega.connect():
                self._send_sweep_pose()
            else:
                self._status = f"Mega offline: {self._mega.connect_error}"
        if self._tof.connect(self._teensy_port, self._teensy_baud):
            self._tof.set_active_channels(4)
            self._status = f"ToF live on {self._teensy_port}"
        else:
            self._status = f"Teensy offline on {self._teensy_port}"

    def _send_sweep_pose(self) -> None:
        if self._mega is None or not self._mega.is_open():
            return
        cmd = cmd_set_ik_polar(
            POSE_THETA_DEG,
            PROFILE_SWEEP_R_MM,
            PROFILE_SWEEP_Z_MM,
            PROFILE_SWEEP_WRIST_OFFSET_DEG,
            PROFILE_SWEEP_WRIST_ROT_DEG,
            PROFILE_SWEEP_GRIPPER_DEG,
        )
        self._mega.send_cmd(cmd)
        self._mega.send_cmd(cmd_get_pos())
        self._status = (
            f"Arm moved to sweep pose theta={POSE_THETA_DEG:.0f} "
            f"r={PROFILE_SWEEP_R_MM:.0f} z={PROFILE_SWEEP_Z_MM:.0f}"
        )

    def _apply_channel(self, ch: int) -> None:
        entry = self._channels[ch]
        set_channel_layout(
            ch,
            mount_angle_deg=entry["mount_angle_deg"],
            roll_deg=entry["roll_deg"],
            radial_offset_m=entry["radial_offset_m"],
            axial_offset_m=entry["axial_offset_m"],
        )

    def _apply_all(self) -> None:
        for ch in range(4):
            self._apply_channel(ch)

    def _save(self) -> None:
        self._status = "URDF is now the only sensor-layout source; save is disabled"

    def _reset_selected(self) -> None:
        cfg = DEFAULT_SENSOR_CONFIG["channels"][self._selected_ch]
        self._channels[self._selected_ch] = {
            "label": str(cfg.get("label", f"ch{self._selected_ch}")),
            "mount_angle_deg": float(cfg.get("mount_angle_deg", 0.0)),
            "roll_deg": float(cfg.get("roll_deg", 0.0)),
            "radial_offset_m": float(cfg.get("radial_offset_m", 0.018)),
            "axial_offset_m": float(cfg.get("axial_offset_m", 0.0)),
        }
        self._apply_channel(self._selected_ch)
        self._status = f"Reset CH{self._selected_ch} to default"

    def _update_selected(self, key: int) -> None:
        ch = self._selected_ch
        entry = self._channels[ch]
        changed = False

        if key in (ord("0"), ord("1"), ord("2"), ord("3")):
            self._selected_ch = key - ord("0")
            return
        if key == ord("a"):
            entry["mount_angle_deg"] = _wrap_deg(entry["mount_angle_deg"] + 5.0)
            changed = True
        elif key == ord("d"):
            entry["mount_angle_deg"] = _wrap_deg(entry["mount_angle_deg"] - 5.0)
            changed = True
        elif key == ord("w"):
            entry["roll_deg"] = _wrap_deg(entry["roll_deg"] + 5.0)
            changed = True
        elif key == ord("s"):
            entry["roll_deg"] = _wrap_deg(entry["roll_deg"] - 5.0)
            changed = True
        elif key == ord("i"):
            entry["radial_offset_m"] = _clamp(entry["radial_offset_m"] + 0.001, 0.005, 0.060)
            changed = True
        elif key == ord("k"):
            entry["radial_offset_m"] = _clamp(entry["radial_offset_m"] - 0.001, 0.005, 0.060)
            changed = True
        elif key == ord("j"):
            entry["axial_offset_m"] = _clamp(entry["axial_offset_m"] - 0.001, -0.050, 0.050)
            changed = True
        elif key == ord("l"):
            entry["axial_offset_m"] = _clamp(entry["axial_offset_m"] + 0.001, -0.050, 0.050)
            changed = True
        else:
            try:
                key_name = chr(key).upper()
            except Exception:
                key_name = ""
            if key_name in ANGLE_SNAP:
                entry["mount_angle_deg"] = ANGLE_SNAP[key_name]
                changed = True
        if key == ord("r"):
            self._reset_selected()
            return
        elif key == ord("p"):
            self._save()
            return
        elif key == ord("m"):
            self._send_sweep_pose()
            return
        elif key in (ord("q"), 27):
            self._running = False
            return

        if changed:
            self._apply_channel(ch)
            self._status = (
                f"CH{ch}: angle={entry['mount_angle_deg']:.1f} deg  "
                f"roll={entry['roll_deg']:.1f} deg  "
                f"rad={entry['radial_offset_m'] * 1000.0:.1f} mm  "
                f"axial={entry['axial_offset_m'] * 1000.0:.1f} mm"
            )

    def _matrix_lines(self, grid: np.ndarray, validity: np.ndarray) -> list[str]:
        if grid.ndim != 2 or validity.ndim != 2:
            return ["(no frame)"]
        lines = []
        for i in range(grid.shape[0]):
            vals = []
            for j in range(grid.shape[1]):
                if int(validity[i, j]) == 0 or not np.isfinite(grid[i, j]) or float(grid[i, j]) <= 0.0:
                    vals.append("  --")
                else:
                    vals.append(f"{int(round(float(grid[i, j]))):4d}")
            lines.append(" ".join(vals))
        return lines

    def _cross_section_lines(self) -> list[str]:
        width = 33
        height = 17
        canvas = [[" " for _ in range(width)] for _ in range(height)]
        cx = width // 2
        cy = height // 2
        radius = 6

        for deg in range(0, 360, 10):
            ang = math.radians(float(deg))
            x = int(round(cx + radius * math.cos(ang)))
            y = int(round(cy - radius * math.sin(ang)))
            if 0 <= x < width and 0 <= y < height:
                canvas[y][x] = "."

        canvas[cy][cx] = "O"
        for label, dx, dy in (("+Z", 0, -8), ("-Z", 0, 8), ("+Y", 11, 0), ("-Y", -11, 0)):
            x = cx + dx
            y = cy + dy
            for idx, ch in enumerate(label):
                xx = x + idx - (len(label) // 2)
                if 0 <= xx < width and 0 <= y < height:
                    canvas[y][xx] = ch

        for ch in range(4):
            entry = self._channels[ch]
            ang = math.radians(float(entry["mount_angle_deg"]))
            x = int(round(cx + radius * math.cos(ang)))
            y = int(round(cy - radius * math.sin(ang)))
            ax = int(round(cx + (radius + 2) * math.cos(ang)))
            ay = int(round(cy - (radius + 2) * math.sin(ang)))
            marker = str(ch)
            if ch == self._selected_ch:
                marker = "*"
            if 0 <= x < width and 0 <= y < height:
                canvas[y][x] = marker
            arrow = ">"
            if abs(math.sin(ang)) > abs(math.cos(ang)):
                arrow = "^" if math.sin(ang) > 0.0 else "v"
            else:
                arrow = ">" if math.cos(ang) > 0.0 else "<"
            if 0 <= ax < width and 0 <= ay < height:
                canvas[ay][ax] = arrow

        lines = ["".join(row).rstrip() for row in canvas]
        lines.append("")
        lines.append("Cross section: +Y right/east, -Y left/west, +Z up/north, -Z down/south")
        lines.append("Markers: * = selected channel, arrow = boresight")
        return lines

    def _draw(self, stdscr) -> None:
        stdscr.erase()
        stdscr.addstr(0, 0, "ToF Wrist Layout Viewer (URDF Source of Truth)", curses.A_BOLD)
        stdscr.addstr(1, 0, f"Arm pose: theta=90  r={PROFILE_SWEEP_R_MM:.0f}  z={PROFILE_SWEEP_Z_MM:.0f}")
        stdscr.addstr(2, 0, "Keys: 0/1/2/3 select | a/d angle | w/s roll | i/k radial | j/l axial | W/E/N/S snap | r reset-view | p disabled | m move pose | q quit")
        stdscr.addstr(3, 0, f"Status: {self._status}")

        lines = self._cross_section_lines()
        for idx, line in enumerate(lines):
            stdscr.addstr(5 + idx, 0, line[:70])

        base_y = 5
        base_x = 38
        snap = self._tof_state.snapshot()
        grids = snap.get("grids", [])
        validity = snap.get("validity", [])
        for ch in range(4):
            gx = base_x + (ch % 2) * 34
            gy = base_y + (ch // 2) * 11
            title = CHANNEL_TITLES[ch]
            if ch == self._selected_ch:
                title = f"[{title}]"
            stdscr.addstr(gy, gx, title[:32], curses.A_BOLD if ch == self._selected_ch else 0)
            grid = np.asarray(grids[ch]) if ch < len(grids) else np.zeros((0, 0), dtype=float)
            val = np.asarray(validity[ch]) if ch < len(validity) else np.zeros((0, 0), dtype=int)
            lines = self._matrix_lines(grid, val)
            for idx, line in enumerate(lines[:8]):
                stdscr.addstr(gy + 1 + idx, gx, line[:32])

        info_y = 25
        stdscr.addstr(info_y, 0, "Runtime layout source: URDF ToF links (note-file overrides disabled)")
        stdscr.addstr(info_y + 1, 0, "Selected channel parameters:", curses.A_BOLD)
        entry = self._channels[self._selected_ch]
        stdscr.addstr(
            info_y + 2,
            0,
            (
                f"CH{self._selected_ch} {entry['label']}  "
                f"mount_angle={entry['mount_angle_deg']:.1f} deg  "
                f"roll={entry['roll_deg']:.1f} deg  "
                f"radial={entry['radial_offset_m'] * 1000.0:.1f} mm  "
                f"axial={entry['axial_offset_m'] * 1000.0:.1f} mm"
            )[:110],
        )
        cfg = SENSOR_CONFIG["channels"][self._selected_ch]
        stdscr.addstr(
            info_y + 3,
            0,
            (
                f"Derived origin={cfg.get('origin_m')}  "
                f"axis={cfg.get('axis_dir')}  "
                f"up={cfg.get('up_hint')}"
            )[:110],
        )
        stdscr.refresh()

    def run(self) -> None:
        self._connect()

        def _curses_main(stdscr) -> None:
            curses.curs_set(0)
            stdscr.nodelay(True)
            stdscr.timeout(100)
            while self._running:
                self._draw(stdscr)
                key = stdscr.getch()
                if key != -1:
                    self._update_selected(key)
                time.sleep(0.03)

        try:
            curses.wrapper(_curses_main)
        finally:
            if self._tof.is_open():
                self._tof.close()
            if self._mega is not None and self._mega.is_open():
                self._mega.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ToF wrist layout configurator")
    parser.add_argument("mega_port", nargs="?", default=None, help="Optional Mega COM port used to move the arm to the sweep pose")
    parser.add_argument("--teensy-port", required=True, help="Teensy COM port for live ToF streaming")
    parser.add_argument("--mega-baud", type=int, default=BAUD_RATE, help="Mega baud rate")
    parser.add_argument("--teensy-baud", type=int, default=TOF_BAUD_RATE, help="Teensy baud rate")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = LayoutConfigurator(
        mega_port=args.mega_port,
        teensy_port=args.teensy_port,
        mega_baud=int(args.mega_baud),
        teensy_baud=int(args.teensy_baud),
    )
    app.run()


if __name__ == "__main__":
    main()
