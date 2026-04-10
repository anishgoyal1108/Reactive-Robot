"""
display.py — curses terminal UI.

CursesDisplay.render() redraws the full screen from a state snapshot dict
produced by ArmState.snapshot().  It never holds any lock — it receives
an already-copied snapshot.

Layout
------
  ┌─────────────────────────────────────────────────────┐
  │              Braccio Arm Controller                  │
  │                                                      │
  │  JOINT STATUS                                        │
  │    Base  :  90° [##########----------]  (0-180°)     │
  │    ...                                               │
  │                                                      │
  │  IK STATE                                            │
  │    theta:  90.0°   r:  152.0 mm   z:  -50.0 mm      │
  │    wrist offset:  +0°   wrist rot:  90°              │
  │    equilibrium:  theta=90.0  r=152.0  z=-50.0       │
  │                                                      │
  │  CONTROLS                                            │
  │    A/D: theta ±5°    W/S: reach ±10mm   Q/E: z ±10mm│
  │    I/K: wristV ±5°   J/L: wristR ±5°   O/P: gripper │
  │    +/-: slew rate    H: go to equil    Shift+H: set  │
  │    ESC: quit                                         │
  │                                                      │
  │  Slew: 1 deg/tick (100 deg/s)   Serial: CONNECTED   │
  │  Sent: 'SET ALL ...'   Recv: 'OK ALL=...'            │
  │  ERROR: ...                                          │
  └─────────────────────────────────────────────────────┘
"""

import curses
import numpy as np
from .constants import (
    JOINT_LIMITS,
    JOINT_NAMES,
    SENSOR_ADVISORY_CHANNELS,
    SENSOR_IGNORE_CHANNELS,
)

# Color pair indices
_C_TITLE = 1
_C_LABEL = 2
_C_OK = 3
_C_ERR = 4
_C_DIM = 5
_C_WARN = 6

_BAR_WIDTH = 20  # characters for the progress bar


class CursesDisplay:
    def __init__(self, stdscr):
        self._scr = stdscr
        curses.curs_set(0)
        self._colors_ready = False
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(_C_TITLE, curses.COLOR_CYAN, -1)
            curses.init_pair(_C_LABEL, curses.COLOR_YELLOW, -1)
            curses.init_pair(_C_OK, curses.COLOR_GREEN, -1)
            curses.init_pair(_C_ERR, curses.COLOR_RED, -1)
            curses.init_pair(_C_DIM, curses.COLOR_WHITE, -1)
            curses.init_pair(_C_WARN, curses.COLOR_MAGENTA, -1)
            self._colors_ready = True

    # ── Public API ────────────────────────────────────────────────────────

    def render(
        self,
        state: dict,
        *,
        imu_snap: dict | None = None,
        sweep_snap: dict | None = None,
        obs_snap: dict | None = None,
    ) -> None:
        """Redraw the full UI from a state snapshot plus optional sub-snapshots."""
        self._scr.erase()
        h, w = self._scr.getmaxyx()
        row = 0

        row = self._draw_title(row, w)
        row = self._draw_joints(row, w, h, state["joints"])
        row = self._draw_ik_state(row, w, h, state)
        row = self._draw_tof_status(row, w, h, state)
        row = self._draw_sweep_status(row, w, h, sweep_snap, obs_snap)
        row = self._draw_imu_status(row, w, h, imu_snap)
        row = self._draw_controls(row, w, h)
        row = self._draw_status(row, w, h, state)

        try:
            self._scr.refresh()
        except curses.error:
            pass

    # ── Section renderers ─────────────────────────────────────────────────

    def _draw_title(self, row: int, w: int) -> int:
        title = " Braccio Arm Controller "
        col = max(0, (w - len(title)) // 2)
        self._safe_addstr(row, col, title, curses.color_pair(_C_TITLE) | curses.A_BOLD)
        return row + 2

    def _draw_joints(self, row: int, w: int, h: int, joints: list) -> int:
        self._safe_addstr(
            row, 0, "JOINT STATUS", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE
        )
        row += 1
        for i, (name, angle) in enumerate(zip(JOINT_NAMES, joints)):
            if row >= h - 1:
                break
            lo, hi = JOINT_LIMITS[i]
            angle_f = float(angle)
            if np.isfinite(angle_f):
                pct = (angle_f - lo) / max(hi - lo, 1)
                filled = max(0, min(_BAR_WIDTH, int(pct * _BAR_WIDTH)))
                angle_txt = f"{angle_f:5.1f}"
            else:
                filled = 0
                angle_txt = "  n/a"
            bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
            line = f"  {name}: {angle_txt}°  [{bar}]  ({lo}–{hi}°)"
            self._safe_addstr(row, 0, line[: w - 1])
            row += 1
        return row + 1

    def _draw_ik_state(self, row: int, w: int, h: int, state: dict) -> int:
        if row >= h - 1:
            return row
        self._safe_addstr(
            row, 0, "IK STATE", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE
        )
        row += 1

        if row < h - 1:
            line = (
                f"  theta: {state['theta']:6.1f}°   "
                f"r: {state['r']:7.1f} mm   "
                f"z: {state['z']:7.1f} mm"
            )
            self._safe_addstr(row, 0, line[: w - 1])
            row += 1

        if row < h - 1:
            offset_str = f"{state['wrist_offset']:+.0f}"
            line = (
                f"  wrist offset: {offset_str}°   "
                f"wrist rot: {state['wrist_rot']}°   "
                f"gripper: {state['joints'][5]}°"
            )
            self._safe_addstr(row, 0, line[: w - 1])
            row += 1

        if row < h - 1:
            line = (
                f"  equilibrium →  "
                f"theta={state['equil_theta']:.1f}°  "
                f"r={state['equil_r']:.1f} mm  "
                f"z={state['equil_z']:.1f} mm"
            )
            self._safe_addstr(row, 0, line[: w - 1], curses.color_pair(_C_DIM))
            row += 1

        return row + 1

    def _draw_tof_status(self, row: int, w: int, h: int, state: dict) -> int:
        """Draw ToF sensor / IR proximity / obstacle status section."""
        if row >= h - 1:
            return row

        tof = state.get("tof_snapshot", {})
        if not tof:
            # No ToF connected — show a one-liner
            self._safe_addstr(
                row,
                0,
                "TOF/IR SENSORS",
                curses.color_pair(_C_LABEL) | curses.A_UNDERLINE,
            )
            row += 1
            if row < h - 1:
                self._safe_addstr(
                    row,
                    0,
                    "  (no Teensy connected — use --teensy-port)",
                    curses.color_pair(_C_DIM),
                )
                row += 1
            return row + 1

        self._safe_addstr(
            row, 0, "TOF/IR SENSORS", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE
        )
        row += 1

        # Serial link (distinct from “no valid ToF cells”)
        if row < h - 1:
            if not tof.get("connected"):
                self._safe_addstr(
                    row,
                    0,
                    f"  SERIAL: not open  (expected port {tof.get('port', '?')!r})",
                    curses.color_pair(_C_ERR) | curses.A_BOLD,
                )
                row += 1
            else:
                self._safe_addstr(
                    row,
                    0,
                    f"  SERIAL: OK  {tof.get('port', '')}",
                    curses.color_pair(_C_OK),
                )
                row += 1

        # Per-channel summary: CH0: 450mm  CH1: 320mm  CH2: --  CH3: 1200mm
        if row < h - 1:
            parts = []
            num_ch = tof.get("num_channels", 4)
            fcnt = tof.get("frame_cnt", [0] * num_ch)
            thrv = tof.get("tof_thresholds_mm", [300.0] * num_ch)
            vmin = tof.get("diag_raw_min_mm", [float("nan")] * num_ch)
            vmax = tof.get("diag_raw_max_mm", [float("nan")] * num_ch)
            vcel = tof.get("diag_valid_cells", [0] * num_ch)
            zcnt = tof.get("diag_zone_count", [0] * num_ch)
            for ch in range(num_ch):
                if ch in SENSOR_IGNORE_CHANNELS:
                    parts.append(f"CH{ch}:[ign]")
                    continue
                grid = tof["grids"][ch]
                thr = thrv[ch] if ch < len(thrv) else 300.0
                suffix = "(adv)" if ch in SENSOR_ADVISORY_CHANNELS else ""
                if np.isnan(grid).all():
                    if fcnt[ch] > 0 and zcnt[ch] > 0 and vcel[ch] == 0:
                        # Frames parse but firmware marked every cell invalid
                        r0 = vmin[ch] if ch < len(vmin) else float("nan")
                        r1 = vmax[ch] if ch < len(vmax) else float("nan")
                        if np.isfinite(r0) and np.isfinite(r1):
                            rp = f"{r0:.0f}-{r1:.0f}"
                        else:
                            rp = "?"
                        parts.append(f"CH{ch}:masked({rp})/{thr:.0f}{suffix}")
                    elif fcnt[ch] > 0:
                        parts.append(f"CH{ch}:no-cells/{thr:.0f}{suffix}")
                    else:
                        parts.append(f"CH{ch}:--/{thr:.0f}{suffix}")
                else:
                    mn = float(np.nanmin(grid))
                    flag = "!" if mn < thr else ""
                    parts.append(f"CH{ch}:{mn:.0f}/{thr:.0f}{flag}{suffix}")
            line = "  " + "   ".join(parts)
            self._safe_addstr(row, 0, line[: w - 1], curses.color_pair(_C_DIM))
            row += 1

        # IR status
        if row < h - 1:
            ir_label = tof.get("ir_label", "?")
            ir_bits = tof.get("ir_bits", 0)
            line = f"  IR (OUT1D): {ir_bits:02b} = {ir_label}"
            attr = curses.color_pair(_C_DIM)
            if ir_bits >= 2:
                attr = curses.color_pair(_C_ERR) | curses.A_BOLD
            elif ir_bits == 1:
                attr = curses.color_pair(_C_WARN)
            self._safe_addstr(row, 0, line[: w - 1], attr)
            row += 1

        # Obstacle response
        if row < h - 1:
            obs = state.get("obstacle_response", "clear")
            src = state.get("obstacle_source", "")
            dist = state.get("obstacle_dist_mm", -1)
            thresholds = tof.get("tof_thresholds_mm", [300.0, 300.0, 50.0, 50.0])
            thresh = min(
                thresholds[ch]
                for ch in range(len(thresholds))
                if ch not in SENSOR_IGNORE_CHANNELS
            )

            if obs == "back_away":
                line = f"  *** OBSTACLE: BACK AWAY (IR, ToF missed!) ***"
                attr = curses.color_pair(_C_ERR) | curses.A_BOLD | curses.A_BLINK
            elif obs == "replan":
                line = (
                    f"  OBSTACLE: REPLAN ({src}, {dist:.0f}mm "
                    f"< {thresh:.0f}mm ch-threshold)"
                )
                attr = curses.color_pair(_C_WARN) | curses.A_BOLD
            else:
                line = f"  Obstacle: CLEAR  (threshold: {thresh:.0f}mm)"
                attr = curses.color_pair(_C_OK)
            self._safe_addstr(row, 0, line[: w - 1], attr)
            row += 1

        return row + 1

    def _draw_sweep_status(
        self, row: int, w: int, h: int, sweep_snap: dict | None, obs_snap: dict | None
    ) -> int:
        """Draw autonomous sweep state and obstacle map summary."""
        if row >= h - 1:
            return row

        self._safe_addstr(
            row,
            0,
            "SWEEP / OBSTACLE MAP",
            curses.color_pair(_C_LABEL) | curses.A_UNDERLINE,
        )
        row += 1

        if sweep_snap is None:
            if row < h - 1:
                self._safe_addstr(
                    row, 0, "  (sweep not initialised)", curses.color_pair(_C_DIM)
                )
                row += 1
            return row + 1

        # Sweep state line
        if row < h - 1:
            state_str = sweep_snap.get("sweep_state", "idle").upper()
            theta = sweep_snap.get("current_theta", 0.0)
            direction = sweep_snap.get("direction", 1)
            dir_sym = "→" if direction > 0 else "←"
            running = sweep_snap.get("running", False)
            bypass = sweep_snap.get("bypass_theta")

            attr = curses.color_pair(_C_DIM)
            if state_str == "RUNNING":
                attr = curses.color_pair(_C_OK)
            elif state_str in ("PAUSED_OBSTACLE", "BACK_AWAY"):
                attr = curses.color_pair(_C_WARN) | curses.A_BOLD

            line = (
                f"  SWEEP: {'[ON]' if running else '[OFF]'} "
                f"{state_str}  theta={theta:.1f}° {dir_sym}"
            )
            if bypass is not None:
                line += f"  bypass→{bypass:.1f}°"
            self._safe_addstr(row, 0, line[: w - 1], attr)
            row += 1

        # Obstacle map summary
        if row < h - 1 and obs_snap is not None:
            num_pts = obs_snap.get("num_points", 0)
            est = obs_snap.get("kalman_estimate")
            unc = obs_snap.get("kalman_uncertainty_mm", -1.0)
            age = obs_snap.get("last_obs_age_s", 99.0)

            if num_pts > 0:
                cen = obs_snap.get("centroid")
                cen_str = f"[{cen[0]:.0f},{cen[1]:.0f},{cen[2]:.0f}]mm" if cen else "?"
                line = f"  OBS MAP: {num_pts} pts  centroid={cen_str}"
                attr = curses.color_pair(_C_WARN)
            elif est is not None and age < 2.0:
                line = (
                    f"  OBS MAP: est=({est[0]:.0f},{est[1]:.0f},"
                    f"{est[2]:.0f})mm  unc={unc:.0f}mm  age={age:.1f}s"
                )
                attr = curses.color_pair(_C_DIM)
            else:
                line = "  OBS MAP: clear"
                attr = curses.color_pair(_C_OK)
            self._safe_addstr(row, 0, line[: w - 1], attr)
            row += 1

        # Persistent memory summary
        if row < h - 1 and obs_snap is not None:
            mem = obs_snap.get("memory") or {}
            if mem:
                line = (
                    f"  OBS MEM: cells={mem.get('num_cells', 0)} "
                    f"occ={mem.get('num_occupied_cells', 0)} "
                    f"max_conf={mem.get('max_confidence', 0.0):.2f}"
                )
                self._safe_addstr(row, 0, line[: w - 1], curses.color_pair(_C_DIM))
                row += 1


        return row + 1

    def _draw_imu_status(self, row: int, w: int, h: int, imu_snap: dict | None) -> int:
        """Draw IMU orientation and calibration status."""
        if row >= h - 1:
            return row

        self._safe_addstr(
            row, 0, "IMU", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE
        )
        row += 1

        if imu_snap is None:
            if row < h - 1:
                self._safe_addstr(
                    row,
                    0,
                    "  (no IMU data — check Teensy firmware)",
                    curses.color_pair(_C_DIM),
                )
                row += 1
            return row + 1

        if row < h - 1:
            cal = imu_snap.get("calibrated", False)
            yaw_rel = imu_snap.get("yaw_relative", 0.0)
            yaw_raw = imu_snap.get("yaw_deg", 0.0)
            roll = imu_snap.get("roll_deg", 0.0)
            pitch = imu_snap.get("pitch_deg", 0.0)
            cal_str = (
                f"[calibrated, ref={imu_snap.get('yaw_calibration_offset',0):.1f}°]"
            )
            if not cal:
                cal_str = "[uncalibrated — press C to calibrate]"

            line = (
                f"  roll={roll:+.1f}°  pitch={pitch:+.1f}°  "
                f"yaw={yaw_raw:.1f}° (rel={yaw_rel:+.1f}°)  {cal_str}"
            )
            attr = curses.color_pair(_C_OK) if cal else curses.color_pair(_C_WARN)
            self._safe_addstr(row, 0, line[: w - 1], attr)
            row += 1

        return row + 1

    def _draw_controls(self, row: int, w: int, h: int) -> int:
        if row >= h - 1:
            return row
        self._safe_addstr(
            row, 0, "CONTROLS", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE
        )
        row += 1
        lines = [
            "  A/D: theta ±5°     W/S: reach ±10mm    Q/E: height ±10mm",
            "  I/K: wristV ±5°    J/L: wristR ±5°     O/[ : grip open",
            "  P/] : grip close   +/-: slew rate",
            "  H: go to equil     Shift+H: set equil   ESC: quit",
            "  M: states menu     X: sequence editor",
            "  Z: start/stop sweep   C: IMU calibrate",
        ]
        for line in lines:
            if row >= h - 1:
                break
            self._safe_addstr(row, 0, line[: w - 1], curses.color_pair(_C_DIM))
            row += 1
        return row + 1

    def _draw_status(self, row: int, w: int, h: int, state: dict) -> int:
        if row >= h - 1:
            return row

        delta = state["delta"]
        slew_dps = delta * 100
        connected = state["connected"]
        conn_str = "CONNECTED" if connected else "DISCONNECTED"
        conn_attr = (
            curses.color_pair(_C_OK) | curses.A_BOLD
            if connected
            else curses.color_pair(_C_ERR) | curses.A_BOLD
        )

        prefix = f"  Slew: {delta} deg/tick ({slew_dps} deg/s)   Serial: "
        self._safe_addstr(row, 0, prefix[: w - 1])
        col = len(prefix)
        if col < w - 1:
            self._safe_addstr(row, col, conn_str[: w - col - 1], conn_attr)
        row += 1

        if row < h - 1:
            cmd_disp = state["last_cmd"][:30]
            resp_disp = state["last_resp"][:30]
            line = f"  Sent: {cmd_disp!r}   Recv: {resp_disp!r}"
            self._safe_addstr(row, 0, line[: w - 1], curses.color_pair(_C_DIM))
            row += 1

        if row < h - 1 and state["last_error"]:
            err_attr = curses.color_pair(_C_ERR) | curses.A_BOLD
            prefix = "  ERROR: "
            full = prefix + state["last_error"]
            if len(full) < w:
                self._safe_addstr(row, 0, full, err_attr)
                row += 1
            else:
                # First line: as much as fits
                self._safe_addstr(row, 0, full[: w - 1], err_attr)
                row += 1
                # Second line: remainder, indented to align after prefix
                if row < h - 1:
                    remainder = "    " + state["last_error"][w - 1 - len(prefix) :]
                    self._safe_addstr(row, 0, remainder[: w - 1], err_attr)
                    row += 1

        return row

    # ── Safe write helper ─────────────────────────────────────────────────

    def _safe_addstr(self, row: int, col: int, text: str, attr: int = 0) -> None:
        """Write text without raising on boundary overruns."""
        h, w = self._scr.getmaxyx()
        if row >= h or col >= w or not text:
            return
        text = text[: w - col - 1]  # clamp to available width
        try:
            self._scr.addstr(row, col, text, attr)
        except curses.error:
            pass
