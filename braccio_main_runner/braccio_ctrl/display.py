"""curses terminal UI for Braccio controller."""

from __future__ import annotations

import curses

import numpy as np

from .constants import JOINT_LIMITS, JOINT_NAMES

_C_TITLE = 1
_C_LABEL = 2
_C_OK = 3
_C_ERR = 4
_C_DIM = 5
_C_WARN = 6

_BAR_WIDTH = 20


class CursesDisplay:
    def __init__(self, stdscr):
        self._scr = stdscr
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(_C_TITLE, curses.COLOR_CYAN, -1)
            curses.init_pair(_C_LABEL, curses.COLOR_YELLOW, -1)
            curses.init_pair(_C_OK, curses.COLOR_GREEN, -1)
            curses.init_pair(_C_ERR, curses.COLOR_RED, -1)
            curses.init_pair(_C_DIM, curses.COLOR_WHITE, -1)
            curses.init_pair(_C_WARN, curses.COLOR_MAGENTA, -1)

    def render(self, state: dict) -> None:
        self._scr.erase()
        h, w = self._scr.getmaxyx()
        row = 0

        row = self._draw_title(row, w)
        row = self._draw_joints(row, w, h, state["joints"])
        row = self._draw_ik_state(row, w, h, state)
        row = self._draw_tof_status(row, w, h, state)
        row = self._draw_controls(row, w, h)
        self._draw_status(row, w, h, state)

        try:
            self._scr.refresh()
        except curses.error:
            pass

    def _draw_title(self, row: int, w: int) -> int:
        title = " Braccio Arm Controller "
        col = max(0, (w - len(title)) // 2)
        self._safe_addstr(row, col, title, curses.color_pair(_C_TITLE) | curses.A_BOLD)
        return row + 2

    def _draw_joints(self, row: int, w: int, h: int, joints: list[int]) -> int:
        self._safe_addstr(row, 0, "JOINT STATUS", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
        row += 1
        for i, (name, angle) in enumerate(zip(JOINT_NAMES, joints)):
            if row >= h - 1:
                break
            lo, hi = JOINT_LIMITS[i]
            pct = (angle - lo) / max(hi - lo, 1)
            filled = int(pct * _BAR_WIDTH)
            bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
            line = f"  {name}: {angle:3d} deg  [{bar}]  ({lo}-{hi} deg)"
            self._safe_addstr(row, 0, line[: w - 1])
            row += 1
        return row + 1

    def _draw_ik_state(self, row: int, w: int, h: int, state: dict) -> int:
        if row >= h - 1:
            return row
        self._safe_addstr(row, 0, "IK STATE", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
        row += 1

        if row < h - 1:
            line = f"  theta: {state['theta']:6.1f} deg   r: {state['r']:7.1f} mm   z: {state['z']:7.1f} mm"
            self._safe_addstr(row, 0, line[: w - 1])
            row += 1

        if row < h - 1:
            offset_str = f"{state['wrist_offset']:+.0f}"
            line = (
                f"  wrist offset: {offset_str} deg   wrist rot: {state['wrist_rot']} deg   "
                f"gripper: {state['joints'][5]} deg"
            )
            self._safe_addstr(row, 0, line[: w - 1])
            row += 1

        if row < h - 1:
            line = (
                f"  equilibrium -> theta={state['equil_theta']:.1f} deg  "
                f"r={state['equil_r']:.1f} mm  z={state['equil_z']:.1f} mm"
            )
            self._safe_addstr(row, 0, line[: w - 1], curses.color_pair(_C_DIM))
            row += 1

        return row + 1

    def _draw_tof_status(self, row: int, w: int, h: int, state: dict) -> int:
        if row >= h - 1:
            return row

        tof = state.get("tof_snapshot", {})
        self._safe_addstr(row, 0, "TOF/IR SENSORS", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
        row += 1

        if not tof:
            if row < h - 1:
                self._safe_addstr(row, 0, "  (no Teensy connected - use --teensy-port)", curses.color_pair(_C_DIM))
                row += 1
            return row + 1

        if row < h - 1:
            parts = []
            grids = tof.get("grids", [])
            for ch in (0, 1):
                try:
                    grid = grids[ch]
                    if np.isnan(grid).all():
                        parts.append(f"CH{ch}: --")
                    else:
                        parts.append(f"CH{ch}: {float(np.nanmin(grid)):.0f}mm")
                except Exception:
                    parts.append(f"CH{ch}: --")
            self._safe_addstr(row, 0, ("  " + "   ".join(parts))[: w - 1], curses.color_pair(_C_DIM))
            row += 1

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

        if row < h - 1:
            obs = state.get("obstacle_response", "clear")
            src = state.get("obstacle_source", "")
            dist = state.get("obstacle_dist_mm", -1.0)
            thresh = tof.get("tof_threshold_mm", 300.0)
            if obs == "back_away":
                line = "  OBSTACLE: BACK AWAY"
                attr = curses.color_pair(_C_ERR) | curses.A_BOLD
            elif obs == "replan":
                line = f"  OBSTACLE: REPLAN ({src}, {dist:.0f}mm < {thresh:.0f}mm)"
                attr = curses.color_pair(_C_WARN) | curses.A_BOLD
            else:
                line = f"  Obstacle: CLEAR (threshold: {thresh:.0f}mm)"
                attr = curses.color_pair(_C_OK)
            self._safe_addstr(row, 0, line[: w - 1], attr)
            row += 1

        if row < h - 1:
            imu = tof.get("imu", {})
            imu_online = bool(imu.get("online", False))
            imu_cal = bool(imu.get("calibrated", False))
            imu_state = "CALIBRATED" if imu_cal else "UNCALIBRATED"
            if imu_online:
                ax = float(imu.get("ax_g", 0.0))
                ay = float(imu.get("ay_g", 0.0))
                az = float(imu.get("az_g", 0.0))
                line = f"  ACCELEROMETER (CH2 MPU6050): [{imu_state}] a=[{ax:+.2f},{ay:+.2f},{az:+.2f}] g"
                attr = curses.color_pair(_C_OK) if imu_cal else curses.color_pair(_C_WARN)
            else:
                line = f"  ACCELEROMETER (CH2 MPU6050): [{imu_state}] (offline)"
                attr = curses.color_pair(_C_ERR) if not imu_cal else curses.color_pair(_C_WARN)
            self._safe_addstr(row, 0, line[: w - 1], attr)
            row += 1

        return row + 1

    def _draw_controls(self, row: int, w: int, h: int) -> int:
        if row >= h - 1:
            return row
        self._safe_addstr(row, 0, "CONTROLS", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
        row += 1
        lines = [
            "  A/D: theta +/-5 deg   W/S: reach +/-10mm   Q/E: height +/-10mm",
            "  I/K: wristV +/-5 deg  J/L: wristR +/-5 deg  O/[ open, P/] close",
            "  +/-: slew rate        H: go equil          Shift+H: set equil",
            "  R: toggle side-to-side profile   B: start/stop recording",
            "  T: tuning menu         C: calibrate IMU      M: states menu",
            "  X: sequence editor",
            "  V: ToF viewer (if enabled)   N: ToF screenshot",
            "  F/Shift+F: ToF threshold +/-50mm",
            "  0-8: optional plot controls (plots disabled by default)",
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
            rec = "ON" if state.get("recording_active") else "off"
            saving = "SAVING" if state.get("saving_recording") else ""
            prof = "ON" if state.get("profile_active") else "off"
            opt = "ON" if state.get("optimizer_active") else "off"
            mode = state.get("control_mode", "nominal_tracking")
            samples = int(state.get("record_samples", 0))
            mem = int(state.get("obstacle_memory_count", 0))
            decay = float(state.get("obstacle_decay_s", 0.0))
            ch = int(state.get("tof_channels_enabled", 0))
            pm = "ACTIVE" if bool(state.get("persistent_memory_active", False)) else "inactive"
            lpf = float(state.get("imu_lpf_alpha", 0.25))
            plan_n = int(state.get("future_plan_points", 0))
            line = f"  Mode:{mode} Opt:{opt} Profile:{prof} Rec:{rec}({samples}) {saving} Mem:{mem}@{decay:.1f}s PM:{pm} LPF:{lpf:.2f} Plan:{plan_n} CH:{ch}"
            self._safe_addstr(row, 0, line[: w - 1], curses.color_pair(_C_DIM))
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
            self._safe_addstr(row, 0, full[: w - 1], err_attr)

        return row

    def _safe_addstr(self, row: int, col: int, text: str, attr: int = 0) -> None:
        h, w = self._scr.getmaxyx()
        if row >= h or col >= w or not text:
            return
        text = text[: w - col - 1]
        try:
            self._scr.addstr(row, col, text, attr)
        except curses.error:
            pass




