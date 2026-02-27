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
from .constants import JOINT_LIMITS, JOINT_NAMES

# Color pair indices
_C_TITLE   = 1
_C_LABEL   = 2
_C_OK      = 3
_C_ERR     = 4
_C_DIM     = 5
_C_WARN    = 6

_BAR_WIDTH = 20   # characters for the progress bar


class CursesDisplay:
    def __init__(self, stdscr):
        self._scr = stdscr
        curses.curs_set(0)
        self._colors_ready = False
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(_C_TITLE, curses.COLOR_CYAN,    -1)
            curses.init_pair(_C_LABEL, curses.COLOR_YELLOW,  -1)
            curses.init_pair(_C_OK,    curses.COLOR_GREEN,   -1)
            curses.init_pair(_C_ERR,   curses.COLOR_RED,     -1)
            curses.init_pair(_C_DIM,   curses.COLOR_WHITE,   -1)
            curses.init_pair(_C_WARN,  curses.COLOR_MAGENTA, -1)
            self._colors_ready = True

    # ── Public API ────────────────────────────────────────────────────────

    def render(self, state: dict) -> None:
        """Redraw the full UI from a state snapshot."""
        self._scr.erase()
        h, w = self._scr.getmaxyx()
        row = 0

        row = self._draw_title(row, w)
        row = self._draw_joints(row, w, h, state['joints'])
        row = self._draw_ik_state(row, w, h, state)
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
        self._safe_addstr(row, col, title,
                          curses.color_pair(_C_TITLE) | curses.A_BOLD)
        return row + 2

    def _draw_joints(self, row: int, w: int, h: int, joints: list) -> int:
        self._safe_addstr(row, 0, "JOINT STATUS",
                          curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
        row += 1
        for i, (name, angle) in enumerate(zip(JOINT_NAMES, joints)):
            if row >= h - 1:
                break
            lo, hi = JOINT_LIMITS[i]
            pct    = (angle - lo) / max(hi - lo, 1)
            filled = int(pct * _BAR_WIDTH)
            bar    = '#' * filled + '-' * (_BAR_WIDTH - filled)
            line   = f"  {name}: {angle:3d}°  [{bar}]  ({lo}–{hi}°)"
            self._safe_addstr(row, 0, line[:w - 1])
            row += 1
        return row + 1

    def _draw_ik_state(self, row: int, w: int, h: int, state: dict) -> int:
        if row >= h - 1:
            return row
        self._safe_addstr(row, 0, "IK STATE",
                          curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
        row += 1

        if row < h - 1:
            line = (f"  theta: {state['theta']:6.1f}°   "
                    f"r: {state['r']:7.1f} mm   "
                    f"z: {state['z']:7.1f} mm")
            self._safe_addstr(row, 0, line[:w - 1])
            row += 1

        if row < h - 1:
            offset_str = f"{state['wrist_offset']:+.0f}"
            line = (f"  wrist offset: {offset_str}°   "
                    f"wrist rot: {state['wrist_rot']}°   "
                    f"gripper: {state['joints'][5]}°")
            self._safe_addstr(row, 0, line[:w - 1])
            row += 1

        if row < h - 1:
            line = (f"  equilibrium →  "
                    f"theta={state['equil_theta']:.1f}°  "
                    f"r={state['equil_r']:.1f} mm  "
                    f"z={state['equil_z']:.1f} mm")
            self._safe_addstr(row, 0, line[:w - 1],
                              curses.color_pair(_C_DIM))
            row += 1

        return row + 1

    def _draw_controls(self, row: int, w: int, h: int) -> int:
        if row >= h - 1:
            return row
        self._safe_addstr(row, 0, "CONTROLS",
                          curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
        row += 1
        lines = [
            "  A/D: theta ±5°     W/S: reach ±10mm    Q/E: height ±10mm",
            "  I/K: wristV ±5°    J/L: wristR ±5°     O/[ : grip open",
            "  P/] : grip close   +/-: slew rate",
            "  H: go to equil     Shift+H: set equil   ESC: quit",
            "  [plot window]  R: reset plot   S: screenshot   L: toggle log",
        ]
        for line in lines:
            if row >= h - 1:
                break
            self._safe_addstr(row, 0, line[:w - 1],
                              curses.color_pair(_C_DIM))
            row += 1
        return row + 1

    def _draw_status(self, row: int, w: int, h: int, state: dict) -> int:
        if row >= h - 1:
            return row

        delta     = state['delta']
        slew_dps  = delta * 100
        connected = state['connected']
        conn_str  = "CONNECTED" if connected else "DISCONNECTED"
        conn_attr = (curses.color_pair(_C_OK) | curses.A_BOLD
                     if connected
                     else curses.color_pair(_C_ERR) | curses.A_BOLD)

        prefix = f"  Slew: {delta} deg/tick ({slew_dps} deg/s)   Serial: "
        self._safe_addstr(row, 0, prefix[:w - 1])
        col = len(prefix)
        if col < w - 1:
            self._safe_addstr(row, col, conn_str[:w - col - 1], conn_attr)
        row += 1

        if row < h - 1:
            cmd_disp  = state['last_cmd'][:30]
            resp_disp = state['last_resp'][:30]
            line = f"  Sent: {cmd_disp!r}   Recv: {resp_disp!r}"
            self._safe_addstr(row, 0, line[:w - 1],
                              curses.color_pair(_C_DIM))
            row += 1

        if row < h - 1 and state['last_error']:
            err = f"  ERROR: {state['last_error']}"
            self._safe_addstr(row, 0, err[:w - 1],
                              curses.color_pair(_C_ERR) | curses.A_BOLD)
            row += 1

        return row

    # ── Safe write helper ─────────────────────────────────────────────────

    def _safe_addstr(self, row: int, col: int, text: str,
                     attr: int = 0) -> None:
        """Write text without raising on boundary overruns."""
        h, w = self._scr.getmaxyx()
        if row >= h or col >= w or not text:
            return
        text = text[:w - col - 1]   # clamp to available width
        try:
            self._scr.addstr(row, col, text, attr)
        except curses.error:
            pass
