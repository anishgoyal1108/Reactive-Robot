"""Curses tuning overlay for optimizer and sensing parameters."""

from __future__ import annotations

import curses
from typing import Callable

_C_TITLE = 1
_C_LABEL = 2
_C_OK = 3
_C_ERR = 4
_C_DIM = 5
_C_WARN = 6
_C_SEL = 7


TOGGLE_KEYS = {"persistent_memory_active"}


def run_tuning_menu(stdscr, current_values: dict, apply_callback: Callable[[dict], None]) -> None:
    if curses.has_colors():
        try:
            curses.init_pair(_C_SEL, curses.COLOR_BLACK, curses.COLOR_CYAN)
        except curses.error:
            pass

    curses.curs_set(0)

    current = dict(current_values)
    pending: dict[str, object] = {}
    message = ""
    msg_ok = True
    cursor = 0
    editing_key: str | None = None
    edit_buf = ""

    rows = [
        ("w_track_theta", "w_track_theta", ""),
        ("w_smooth_theta", "w_smooth_theta", ""),
        ("w_track_r", "w_track_r", ""),
        ("w_smooth_r", "w_smooth_r", ""),
        ("w_avoid_r", "w_avoid_r", ""),
        ("w_track_z", "w_track_z", ""),
        ("w_smooth_z", "w_smooth_z", ""),
        ("w_avoid_z", "w_avoid_z", ""),
        ("tof_threshold_mm", "Distance Threshold", "mm"),
        ("obstacle_decay_s", "Obstacle Decay Time", "s"),
        ("persistent_memory_active", "Persistent Memory", ""),
        ("imu_lpf_alpha", "IMU LPF alpha", ""),
        ("__update__", "Update", ""),
        ("__exit__", "Exit", ""),
    ]

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()

        title = "  TUNING  "
        _safe(stdscr, 0, max(0, (w - len(title)) // 2), title, curses.color_pair(_C_TITLE) | curses.A_BOLD)
        _safe(stdscr, 1, 0, "-" * max(0, min(100, w - 1)), curses.color_pair(_C_DIM))

        _draw_objective(stdscr, 2, w)

        _safe(
            stdscr,
            6,
            0,
            "Use Up/Down to navigate, Enter to select/edit, number keys for numeric fields.",
            curses.color_pair(_C_DIM),
        )
        _safe(stdscr, 7, 0, "Pending values are shown in parentheses and apply only on Update.", curses.color_pair(_C_DIM))

        row_y = 9
        for i, (key, label, unit) in enumerate(rows):
            selected = i == cursor
            attr = curses.color_pair(_C_SEL) | curses.A_BOLD if selected else 0

            if key == "__update__":
                text = "[ Update ]"
                _safe(stdscr, row_y + i, 2, text, attr)
                continue
            if key == "__exit__":
                text = "[ Exit ]"
                _safe(stdscr, row_y + i, 2, text, attr)
                continue

            val = pending.get(key, current.get(key, 0.0))
            val_text = _format_value(key, val, pending=(key in pending))
            val_attr = (curses.color_pair(_C_WARN) | curses.A_BOLD) if (key in pending) else curses.color_pair(_C_DIM)

            text = f"{label:<20}: "
            _safe(stdscr, row_y + i, 2, text, attr)
            _safe(stdscr, row_y + i, 26, val_text, val_attr | (curses.A_BOLD if selected else 0))
            if unit:
                _safe(stdscr, row_y + i, 26 + len(val_text) + 1, f"[{unit}]", curses.color_pair(_C_DIM))

        status_y = min(h - 2, row_y + len(rows) + 1)
        if editing_key is not None:
            _safe(stdscr, status_y, 0, f"Editing {editing_key}: {edit_buf}", curses.color_pair(_C_WARN) | curses.A_BOLD)
        elif message:
            attr = curses.color_pair(_C_OK) | curses.A_BOLD if msg_ok else curses.color_pair(_C_ERR) | curses.A_BOLD
            _safe(stdscr, status_y, 0, message[: w - 1], attr)

        stdscr.refresh()

        curses.halfdelay(1)
        try:
            key = stdscr.getch()
        except curses.error:
            continue
        if key == curses.ERR:
            continue

        if editing_key is not None:
            if key in (10, 13):
                try:
                    val = float(edit_buf)
                    pending[editing_key] = val
                    message = f"Pending: {editing_key} = ({val:g})"
                    msg_ok = True
                except ValueError:
                    message = f"Invalid value: {edit_buf}"
                    msg_ok = False
                editing_key = None
                edit_buf = ""
            elif key in (27,):
                editing_key = None
                edit_buf = ""
                message = "Edit cancelled"
                msg_ok = False
            elif key in (curses.KEY_BACKSPACE, 127, 8):
                edit_buf = edit_buf[:-1]
            elif key == ord("-") and len(edit_buf) == 0:
                edit_buf = "-"
            elif key == ord(".") and "." not in edit_buf:
                edit_buf += "."
            elif ord("0") <= key <= ord("9"):
                edit_buf += chr(key)
            continue

        if key == 27:
            return
        if key == curses.KEY_UP:
            cursor = max(0, cursor - 1)
            continue
        if key == curses.KEY_DOWN:
            cursor = min(len(rows) - 1, cursor + 1)
            continue
        if key not in (10, 13):
            continue

        sel_key, _, _ = rows[cursor]
        if sel_key == "__exit__":
            return

        if sel_key == "__update__":
            if not pending:
                message = "No pending changes to apply"
                msg_ok = False
                continue
            merged = dict(current)
            merged.update(pending)
            try:
                apply_callback(merged)
                current = merged
                pending.clear()
                message = "Updated optimizer and sensing parameters"
                msg_ok = True
            except Exception as exc:
                message = f"Update failed: {exc}"
                msg_ok = False
            continue

        if sel_key in TOGGLE_KEYS:
            cur = bool(pending.get(sel_key, current.get(sel_key, False)))
            pending[sel_key] = (not cur)
            message = f"Pending: {sel_key} = ({'ACTIVE' if not cur else 'INACTIVE'})"
            msg_ok = True
            continue

        # Numeric parameter -> enter edit mode.
        editing_key = sel_key
        base_val = pending.get(sel_key, current.get(sel_key, 0.0))
        edit_buf = f"{float(base_val):g}"
        message = ""


def _format_value(key: str, value: object, pending: bool) -> str:
    if key in TOGGLE_KEYS:
        txt = "ACTIVE" if bool(value) else "INACTIVE"
        return f"({txt})" if pending else txt

    try:
        f = float(value)
        txt = f"{f:g}"
    except Exception:
        txt = str(value)
    return f"({txt})" if pending else txt


def _draw_objective(stdscr, row: int, width: int) -> None:
    _safe(stdscr, row, 0, "Simplified Minimization (Linear-Relaxed):", curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
    row += 1
    _safe(stdscr, row, 2, "min  J = ", curses.color_pair(_C_DIM))
    _safe(stdscr, row, 11, "w_track", curses.color_pair(_C_WARN) | curses.A_BOLD)
    _safe(stdscr, row, 19, "*||u-u_nom||^2 + ", curses.color_pair(_C_DIM))
    _safe(stdscr, row, 36, "w_smooth", curses.color_pair(_C_WARN) | curses.A_BOLD)
    _safe(stdscr, row, 45, "*||u-u_prev||^2 + ", curses.color_pair(_C_DIM))
    _safe(stdscr, row, 66, "w_avoid", curses.color_pair(_C_WARN) | curses.A_BOLD)
    _safe(stdscr, row, 74, "*||u-u_avoid||^2", curses.color_pair(_C_DIM))
    row += 1
    _safe(stdscr, row, 2, "Obstacle logic uses threshold, decay, and ", curses.color_pair(_C_DIM))
    _safe(stdscr, row, 39, "Persistent Memory", curses.color_pair(_C_WARN) | curses.A_BOLD)
    _safe(stdscr, row, 57, ". IMU uses configurable ", curses.color_pair(_C_DIM))
    _safe(stdscr, row, 80, "LPF alpha", curses.color_pair(_C_WARN) | curses.A_BOLD)


def _safe(stdscr, row: int, col: int, text: str, attr: int = 0) -> None:
    h, w = stdscr.getmaxyx()
    if row >= h or col >= w or not text:
        return
    try:
        stdscr.addstr(row, col, text[: w - col - 1], attr)
    except curses.error:
        pass
