"""
states_menu.py — Curses overlay for browsing and managing saved states.

run_states_menu() is called synchronously from the main control loop.
It takes over the terminal until the user presses ESC or triggers a run.

Controls
--------
  ↑ / ↓       Navigate the list
  Enter        Run selected state (applied to arm on return)
  S            Save current arm pose as a new named state
  R            Rename selected state
  D / Delete   Delete selected state
  ESC          Close menu without action

Returns the selected state dict (joints + IK params) or None if cancelled.
"""

import curses
from typing import Optional

from .state_library import StateLibrary
from .constants     import JOINT_NAMES

# Color pair indices — same as display.py so pairs 1-6 are already initialised
_C_TITLE = 1
_C_LABEL = 2
_C_OK    = 3
_C_ERR   = 4
_C_DIM   = 5
_C_WARN  = 6
_C_SEL   = 7   # selection highlight: black text on cyan background


def run_states_menu(stdscr,
                    state_lib: StateLibrary,
                    current_snapshot: dict) -> Optional[dict]:
    """
    Full-screen states-management overlay.

    Parameters
    ----------
    stdscr          : curses window (from curses.wrapper)
    state_lib       : StateLibrary instance
    current_snapshot: ArmState.snapshot() dict of current arm position
    """
    if curses.has_colors():
        try:
            curses.init_pair(_C_SEL, curses.COLOR_BLACK, curses.COLOR_CYAN)
        except curses.error:
            pass

    curses.curs_set(0)

    cursor  = 0
    message = ""
    msg_ok  = True

    while True:
        names = state_lib.names()
        if names:
            cursor = max(0, min(cursor, len(names) - 1))

        h, w = stdscr.getmaxyx()
        stdscr.erase()

        row = _draw_header(stdscr, "  SAVED STATES  ", w, 0)
        row = _draw_hint(
            stdscr, row, w,
            "[↑/↓] Navigate  [Enter] Run  [S] Save  [R] Rename  [D/Del] Delete  [ESC] Back"
        )

        # ── State list ────────────────────────────────────────────────────
        list_end = h - 9   # leave room for preview + status
        if not names:
            _safe(stdscr, row, 2,
                  "(no states saved — press S to save the current arm pose)",
                  curses.color_pair(_C_DIM))
            row += 1
        else:
            for i, name in enumerate(names):
                if row >= list_end:
                    break
                st   = state_lib.get_state(name)
                jstr = ', '.join(f"{v:3d}" for v in st['joints']) if st else '?'
                if i == cursor:
                    attr = curses.color_pair(_C_SEL) | curses.A_BOLD
                    line = f" ▶ {name:<22}  [{jstr}]"
                else:
                    attr = 0
                    line = f"   {name:<22}  [{jstr}]"
                _safe(stdscr, row, 0, line, attr)
                row += 1

        # ── Preview panel for the selected state ─────────────────────────
        if names and cursor < len(names):
            st = state_lib.get_state(names[cursor])
            if st:
                row += 1
                _safe(stdscr, row, 0, " PREVIEW",
                      curses.color_pair(_C_LABEL) | curses.A_UNDERLINE)
                row += 1
                col2 = w // 2
                for i, (jname, angle) in enumerate(zip(JOINT_NAMES, st['joints'])):
                    if row + (0 if i < 3 else -3) >= h - 4:
                        break
                    if i < 3:
                        _safe(stdscr, row,     2,
                              f"{jname}: {angle:3d}°",
                              curses.color_pair(_C_DIM))
                    else:
                        _safe(stdscr, row - 3, col2,
                              f"{jname}: {angle:3d}°",
                              curses.color_pair(_C_DIM))
                row += 1
                if row < h - 3:
                    _safe(stdscr, row, 2,
                          (f"θ={st['theta']:.1f}°  r={st['r']:.1f} mm  "
                           f"z={st['z']:.1f} mm  wrist_off={st['wrist_offset']:+.0f}°"),
                          curses.color_pair(_C_DIM))
                    row += 1

        # ── Status / feedback ─────────────────────────────────────────────
        if message and h >= 2:
            attr = (curses.color_pair(_C_OK)  | curses.A_BOLD if msg_ok
                    else curses.color_pair(_C_ERR) | curses.A_BOLD)
            _safe(stdscr, h - 2, 0, f"  {message}", attr)

        stdscr.refresh()

        # ── Input ─────────────────────────────────────────────────────────
        curses.halfdelay(10)
        try:
            key = stdscr.getch()
        except curses.error:
            continue

        if key == curses.ERR:
            continue

        elif key == 27:   # ESC
            return None

        elif key == curses.KEY_UP:
            cursor  = max(0, cursor - 1)
            message = ""

        elif key == curses.KEY_DOWN:
            cursor  = min(max(0, len(names) - 1), cursor + 1)
            message = ""

        elif key in (curses.KEY_ENTER, 10, 13):
            if names and cursor < len(names):
                return state_lib.get_state(names[cursor])
            message = "No state selected"
            msg_ok  = False

        elif key in (ord('s'), ord('S')):
            name = _prompt_name(stdscr, "Save state as: ", w)
            if name:
                state_lib.save_state(name, current_snapshot)
                message = f"Saved '{name}'"
                msg_ok  = True
                names2  = state_lib.names()
                if name in names2:
                    cursor = names2.index(name)
            else:
                message = "Save cancelled"
                msg_ok  = False

        elif key in (ord('r'), ord('R')):
            if names and cursor < len(names):
                old = names[cursor]
                new = _prompt_name(stdscr, f"Rename '{old}' to: ", w)
                if new and new != old:
                    state_lib.rename_state(old, new)
                    message = f"Renamed '{old}' → '{new}'"
                    msg_ok  = True
                    names2  = state_lib.names()
                    if new in names2:
                        cursor = names2.index(new)
                else:
                    message = "Rename cancelled"
                    msg_ok  = False

        elif key in (ord('d'), ord('D'), curses.KEY_DC):
            if names and cursor < len(names):
                name = names[cursor]
                state_lib.delete_state(name)
                message = f"Deleted '{name}'"
                msg_ok  = True
                cursor  = max(0, cursor - 1)

    return None   # unreachable; satisfies type checker


# ── Helpers ───────────────────────────────────────────────────────────────

def _draw_header(stdscr, title: str, w: int, row: int) -> int:
    col = max(0, (w - len(title)) // 2)
    try:
        stdscr.addstr(row,     col, title,
                      curses.color_pair(_C_TITLE) | curses.A_BOLD)
        stdscr.addstr(row + 1, 0,   '─' * min(w - 1, 80))
    except curses.error:
        pass
    return row + 2


def _draw_hint(stdscr, row: int, w: int, text: str) -> int:
    _safe(stdscr, row, 0, "  " + text, curses.color_pair(_C_DIM))
    return row + 2


def _prompt_name(stdscr, prompt: str, w: int) -> str:
    """Single-line text input at the bottom of the screen."""
    h, _ = stdscr.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    _safe(stdscr, h - 1, 0, prompt + " " * max(0, w - len(prompt) - 1))
    stdscr.refresh()
    try:
        raw  = stdscr.getstr(h - 1, len(prompt), 32)
        name = raw.decode('utf-8', errors='replace').strip()
    except (curses.error, UnicodeDecodeError):
        name = ""
    finally:
        curses.noecho()
        curses.curs_set(0)
    return name


def _safe(stdscr, row: int, col: int, text: str, attr: int = 0) -> None:
    h, w = stdscr.getmaxyx()
    if row >= h or col >= w or not text:
        return
    try:
        stdscr.addstr(row, col, text[:w - col - 1], attr)
    except curses.error:
        pass
