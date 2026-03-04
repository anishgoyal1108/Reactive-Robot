"""
sequence_editor.py — Curses overlay text editor for pose sequences.

A sequence is a plain-text program where each line is one of:

  STATE_NAME WAIT_MS   — move to the named state, then wait WAIT_MS ms
  REPEAT N             — repeat the whole program N times (default 1)
  # comment            — ignored

Example program
---------------
  # pick-and-place cycle
  HOME    1000
  GRAB     500
  PLACE    800
  HOME     500
  REPEAT 3

Syntax highlighting
-------------------
  green   — valid step (known state + integer wait)
  red     — error (unknown state, non-integer wait, or bad syntax)
  magenta — REPEAT line
  dim     — blank lines and comments

Editor key bindings
-------------------
  Arrow keys / Home / End / PgUp / PgDn  — cursor movement
  Enter                                   — insert newline
  Backspace / Delete                      — delete character
  F3 or Ctrl+K                            — delete current line
  F7                                      — undo
  F8                                      — redo
  F9                                      — clear all lines
  F5  or  Ctrl+R                          — run / restart sequence
  F6  or  Ctrl+T                          — stop running sequence
  ESC                                     — close editor (also stops runner)
"""

import curses
import re
import threading
import time
from typing import Callable, Optional

from .state_library import StateLibrary

# Color pair indices (same scheme as display.py; pairs 1-6 already initialised)
_C_TITLE  = 1
_C_LABEL  = 2
_C_OK     = 3
_C_ERR    = 4
_C_DIM    = 5
_C_WARN   = 6
_C_LNUM   = 8   # line-number gutter color

_NUM_PREFIX = 6   # width of "NNN │ " gutter


def run_sequence_editor(stdscr,
                        state_lib: StateLibrary,
                        run_state_fn: Callable[[str], None]) -> None:
    """
    Full-screen sequence editor overlay.

    Parameters
    ----------
    stdscr       : curses window (from curses.wrapper)
    state_lib    : StateLibrary instance (used for name validation + reference)
    run_state_fn : callable(state_name: str) — moves the arm to the named state.
                   Called from a background thread while the sequence runs.
    """
    if curses.has_colors():
        try:
            curses.init_pair(_C_LNUM, curses.COLOR_WHITE, -1)
        except curses.error:
            pass

    lines:   list                        = [""]
    row_cur: int                         = 0
    col_cur: int                         = 0
    message: str                         = ""
    msg_ok:  bool                        = True
    runner:  Optional[SequenceRunner]    = None
    undo_stack: list                     = []
    redo_stack: list                     = []
    _MAX_HISTORY                         = 200

    def _snapshot():
        return (list(lines), row_cur, col_cur)

    def _push_undo():
        undo_stack.append(_snapshot())
        if len(undo_stack) > _MAX_HISTORY:
            del undo_stack[0]
        redo_stack.clear()

    def _restore(snapshot):
        nonlocal lines, row_cur, col_cur
        snap_lines, snap_row, snap_col = snapshot
        lines = list(snap_lines) if snap_lines else [""]
        row_cur = max(0, min(snap_row, len(lines) - 1))
        col_cur = max(0, min(snap_col, len(lines[row_cur])))

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()

        # ── Header ────────────────────────────────────────────────────────
        title = "  SEQUENCE EDITOR  "
        try:
            stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                          curses.color_pair(_C_TITLE) | curses.A_BOLD)
            stdscr.addstr(1, 0, '─' * min(w - 1, 80))
        except curses.error:
            pass

        _safe(stdscr, 2, 0,
              "  Syntax:  STATE_NAME WAIT_MS  |  REPEAT N  |  # comment",
              curses.color_pair(_C_DIM))

        known     = state_lib.names()
        ndisp     = ", ".join(known[:10]) + (" …" if len(known) > 10 else "")
        _safe(stdscr, 3, 0,
              f"  States:  {ndisp or '(none saved)'}",
              curses.color_pair(_C_DIM))
        try:
            stdscr.addstr(4, 0, '─' * min(w - 1, 80))
        except curses.error:
            pass

        # ── Text area ─────────────────────────────────────────────────────
        TEXT_TOP  = 5
        TEXT_ROWS = max(1, h - TEXT_TOP - 4)

        # Scroll window: keep cursor line visible
        scroll_top = max(0, row_cur - TEXT_ROWS // 2)
        scroll_top = min(scroll_top, max(0, len(lines) - TEXT_ROWS))

        for local_i, abs_i in enumerate(
                range(scroll_top, min(scroll_top + TEXT_ROWS, len(lines)))):
            scr_row = TEXT_TOP + local_i
            if scr_row >= h - 4:
                break
            num_str = f"{abs_i + 1:3d} \u2502 "   # "NNN │ "
            _safe(stdscr, scr_row, 0, num_str, curses.color_pair(_C_LNUM))
            _safe(stdscr, scr_row, len(num_str),
                  lines[abs_i],
                  _line_color(lines[abs_i], state_lib))

        # ── Status bar ────────────────────────────────────────────────────
        running   = runner is not None and runner.running
        if running:
            badge = (f"  \u25cf RUNNING  "
                     f"step {runner.current_step}/{runner.total_steps}  "
                     f"pass {runner.current_pass}/{runner.total_passes}")
            bar_attr = curses.color_pair(_C_WARN) | curses.A_BOLD
        else:
            badge    = ""
            bar_attr = curses.color_pair(_C_DIM)

        _safe(stdscr, h - 3, 0,
              f"  [F3/Ctrl+K] Del line  [F7] Undo  [F8] Redo  [F9] Clear all"
              f"  [F5/Ctrl+R] Run  [F6/Ctrl+T] Stop  [ESC] Back"
              f"  \u2500  Lines: {len(lines)}{badge}",
              bar_attr)

        if message:
            _safe(stdscr, h - 2, 0, f"  {message}",
                  (curses.color_pair(_C_OK)  | curses.A_BOLD if msg_ok
                   else curses.color_pair(_C_ERR) | curses.A_BOLD))

        _safe(stdscr, h - 1, 0,
              f"  Ln {row_cur + 1}/{len(lines)}  Col {col_cur + 1}",
              curses.color_pair(_C_DIM))

        # ── Physical cursor ───────────────────────────────────────────────
        curses.curs_set(1)
        phys_row = TEXT_TOP + (row_cur - scroll_top)
        phys_col = _NUM_PREFIX + col_cur
        if 0 <= phys_row < h - 4 and phys_col < w:
            try:
                stdscr.move(phys_row, phys_col)
            except curses.error:
                pass

        stdscr.refresh()

        # ── Input ─────────────────────────────────────────────────────────
        curses.halfdelay(2)
        try:
            key = stdscr.getch()
        except curses.error:
            key = curses.ERR

        if key == curses.ERR:
            continue

        # ── Exit ──────────────────────────────────────────────────────────
        elif key == 27:   # ESC
            if runner and runner.running:
                runner.stop()
            curses.curs_set(0)
            return

        # ── Cursor movement ───────────────────────────────────────────────
        elif key == curses.KEY_UP:
            if row_cur > 0:
                row_cur -= 1
                col_cur  = min(col_cur, len(lines[row_cur]))

        elif key == curses.KEY_DOWN:
            if row_cur < len(lines) - 1:
                row_cur += 1
                col_cur  = min(col_cur, len(lines[row_cur]))

        elif key == curses.KEY_LEFT:
            if col_cur > 0:
                col_cur -= 1
            elif row_cur > 0:
                row_cur -= 1
                col_cur  = len(lines[row_cur])

        elif key == curses.KEY_RIGHT:
            if col_cur < len(lines[row_cur]):
                col_cur += 1
            elif row_cur < len(lines) - 1:
                row_cur += 1
                col_cur  = 0

        elif key == curses.KEY_HOME:
            col_cur = 0

        elif key == curses.KEY_END:
            col_cur = len(lines[row_cur])

        elif key == curses.KEY_PPAGE:   # Page Up
            row_cur = max(0, row_cur - TEXT_ROWS)
            col_cur = min(col_cur, len(lines[row_cur]))

        elif key == curses.KEY_NPAGE:   # Page Down
            row_cur = min(len(lines) - 1, row_cur + TEXT_ROWS)
            col_cur = min(col_cur, len(lines[row_cur]))

        # ── Editing ───────────────────────────────────────────────────────
        elif key in (10, 13):   # Enter — split line at cursor
            _push_undo()
            cur = lines[row_cur]
            lines[row_cur] = cur[:col_cur]
            lines.insert(row_cur + 1, cur[col_cur:])
            row_cur += 1
            col_cur  = 0

        elif key in (curses.KEY_BACKSPACE, 127, 8):
            changed = False
            if col_cur > 0:
                _push_undo()
                cur = lines[row_cur]
                lines[row_cur] = cur[:col_cur - 1] + cur[col_cur:]
                col_cur -= 1
                changed = True
            elif row_cur > 0:
                _push_undo()
                prev    = lines[row_cur - 1]
                cur     = lines[row_cur]
                col_cur = len(prev)
                lines[row_cur - 1] = prev + cur
                del lines[row_cur]
                row_cur -= 1
                changed = True

        elif key == curses.KEY_DC:   # Delete key
            cur = lines[row_cur]
            if col_cur < len(cur):
                _push_undo()
                lines[row_cur] = cur[:col_cur] + cur[col_cur + 1:]
            elif row_cur < len(lines) - 1:
                _push_undo()
                lines[row_cur] = cur + lines[row_cur + 1]
                del lines[row_cur + 1]

        elif key in (curses.KEY_F3, 11):   # F3 or Ctrl+K
            _push_undo()
            if len(lines) > 1:
                del lines[row_cur]
                row_cur = min(row_cur, len(lines) - 1)
            else:
                lines[0] = ""
                row_cur = 0
            col_cur = min(col_cur, len(lines[row_cur]))
            message = "Line deleted"
            msg_ok = True

        elif key == curses.KEY_F7:   # Undo
            if undo_stack:
                redo_stack.append(_snapshot())
                _restore(undo_stack.pop())
                message = "Undo"
                msg_ok = True
            else:
                message = "Nothing to undo"
                msg_ok = False

        elif key == curses.KEY_F8:   # Redo
            if redo_stack:
                undo_stack.append(_snapshot())
                _restore(redo_stack.pop())
                message = "Redo"
                msg_ok = True
            else:
                message = "Nothing to redo"
                msg_ok = False

        elif key == curses.KEY_F9:   # Clear all lines
            if lines != [""]:
                _push_undo()
                lines = [""]
                row_cur = 0
                col_cur = 0
                message = "Cleared all lines"
                msg_ok = True
            else:
                message = "Already empty"
                msg_ok = False

        # ── Run sequence: F5 or Ctrl+R ────────────────────────────────────
        elif key in (curses.KEY_F5, 18):
            steps, repeat, errors = _parse_sequence(lines, state_lib)
            if errors:
                message = "Error: " + errors[0]
                msg_ok  = False
            elif not steps:
                message = "Nothing to run — add steps like:  HOME 1000"
                msg_ok  = False
            else:
                if runner and runner.running:
                    runner.stop()
                runner  = SequenceRunner(steps, repeat, run_state_fn)
                runner.start()
                message = (f"Started: {len(steps)} step(s) \u00d7 {repeat} pass(es)")
                msg_ok  = True

        # ── Stop sequence: F6 or Ctrl+T ───────────────────────────────────
        elif key in (curses.KEY_F6, 20):
            if runner and runner.running:
                runner.stop()
                message = "Sequence stopped"
                msg_ok  = True
            else:
                message = "No sequence running"
                msg_ok  = False

        # ── Printable ASCII ───────────────────────────────────────────────
        elif 32 <= key <= 126:
            _push_undo()
            cur = lines[row_cur]
            lines[row_cur] = cur[:col_cur] + chr(key) + cur[col_cur:]
            col_cur += 1


# ── Sequence parsing ──────────────────────────────────────────────────────

def _parse_sequence(lines: list, state_lib: StateLibrary):
    """
    Parse sequence lines into steps.

    Returns
    -------
    steps  : list of {'name': str, 'wait_ms': int}
    repeat : int (≥ 1)
    errors : list of str — non-empty means the sequence cannot run
    """
    steps  = []
    repeat = 1
    errors = []

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        # REPEAT N
        m = re.match(r'^REPEAT\s+(\d+)$', line, re.IGNORECASE)
        if m:
            repeat = max(1, int(m.group(1)))
            continue

        # STATE_NAME WAIT_MS
        parts = line.split()
        if len(parts) == 2:
            name, wait_str = parts
            if not wait_str.isdigit():
                errors.append(
                    f"L{i + 1}: wait must be a positive integer, got '{wait_str}'")
                continue
            if state_lib.get_state(name) is None:
                errors.append(f"L{i + 1}: unknown state '{name}'")
                continue
            steps.append({'name': name, 'wait_ms': int(wait_str)})
        else:
            errors.append(
                f"L{i + 1}: expected 'STATE_NAME WAIT_MS', got '{line[:30]}'")

    return steps, repeat, errors


def _line_color(line: str, state_lib: StateLibrary) -> int:
    """Return a curses color attribute for a text line based on validity."""
    s = line.strip()
    if not s or s.startswith('#'):
        return curses.color_pair(_C_DIM)
    if re.match(r'^REPEAT\s+\d+$', s, re.IGNORECASE):
        return curses.color_pair(_C_WARN)   # magenta — control flow
    parts = s.split()
    if len(parts) == 2:
        name, wait = parts
        if wait.isdigit() and state_lib.get_state(name) is not None:
            return curses.color_pair(_C_OK)   # green — valid
        return curses.color_pair(_C_ERR)      # red — bad state or wait
    return curses.color_pair(_C_ERR)          # red — wrong token count


def _safe(stdscr, row: int, col: int, text: str, attr: int = 0) -> None:
    h, w = stdscr.getmaxyx()
    if row >= h or col >= w or not text:
        return
    try:
        stdscr.addstr(row, col, text[:w - col - 1], attr)
    except curses.error:
        pass


# ── Sequence runner ───────────────────────────────────────────────────────

class SequenceRunner:
    """
    Execute a list of pose steps in a background daemon thread.

    For each step:  move arm to state  →  wait wait_ms ms.
    Repeats the whole list `repeat` times.  Can be stopped at any time.
    """

    def __init__(self, steps: list, repeat: int,
                 run_state_fn: Callable[[str], None]):
        self._steps      = steps
        self._repeat     = repeat
        self._run_state  = run_state_fn
        self._thread:    Optional[threading.Thread] = None
        self._stop       = threading.Event()

        self.running:       bool = False
        self.current_step:  int  = 0
        self.current_pass:  int  = 0
        self.total_steps:   int  = len(steps)
        self.total_passes:  int  = repeat

    def start(self) -> None:
        self._stop.clear()
        self.running      = True
        self.current_step = 0
        self.current_pass = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.running = False

    def _run(self) -> None:
        try:
            for p in range(self._repeat):
                self.current_pass = p + 1
                for s, step in enumerate(self._steps):
                    if self._stop.is_set():
                        return
                    self.current_step = s + 1
                    self._run_state(step['name'])
                    # Wait in short increments so stop() is responsive
                    deadline = time.monotonic() + step['wait_ms'] / 1000.0
                    while time.monotonic() < deadline:
                        if self._stop.is_set():
                            return
                        time.sleep(0.02)
        finally:
            self.running = False
