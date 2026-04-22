"""
sequence_editor.py — Curses overlay text editor for pose sequences.

Legacy format (backwards compatible)
------------------------------------
  STATE_NAME WAIT_MS   — move to the named state, then wait WAIT_MS ms
  REPEAT N             — repeat the whole program N times (default 1)
  # comment            — ignored

Extended DSL (Phase 2, braccio_ctrl.dsl)
----------------------------------------
  MOVE STATE_NAME [WAIT ms]
  SET <B|S|E|WV|WR|G> <deg> [WAIT ms]
  WAIT <ms>
  HOME
  REPEAT N { ... }
  IF <sensor> <op> <num> { ... } [ELSE { ... }]
  WHILE <sensor> <op> <num> { ... }
  STATE NAME { JOINTS b s e wv wr g }
  OBSTACLE NAME { POS x y z  RADIUS r  SHAPE BOX|SPHERE|CYLINDER }

Sensor expressions: TOF[0..3], IR, JOINT[B..G]

Syntax highlighting
-------------------
  green   — legacy step (known state + integer wait)
  magenta — legacy REPEAT line
  cyan    — extended DSL line (parsed successfully as part of the program)
  red     — line the parser rejected
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
from typing import Callable, Optional

from .dsl import SequenceRunner, parse, parse_auto
from .state_library import StateLibrary

# Color pair indices (same scheme as display.py; pairs 1-6 already initialised)
_C_TITLE = 1
_C_LABEL = 2
_C_OK = 3
_C_ERR = 4
_C_DIM = 5
_C_WARN = 6
_C_LNUM = 8  # line-number gutter color

_NUM_PREFIX = 6  # width of "NNN │ " gutter


def run_sequence_editor(
    stdscr, state_lib: StateLibrary, run_state_fn: Callable[[str], None]
) -> None:
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

    lines: list = [""]
    row_cur: int = 0
    col_cur: int = 0
    message: str = ""
    msg_ok: bool = True
    runner: Optional[SequenceRunner] = None
    undo_stack: list = []
    redo_stack: list = []
    _MAX_HISTORY = 200

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
            stdscr.addstr(
                0,
                max(0, (w - len(title)) // 2),
                title,
                curses.color_pair(_C_TITLE) | curses.A_BOLD,
            )
            stdscr.addstr(1, 0, "─" * min(w - 1, 80))
        except curses.error:
            pass

        _safe(
            stdscr,
            2,
            0,
            "  Syntax:  STATE_NAME WAIT_MS  |  REPEAT N  |  # comment",
            curses.color_pair(_C_DIM),
        )

        known = state_lib.names()
        ndisp = ", ".join(known[:10]) + (" …" if len(known) > 10 else "")
        _safe(
            stdscr,
            3,
            0,
            f"  States:  {ndisp or '(none saved)'}",
            curses.color_pair(_C_DIM),
        )
        try:
            stdscr.addstr(4, 0, "─" * min(w - 1, 80))
        except curses.error:
            pass

        # ── Text area ─────────────────────────────────────────────────────
        TEXT_TOP = 5
        TEXT_ROWS = max(1, h - TEXT_TOP - 4)

        # Scroll window: keep cursor line visible
        scroll_top = max(0, row_cur - TEXT_ROWS // 2)
        scroll_top = min(scroll_top, max(0, len(lines) - TEXT_ROWS))

        for local_i, abs_i in enumerate(
            range(scroll_top, min(scroll_top + TEXT_ROWS, len(lines)))
        ):
            scr_row = TEXT_TOP + local_i
            if scr_row >= h - 4:
                break
            num_str = f"{abs_i + 1:3d} \u2502 "  # "NNN │ "
            _safe(stdscr, scr_row, 0, num_str, curses.color_pair(_C_LNUM))
            _safe(
                stdscr,
                scr_row,
                len(num_str),
                lines[abs_i],
                _line_color(lines[abs_i], state_lib),
            )

        # ── Status bar ────────────────────────────────────────────────────
        if runner is not None and runner.running:
            badge = (
                f"  \u25cf RUNNING  "
                f"step {runner.current_step}/{runner.total_steps}  "
                f"pass {runner.current_pass}/{runner.total_passes}"
            )
            bar_attr = curses.color_pair(_C_WARN) | curses.A_BOLD
        else:
            badge = ""
            bar_attr = curses.color_pair(_C_DIM)

        _safe(
            stdscr,
            h - 3,
            0,
            f"  [F3/Ctrl+K] Del line  [F7] Undo  [F8] Redo  [F9] Clear all"
            f"  [F5/Ctrl+R] Run  [F6/Ctrl+T] Stop  [ESC] Back"
            f"  \u2500  Lines: {len(lines)}{badge}",
            bar_attr,
        )

        if message:
            _safe(
                stdscr,
                h - 2,
                0,
                f"  {message}",
                (
                    curses.color_pair(_C_OK) | curses.A_BOLD
                    if msg_ok
                    else curses.color_pair(_C_ERR) | curses.A_BOLD
                ),
            )

        _safe(
            stdscr,
            h - 1,
            0,
            f"  Ln {row_cur + 1}/{len(lines)}  Col {col_cur + 1}",
            curses.color_pair(_C_DIM),
        )

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
        elif key == 27:  # ESC
            if runner and runner.running:
                runner.stop()
            curses.curs_set(0)
            return

        # ── Cursor movement ───────────────────────────────────────────────
        elif key == curses.KEY_UP:
            if row_cur > 0:
                row_cur -= 1
                col_cur = min(col_cur, len(lines[row_cur]))

        elif key == curses.KEY_DOWN:
            if row_cur < len(lines) - 1:
                row_cur += 1
                col_cur = min(col_cur, len(lines[row_cur]))

        elif key == curses.KEY_LEFT:
            if col_cur > 0:
                col_cur -= 1
            elif row_cur > 0:
                row_cur -= 1
                col_cur = len(lines[row_cur])

        elif key == curses.KEY_RIGHT:
            if col_cur < len(lines[row_cur]):
                col_cur += 1
            elif row_cur < len(lines) - 1:
                row_cur += 1
                col_cur = 0

        elif key == curses.KEY_HOME:
            col_cur = 0

        elif key == curses.KEY_END:
            col_cur = len(lines[row_cur])

        elif key == curses.KEY_PPAGE:  # Page Up
            row_cur = max(0, row_cur - TEXT_ROWS)
            col_cur = min(col_cur, len(lines[row_cur]))

        elif key == curses.KEY_NPAGE:  # Page Down
            row_cur = min(len(lines) - 1, row_cur + TEXT_ROWS)
            col_cur = min(col_cur, len(lines[row_cur]))

        # ── Editing ───────────────────────────────────────────────────────
        elif key in (10, 13):  # Enter — split line at cursor
            _push_undo()
            cur = lines[row_cur]
            lines[row_cur] = cur[:col_cur]
            lines.insert(row_cur + 1, cur[col_cur:])
            row_cur += 1
            col_cur = 0

        elif key in (curses.KEY_BACKSPACE, 127, 8):
            changed = False
            if col_cur > 0:
                _push_undo()
                cur = lines[row_cur]
                lines[row_cur] = cur[: col_cur - 1] + cur[col_cur:]
                col_cur -= 1
                changed = True
            elif row_cur > 0:
                _push_undo()
                prev = lines[row_cur - 1]
                cur = lines[row_cur]
                col_cur = len(prev)
                lines[row_cur - 1] = prev + cur
                del lines[row_cur]
                row_cur -= 1
                changed = True

        elif key == curses.KEY_DC:  # Delete key
            cur = lines[row_cur]
            if col_cur < len(cur):
                _push_undo()
                lines[row_cur] = cur[:col_cur] + cur[col_cur + 1 :]
            elif row_cur < len(lines) - 1:
                _push_undo()
                lines[row_cur] = cur + lines[row_cur + 1]
                del lines[row_cur + 1]

        elif key in (curses.KEY_F3, 11):  # F3 or Ctrl+K
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

        elif key == curses.KEY_F7:  # Undo
            if undo_stack:
                redo_stack.append(_snapshot())
                _restore(undo_stack.pop())
                message = "Undo"
                msg_ok = True
            else:
                message = "Nothing to undo"
                msg_ok = False

        elif key == curses.KEY_F8:  # Redo
            if redo_stack:
                undo_stack.append(_snapshot())
                _restore(redo_stack.pop())
                message = "Redo"
                msg_ok = True
            else:
                message = "Nothing to redo"
                msg_ok = False

        elif key == curses.KEY_F9:  # Clear all lines
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
            text = "\n".join(lines)
            program, parse_errors = parse_auto(text)
            static_errors = _static_check(program, state_lib)
            errors = list(parse_errors) + list(static_errors)
            if errors:
                message = "Error: " + errors[0]
                msg_ok = False
            elif not program.body:
                message = "Nothing to run — add steps like:  HOME 1000"
                msg_ok = False
            else:
                if runner and runner.running:
                    runner.stop()
                runner = SequenceRunner(program, run_state_fn)
                runner.start()
                message = (
                    f"Started: {runner.total_steps} step(s)"
                    f" \u00d7 {runner.total_passes} pass(es)"
                )
                msg_ok = True

        # ── Stop sequence: F6 or Ctrl+T ───────────────────────────────────
        elif key in (curses.KEY_F6, 20):
            if runner and runner.running:
                runner.stop()
                message = "Sequence stopped"
                msg_ok = True
            else:
                message = "No sequence running"
                msg_ok = False

        # ── Printable ASCII ───────────────────────────────────────────────
        elif 32 <= key <= 126:
            _push_undo()
            cur = lines[row_cur]
            lines[row_cur] = cur[:col_cur] + chr(key) + cur[col_cur:]
            col_cur += 1


# ── Sequence validation ───────────────────────────────────────────────────


def _static_check(program, state_lib: StateLibrary) -> list:
    """
    Walk a parsed Program AST and flag semantic errors that the Lark
    grammar can't catch:

      * MoveStmt referencing an unknown saved state
      * legacy REPEAT with 0 body (empty program)

    Returns a list of human-readable errors (empty on success). Purely
    static — no side effects, so the curses editor can call it on
    every F5 without touching hardware.
    """
    from .dsl import MoveStmt, Program  # local import avoids module cycle

    errors: list = []
    if not isinstance(program, Program):
        return ["internal: parse did not return a Program"]
    for _depth, stmt in program.walk():
        if isinstance(stmt, MoveStmt):
            if state_lib.get_state(stmt.state_name) is None:
                errors.append(
                    f"L{stmt.line}: unknown state '{stmt.state_name}'"
                )
    return errors


# Reused by the curses renderer — matches the legacy two-token line
# format so lines that look valid in the old grammar get the green
# highlight. Anything else falls back to extended-DSL highlighting.
_RE_LEGACY_MOVE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(\d+)$")
_RE_LEGACY_REPEAT = re.compile(r"^REPEAT\s+\d+$", re.IGNORECASE)


def _line_color(line: str, state_lib: StateLibrary) -> int:
    """
    Return a curses color attribute for a line of source text.

    Decision order:
      1. blank / comment   → dim
      2. legacy REPEAT N   → magenta
      3. legacy STATE WAIT → green (known state) or red (unknown state)
      4. anything that parses as new-DSL on its own → cyan (neutral info)
      5. rejected by both parsers → red
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return curses.color_pair(_C_DIM)
    if _RE_LEGACY_REPEAT.match(s):
        return curses.color_pair(_C_WARN)  # magenta — control flow
    m = _RE_LEGACY_MOVE.match(s)
    if m:
        name = m.group(1)
        if state_lib.get_state(name) is not None:
            return curses.color_pair(_C_OK)  # green — known legacy line
        return curses.color_pair(_C_ERR)  # red — unknown state name
    # Try the extended DSL parser on the single line. This is tolerant
    # of block heads like "IF TOF[0] < 250 {" because Lark recognises
    # the open brace, but the matching "}" on a separate line is still
    # highlighted individually. Rather than maintaining a per-line
    # status from a whole-text parse, we treat anything that survives
    # the single-line parse as "looks OK".
    try:
        parse(s)
        return curses.color_pair(_C_OK)  # green — recognised DSL line
    except Exception:
        pass
    # Braces and partial lines don't parse in isolation but are still
    # valid inside a multi-line program — mark them neutral so the
    # user isn't scared by a sea of red while typing a block.
    if s in {"{", "}"} or s.endswith("{") or s.startswith("}"):
        return curses.color_pair(_C_WARN)
    return curses.color_pair(_C_ERR)


def _safe(stdscr, row: int, col: int, text: str, attr: int = 0) -> None:
    h, w = stdscr.getmaxyx()
    if row >= h or col >= w or not text:
        return
    try:
        stdscr.addstr(row, col, text[: w - col - 1], attr)
    except curses.error:
        pass


# The SequenceRunner used by this module now lives in
# ``braccio_ctrl.dsl.interpreter`` — it wraps a Program AST instead of
# a flat step list. Imported at the top of the file and reused by F5.
