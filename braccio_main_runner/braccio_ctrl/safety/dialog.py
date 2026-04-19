"""
dialog.py — Manual-mode refusal dialog.

Research §8: when a manual keyboard intent would collide, the safety stack
refuses and presents the user with options:

    [R]etreat to last safe pose
    [A]xis    try a different axis
    [S]tep    reduce step size and retry
    [O]verride  accept one-step forward over user confirmation  [DANGEROUS]
    [C]ancel

This module ships:

  * ``RefusalDialog`` — pure-logic state machine that doesn't know about
    curses. Takes the ``RefusalInfo`` payload (see ``behavior.RefusalInfo``)
    and a single key character, returns a ``DialogChoice`` enum.
  * ``render_to_curses(stdscr, info)`` — stateless helper that draws the
    dialog onto an arbitrary curses window. The controller decides where
    and when to call it.

The split keeps the dialog testable without a curses terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

try:
    import curses as _curses  # Only used by render_to_curses().
except Exception:  # pragma: no cover
    _curses = None  # type: ignore[assignment]


class DialogChoice(Enum):
    RETREAT = "retreat"
    TRY_AXIS = "try_axis"
    REDUCE_STEP = "reduce_step"
    OVERRIDE = "override"
    CANCEL = "cancel"


_KEY_TO_CHOICE: dict[int, DialogChoice] = {
    # accept both upper- and lower-case so Shift-state doesn't matter
    ord("r"): DialogChoice.RETREAT,
    ord("R"): DialogChoice.RETREAT,
    ord("a"): DialogChoice.TRY_AXIS,
    ord("A"): DialogChoice.TRY_AXIS,
    ord("s"): DialogChoice.REDUCE_STEP,
    ord("S"): DialogChoice.REDUCE_STEP,
    ord("o"): DialogChoice.OVERRIDE,
    ord("O"): DialogChoice.OVERRIDE,
    ord("c"): DialogChoice.CANCEL,
    ord("C"): DialogChoice.CANCEL,
    27:       DialogChoice.CANCEL,    # ESC
}


@dataclass
class DialogLine:
    text: str
    highlight: bool = False


class RefusalDialog:
    """Pure-Python refusal dialog state.

    Usage
    -----
        dlg = RefusalDialog()
        dlg.open(info)
        while dlg.is_open:
            key = stdscr.getch()
            choice = dlg.handle_key(key)
            if choice is not None:
                ...use choice...
    """

    def __init__(self):
        self._info = None  # type: Optional[object]

    @property
    def is_open(self) -> bool:
        return self._info is not None

    @property
    def info(self):
        return self._info

    def open(self, info) -> None:
        self._info = info

    def close(self) -> None:
        self._info = None

    def handle_key(self, key: int) -> Optional[DialogChoice]:
        """Consume one keypress. Returns a ``DialogChoice`` if the dialog
        is open and the press is a recognised binding (and closes the
        dialog); otherwise returns ``None``."""
        if not self.is_open:
            return None
        choice = _KEY_TO_CHOICE.get(int(key))
        if choice is None:
            return None
        self.close()
        return choice

    def render_lines(self) -> list[DialogLine]:
        """Render the dialog as a list of lines for any display backend.

        The curses renderer in ``render_to_curses`` just walks this list
        and calls ``addstr``. Tests call ``render_lines`` directly to make
        assertions on the content.
        """
        if self._info is None:
            return []
        info = self._info
        lines: list[DialogLine] = []
        lines.append(DialogLine("  SAFETY REFUSAL  ", highlight=True))
        reason = getattr(info, "reason", "unknown")
        lines.append(DialogLine(f"  The planner refused this move."))
        lines.append(DialogLine(f"  reason: {reason}"))
        nl = getattr(info, "nearest_link", None)
        gap = float(getattr(info, "nearest_gap_mm", 0.0) or 0.0)
        if nl:
            lines.append(DialogLine(
                f"  nearest link: {nl}  (gap {gap:.0f} mm)"
            ))
        attempted = list(getattr(info, "attempted_q", []) or [])
        current = list(getattr(info, "current_q", []) or [])
        if attempted and current:
            lines.append(DialogLine(
                f"  current: {current}"
            ))
            lines.append(DialogLine(
                f"  target : {attempted}"
            ))
        lines.append(DialogLine(""))
        lines.append(DialogLine("  [R]etreat    [A]xis    [S]tep-down", highlight=True))
        lines.append(DialogLine("  [O]verride   [C]ancel / Esc",      highlight=True))
        return lines


def render_to_curses(stdscr, dialog: RefusalDialog) -> None:
    """Draw the dialog centred inside ``stdscr``. No-op if curses is
    unavailable or the dialog is closed."""
    if _curses is None or not dialog.is_open:  # pragma: no cover
        return
    lines = dialog.render_lines()
    if not lines:
        return
    try:
        h, w = stdscr.getmaxyx()
    except Exception:  # pragma: no cover
        return
    height = len(lines) + 4
    width = max(len(line.text) for line in lines) + 4
    height = min(height, h - 2)
    width = min(width, w - 2)
    y0 = max(0, (h - height) // 2)
    x0 = max(0, (w - width) // 2)
    try:
        win = _curses.newwin(height, width, y0, x0)
        win.box()
        for i, line in enumerate(lines):
            attr = _curses.A_REVERSE if line.highlight else _curses.A_NORMAL
            try:
                win.addstr(i + 2, 2, line.text[: width - 4], attr)
            except _curses.error:
                pass
        win.refresh()
    except Exception:  # pragma: no cover
        pass
