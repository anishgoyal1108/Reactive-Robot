"""Non-blocking keyboard input for the high-rate control loop."""

from __future__ import annotations

import curses

from .constants import KEY_BINDINGS


class KeyboardHandler:
    """Translate `getch()` key codes into action strings."""

    def __init__(self, stdscr):
        self._scr = stdscr
        self._configure_window()

    def get_action(self):
        """Return one mapped action or `None` if no key is pending."""
        try:
            key = self._scr.getch()
        except curses.error:
            return None
        if key == curses.ERR:
            return None
        return KEY_BINDINGS.get(key, None)

    def reset_after_overlay(self) -> None:
        """Restore non-blocking settings after menus/editors."""
        self._configure_window()

    def _configure_window(self) -> None:
        self._scr.keypad(True)
        self._scr.nodelay(True)
        self._scr.timeout(0)
