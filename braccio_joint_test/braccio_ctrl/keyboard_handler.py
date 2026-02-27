"""
keyboard_handler.py — Non-blocking keyboard input via curses.

KeyboardHandler wraps curses getch() with halfdelay so the main control
loop blocks for at most ~200 ms per iteration (5 Hz loop rate driven by
keyboard polling).  It returns action strings from the KEY_BINDINGS table
in constants.py, or None if no recognised key was pressed.
"""

import curses
from .constants import KEY_BINDINGS


class KeyboardHandler:
    """
    Read one key per call and translate it to an action string.

    Parameters
    ----------
    stdscr : curses window object passed from curses.wrapper()
    tenths : halfdelay units (each = 0.1 s).  Default 2 → 200 ms max block.
    """

    def __init__(self, stdscr, tenths: int = 2):
        self._scr = stdscr
        # halfdelay(N): getch() blocks at most N * 0.1 seconds, then
        # returns curses.ERR — so the loop stays responsive without
        # burning 100 % CPU.
        curses.halfdelay(tenths)
        self._scr.keypad(True)   # enable special-key translation

    def get_action(self):
        """
        Return an action string (e.g. 'theta_inc') or None.

        Blocks at most `tenths * 0.1` seconds.
        """
        try:
            key = self._scr.getch()
        except curses.error:
            return None
        if key == curses.ERR:
            return None
        return KEY_BINDINGS.get(key, None)
