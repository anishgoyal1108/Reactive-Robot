"""
plotter.py — Real-time joint angle plotter for the Braccio arm.

ArmPlotter opens a matplotlib figure in a background thread and streams
all six servo angles live.  A separate sampler thread reads ArmState at
PLOT_SAMPLE_HZ and feeds a rolling deque buffer; FuncAnimation redraws
the figure at PLOT_UPDATE_HZ.

Keyboard shortcuts (focus the plot window):
  R — reset: clear all data and restart t = 0
  S — screenshot: save PNG to screenshots/screenshot_YYYYMMDD_HHMMSS.png
  L — logging: toggle CSV logging; each enable creates a new timestamped
               file in logs/braccio_YYYYMMDD_HHMMSS.csv

Usage — controller attaches before calling run():
    ctrl = BraccioController(port, baud)
    ctrl.attach_plotter(ArmPlotter(ctrl._state))
    ctrl.run()

Or use __main__.py which does this automatically unless --no-plot is set.
"""

import csv
import importlib
import os
import threading
import time
from collections import deque
from datetime import datetime

from .arm_state import ArmState
from .constants import (
    JOINT_NAMES, JOINT_COLORS, JOINT_LIMITS,
    PLOT_WINDOW_S, PLOT_SAMPLE_HZ, PLOT_UPDATE_HZ,
    PLOT_KEY_RESET, PLOT_KEY_SCREENSHOT, PLOT_KEY_LOG,
    LOG_DIR, SCREENSHOT_DIR,
)


class ArmPlotter:
    """
    Real-time matplotlib plotter for all six Braccio servo joints.

    Parameters
    ----------
    state      : shared ArmState instance (same object as the controller uses)
    window_s   : seconds of history shown in the scrolling window (default 60)
    sample_hz  : how often to read ArmState angles (default 20 Hz)
    update_hz  : matplotlib FuncAnimation redraw rate (default 10 Hz)
    """

    def __init__(
        self,
        state: ArmState,
        window_s: float = PLOT_WINDOW_S,
        sample_hz: float = PLOT_SAMPLE_HZ,
        update_hz: float = PLOT_UPDATE_HZ,
    ) -> None:
        self._state     = state
        self._window_s  = window_s
        self._sample_hz = sample_hz
        self._update_hz = update_hz

        # ── Rolling data buffers ──────────────────────────────────────────
        maxlen = int(window_s * sample_hz)   # e.g. 1200 pts @ 60 s × 20 Hz
        self._data_lock = threading.Lock()
        self._times:  deque = deque(maxlen=maxlen)
        self._angles: list  = [deque(maxlen=maxlen) for _ in range(6)]
        self._t0: float     = time.monotonic()

        # ── CSV logging state ─────────────────────────────────────────────
        self._logging:    bool             = False
        self._log_file                     = None
        self._log_writer                   = None
        self._log_lock    = threading.Lock()

        # ── Lifecycle ─────────────────────────────────────────────────────
        self._stop_event  = threading.Event()
        self._fig         = None   # created inside plot thread
        self._ax          = None
        self._lines       = []
        self._anim        = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the sampler thread and the plot thread (both daemons)."""
        self._stop_event.clear()

        sampler = threading.Thread(
            target=self._sampler_loop,
            daemon=True,
            name='braccio-plotter-sampler',
        )
        sampler.start()

        plotter = threading.Thread(
            target=self._plot_loop,
            daemon=True,
            name='braccio-plotter-ui',
        )
        plotter.start()

    def stop(self) -> None:
        """Signal threads to stop and close any open log file."""
        self._stop_event.set()
        self._close_log()

    def reset(self) -> None:
        """Clear all buffered data and reset the time origin to now."""
        with self._data_lock:
            self._t0 = time.monotonic()
            self._times.clear()
            for d in self._angles:
                d.clear()

    def save_screenshot(self) -> str:
        """
        Save the current figure as a PNG.

        Creates the screenshots/ directory if absent.
        Returns the saved file path (also printed to stdout).
        """
        if self._fig is None:
            return ""
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(SCREENSHOT_DIR, f'screenshot_{ts}.png')
        self._fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'\n[plotter] screenshot saved → {path}')
        return path

    def toggle_logging(self) -> bool:
        """
        Enable or disable CSV logging.

        On enable: opens a new timestamped file in logs/.
        On disable: flushes and closes the current file.
        Returns the new logging state (True = now logging).
        """
        with self._log_lock:
            if self._logging:
                self._close_log()
                self._logging = False
                print('\n[plotter] logging stopped')
            else:
                os.makedirs(LOG_DIR, exist_ok=True)
                ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(LOG_DIR, f'braccio_{ts}.csv')
                self._log_file   = open(path, 'w', newline='', buffering=1)
                self._log_writer = csv.writer(self._log_file)
                # Header uses the stripped joint names from constants
                self._log_writer.writerow(
                    ['time_s'] + [n.strip() for n in JOINT_NAMES]
                )
                self._logging = True
                print(f'\n[plotter] logging started → {path}')
            return self._logging

    # ── Internal helpers ──────────────────────────────────────────────────

    def _close_log(self) -> None:
        """Flush and close the CSV file (idempotent)."""
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file   = None
            self._log_writer = None

    def _select_matplotlib_backend(self):
        """
        Choose and activate a usable interactive backend.

        Probe backend importability directly because matplotlib.use('qtagg')
        may defer failure until figure/canvas creation.
        """
        matplotlib = importlib.import_module('matplotlib')
        candidates = (
            ('qtagg', 'matplotlib.backends.backend_qtagg'),
            ('tkagg', 'matplotlib.backends.backend_tkagg'),
        )
        errors = []

        for backend_name, backend_module in candidates:
            try:
                importlib.import_module(backend_module)
                matplotlib.use(backend_name, force=True)
                return backend_name
            except Exception as exc:
                errors.append(f'{backend_name}: {exc}')

        joined = '; '.join(errors) if errors else 'no backend candidates'
        raise RuntimeError(f'No usable matplotlib GUI backend ({joined})')

    def _load_plot_modules(self):
        """
        Import plotting modules after backend selection, inside plot thread.
        """
        backend = self._select_matplotlib_backend()
        pyplot = importlib.import_module('matplotlib.pyplot')
        animation = importlib.import_module('matplotlib.animation')
        return backend, pyplot, animation.FuncAnimation

    # ── Sampler thread ────────────────────────────────────────────────────

    def _sampler_loop(self) -> None:
        """
        Runs at PLOT_SAMPLE_HZ.  Reads joint angles from ArmState and
        appends timestamped samples to the rolling deques.
        Writes a CSV row if logging is enabled.
        """
        interval = 1.0 / self._sample_hz
        while not self._stop_event.is_set():
            t_sample = time.monotonic() - self._t0
            snap     = self._state.snapshot()
            joints   = snap['joints']          # list of 6 ints — from ArmState

            with self._data_lock:
                self._times.append(t_sample)
                for i, angle in enumerate(joints):
                    self._angles[i].append(angle)

            # CSV logging (under its own lock to avoid blocking the data lock)
            with self._log_lock:
                if self._logging and self._log_writer is not None:
                    self._log_writer.writerow(
                        [f'{t_sample:.4f}'] + [str(a) for a in joints]
                    )

            time.sleep(interval)

    # ── Plot thread ───────────────────────────────────────────────────────

    def _plot_loop(self) -> None:
        """
        Creates the matplotlib figure, registers FuncAnimation, and blocks
        on plt.show().  All matplotlib state lives in this thread.
        """
        try:
            backend, plt, FuncAnimation = self._load_plot_modules()
        except Exception as exc:
            print(f'\n[plotter] disabled: {exc}')
            return

        print(f'\n[plotter] backend: {backend}')

        self._fig, self._ax = plt.subplots(figsize=(11, 5))
        self._fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.12)

        # Title and axis labels
        self._fig.suptitle('Braccio — Joint Angles (Real-Time)', fontsize=11)
        self._ax.set_xlabel('Time (s)', fontsize=9)
        self._ax.set_ylabel('Angle (°)', fontsize=9)

        # Fixed y-axis covering the full servo range plus a small margin
        self._ax.set_ylim(-5, 185)
        self._ax.set_xlim(0, self._window_s)

        # Horizontal reference lines at each joint's min/max
        for lo, hi in JOINT_LIMITS:
            self._ax.axhline(lo, color='#cccccc', lw=0.5, ls='--', zorder=0)
            self._ax.axhline(hi, color='#cccccc', lw=0.5, ls='--', zorder=0)

        self._ax.grid(True, alpha=0.25)
        self._ax.tick_params(labelsize=8)

        # One line per joint — reuse JOINT_NAMES and JOINT_COLORS from constants
        self._lines = [
            self._ax.plot(
                [], [],
                color=JOINT_COLORS[i],
                label=JOINT_NAMES[i].strip(),
                lw=1.6,
                antialiased=True,
            )[0]
            for i in range(6)
        ]

        # Legend outside the plot area on the right
        self._ax.legend(
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            fontsize=8,
            framealpha=0.8,
        )

        # Status text (logging indicator) in the top-left corner
        self._status_txt = self._ax.text(
            0.01, 0.97, '',
            transform=self._ax.transAxes,
            fontsize=7, va='top', color='green',
        )

        # Key binding hint at bottom of figure
        hint = (
            f'[{PLOT_KEY_RESET.upper()}] reset   '
            f'[{PLOT_KEY_SCREENSHOT.upper()}] screenshot   '
            f'[{PLOT_KEY_LOG.upper()}] toggle log'
        )
        self._fig.text(
            0.5, 0.01, hint,
            ha='center', fontsize=7, color='#666666',
        )

        # Connect keyboard handler
        self._fig.canvas.mpl_connect('key_press_event', self._on_key)

        # FuncAnimation — blit=False because we update xlim each frame
        interval_ms = int(1000.0 / self._update_hz)
        self._anim = FuncAnimation(
            self._fig,
            self._update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

        plt.show(block=True)

    # ── FuncAnimation callback ────────────────────────────────────────────

    def _update(self, _frame) -> list:
        """
        Called by FuncAnimation every 1/PLOT_UPDATE_HZ seconds.
        Copies the rolling deques and updates all six line artists.
        Scrolls the x-axis so the latest data is always at the right edge.
        """
        with self._data_lock:
            t_data = list(self._times)
            a_data = [list(d) for d in self._angles]

        if not t_data:
            return self._lines

        t_now = t_data[-1]

        # Scrolling window: fixed width, right edge = t_now
        x_min = max(0.0, t_now - self._window_s)
        x_max = x_min + self._window_s
        self._ax.set_xlim(x_min, x_max)

        # Update each line
        for line, angles in zip(self._lines, a_data):
            line.set_data(t_data, angles)

        # Update logging indicator
        with self._log_lock:
            logging = self._logging
        self._status_txt.set_text('● REC' if logging else '')

        return self._lines

    # ── Keyboard handler ──────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        """Handle key_press_event from the matplotlib figure."""
        key = (event.key or '').lower()

        if key == PLOT_KEY_RESET:
            self.reset()

        elif key == PLOT_KEY_SCREENSHOT:
            # savefig is safe to call from within the plot thread
            self.save_screenshot()

        elif key == PLOT_KEY_LOG:
            self.toggle_logging()
