"""
plotter.py — Real-time joint angle plotter for the Braccio arm.

ArmPlotter streams all six servo angles live.  A sampler thread reads
ArmState at PLOT_SAMPLE_HZ and feeds a rolling deque buffer; the main
thread pumps the matplotlib GUI and FuncAnimation redraws all open
figures at PLOT_UPDATE_HZ.

Main window layout: 2 rows × 3 columns, one subplot per joint.
Each subplot shows a translucent red band marking the joint's safe
operating range (JOINT_LIMITS), with the y-axis padded by
PLOT_Y_MARGIN_DEG degrees above and below.

Keyboard shortcuts (focus any plot window):
  U       — reset: clear all data and restart t = 0
  Y       — screenshot: save main combined figure as PNG
  T       — log: toggle CSV logging (all 6 joints always recorded)
  0       — toggle main combined window (hide / show)
  1 – 6   — toggle individual pop-out window for joint 1 – 6

Usage — controller attaches before calling run():
    ctrl = BraccioController(port, baud)
    ctrl.attach_plotter(ArmPlotter(ctrl._state))
    ctrl.run()

Or use __main__.py which does this automatically unless --no-plot is set.
"""

import csv
import importlib
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime

from .arm_state import ArmState
from .constants import (
    JOINT_NAMES, JOINT_COLORS, JOINT_LIMITS,
    PLOT_WINDOW_S, PLOT_SAMPLE_HZ, PLOT_UPDATE_HZ,
    PLOT_KEY_RESET, PLOT_KEY_SCREENSHOT, PLOT_KEY_LOG,
    PLOT_KEY_MAIN_TOGGLE, PLOT_Y_MARGIN_DEG,
    LOG_DIR, SCREENSHOT_DIR,
)

# ── Visual constants for the limit band ───────────────────────────────────────
_LIMIT_COLOR = '#F38BA8'   # Catppuccin Mocha red — marks the safe operating band
_LIMIT_ALPHA = 0.18


class ArmPlotter:
    """
    Real-time matplotlib plotter for all six Braccio servo joints.

    Parameters
    ----------
    state      : shared ArmState instance (same object as the controller uses)
    window_s   : seconds of history shown in the scrolling window (default 60)
    sample_hz  : how often to read ArmState angles (default 20 Hz)
    update_hz  : FuncAnimation redraw rate (default 10 Hz)
    """

    def __init__(
        self,
        state: ArmState,
        window_s: float  = PLOT_WINDOW_S,
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
        self._logging:    bool = False
        self._log_file         = None
        self._log_writer       = None
        self._log_lock    = threading.Lock()

        # ── Lifecycle ─────────────────────────────────────────────────────
        self._stop_event  = threading.Event()
        self._plt         = None   # pyplot module, set after GUI init
        self._ui_ready    = False
        self._ui_thread   = None
        self._ui_thread_id = None
        self._ui_cmd_q    = queue.SimpleQueue()

        # ── Main combined figure (6 subplots, 2×3 grid) ───────────────────
        self._fig    = None   # Figure — also used by save_screenshot()
        self._axes   = []     # list[Axes], length 6, in joint order
        self._lines  = []     # list[Line2D], one per joint
        self._anim   = None   # FuncAnimation instance
        self._status_txt = None   # logging indicator text artist

        # ── Per-joint pop-out figures ─────────────────────────────────────
        self._ind_figs  = [None] * 6   # Figure per joint, or None if closed
        self._ind_axes  = [None] * 6
        self._ind_lines = [None] * 6

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch sampler thread + dedicated plot UI thread."""
        self._stop_event.clear()

        threading.Thread(
            target=self._sampler_loop,
            daemon=True,
            name='braccio-plotter-sampler',
        ).start()

        self._ui_thread = threading.Thread(
            target=self._ui_loop,
            daemon=True,
            name='braccio-plotter-ui',
        )
        self._ui_thread.start()

    def pump(self) -> None:
        """Legacy no-op: UI is pumped by the dedicated plotter thread."""
        return

    def toggle_main(self) -> None:
        """Toggle the combined 2x3 plot window from controller/curses keys."""
        if self._stop_event.is_set():
            return
        if threading.get_ident() == self._ui_thread_id:
            self._toggle_main_window()
        else:
            self._ui_cmd_q.put(('toggle_main', None))

    def toggle_joint(self, i: int) -> None:
        """Toggle a single-joint pop-out window for joint index i (0..5)."""
        if self._stop_event.is_set() or not (0 <= i < 6):
            return
        if threading.get_ident() == self._ui_thread_id:
            self._toggle_ind_window(i)
        else:
            self._ui_cmd_q.put(('toggle_joint', i))

    def stop(self) -> None:
        """Signal sampler to stop, close logs, and close open figures."""
        self._stop_event.set()
        self._close_log()
        self._ui_cmd_q.put(('shutdown', None))
        if self._plt is not None:
            try:
                if self._fig is not None:
                    self._plt.close(self._fig)
                for fig in self._ind_figs:
                    if fig is not None:
                        self._plt.close(fig)
            except Exception:
                pass
        if self._ui_thread is not None and self._ui_thread.is_alive():
            self._ui_thread.join(timeout=0.4)
        self._ui_ready = False
        self._ui_thread_id = None
        self._plt = None

    def reset(self) -> None:
        """Clear all buffered data and reset the time origin to now."""
        with self._data_lock:
            self._t0 = time.monotonic()
            self._times.clear()
            for d in self._angles:
                d.clear()

    def save_screenshot(self) -> str:
        """
        Save the main combined figure as a PNG.

        Creates the screenshots/ directory if absent.
        Returns the saved file path (also printed to stdout).
        """
        if threading.get_ident() != self._ui_thread_id:
            self._ui_cmd_q.put(('screenshot', None))
            return ""
        if self._fig is None:
            return ""
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(SCREENSHOT_DIR, f'screenshot_{ts}.png')
        self._fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def toggle_logging(self) -> bool:
        """
        Enable or disable CSV logging.

        On enable: opens a new timestamped file in logs/.
        On disable: flushes and closes the current file.
        CSV always contains all 6 joint columns regardless of visible windows.
        Returns the new logging state (True = now logging).
        """
        with self._log_lock:
            if self._logging:
                self._close_log()
                self._logging = False
            else:
                os.makedirs(LOG_DIR, exist_ok=True)
                ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(LOG_DIR, f'braccio_{ts}.csv')
                self._log_file   = open(path, 'w', newline='', buffering=1)
                self._log_writer = csv.writer(self._log_file)
                self._log_writer.writerow(
                    ['time_s'] + [n.strip() for n in JOINT_NAMES]
                )
                self._logging = True
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
        # Qt backends are safe to create from a non-main thread (unlike TkAgg,
        # whose C layer enforces Tcl thread affinity and crashes on Python 3.14).
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
        """Import plotting modules after backend selection."""
        backend      = self._select_matplotlib_backend()
        pyplot       = importlib.import_module('matplotlib.pyplot')
        animation    = importlib.import_module('matplotlib.animation')
        return backend, pyplot, animation.FuncAnimation

    # ── Sampler thread ────────────────────────────────────────────────────

    def _sampler_loop(self) -> None:
        """
        Runs at PLOT_SAMPLE_HZ.  Reads joint angles from ArmState and
        appends timestamped samples to the rolling deques.
        Writes a CSV row if logging is enabled.
        All 6 joints are always written regardless of which windows are open.
        """
        interval = 1.0 / self._sample_hz
        while not self._stop_event.is_set():
            t_sample = time.monotonic() - self._t0
            snap     = self._state.snapshot()
            joints   = snap['joints']

            with self._data_lock:
                self._times.append(t_sample)
                for i, angle in enumerate(joints):
                    self._angles[i].append(angle)

            # Copy writer ref under lock, but do I/O outside lock so GUI update
            # cannot block on slow filesystem writes.
            with self._log_lock:
                writer = self._log_writer if self._logging else None
            if writer is not None:
                try:
                    writer.writerow([f'{t_sample:.4f}'] + [str(a) for a in joints])
                except Exception:
                    with self._log_lock:
                        self._close_log()
                        self._logging = False

            time.sleep(interval)

    # ── GUI lifecycle (UI thread) ────────────────────────────────────────

    def _ui_loop(self) -> None:
        """Dedicated UI thread: owns matplotlib event processing."""
        self._ui_thread_id = threading.get_ident()
        # Python 3.14 / matplotlib 3.10: the GUI thread check compares against
        # threading.main_thread().  Redirect that to this thread so the check
        # passes when matplotlib creates the figure and calls plt.show().
        threading.main_thread = threading.current_thread
        self._init_ui(show_main=False)
        while not self._stop_event.is_set():
            self._drain_ui_commands()
            if not self._ui_ready or self._plt is None:
                time.sleep(0.02)
                continue
            try:
                if self._fig is not None and self._plt.fignum_exists(self._fig.number):
                    self._fig.canvas.flush_events()
                for fig in self._ind_figs:
                    if fig is not None and self._plt.fignum_exists(fig.number):
                        fig.canvas.flush_events()
            except Exception:
                pass
            time.sleep(0.01)

    def _drain_ui_commands(self) -> None:
        """Run queued UI commands on the UI thread."""
        while True:
            try:
                cmd, payload = self._ui_cmd_q.get_nowait()
            except Exception:
                return
            if not self._ui_ready and cmd in ('toggle_main', 'toggle_joint', 'screenshot'):
                self._ui_cmd_q.put((cmd, payload))
                return
            if cmd == 'toggle_main':
                self._toggle_main_window()
            elif cmd == 'toggle_joint':
                self._toggle_ind_window(int(payload))
            elif cmd == 'screenshot':
                self.save_screenshot()
            elif cmd == 'shutdown':
                return

    def _init_ui(self, show_main: bool = True) -> None:
        """
        Create matplotlib UI and animation (main thread only).

        show(block=False) + pause() keeps the window responsive while
        controller curses loop continues.
        """
        try:
            backend, plt, FuncAnimation = self._load_plot_modules()
        except Exception:
            return

        self._plt = plt
        del backend

        self._build_main_figure()

        interval_ms = int(1000.0 / self._update_hz)
        self._anim = FuncAnimation(
            self._fig,
            self._update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=False)
        # Force an initial draw so users don't stare at a blank white window.
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        if not show_main:
            self._set_main_window_visible(False)
        self._ui_ready = True

    # ── Figure construction ───────────────────────────────────────────────

    def _build_main_figure(self) -> None:
        """Build the main 2×3 combined figure.  Called once from _init_ui()."""
        plt = self._plt

        fig, axes_arr = plt.subplots(
            2, 3,
            figsize=(14, 7),
            sharex=True,
            constrained_layout=True,
        )
        self._fig   = fig
        self._axes  = axes_arr.flatten().tolist()
        self._lines = []

        fig.suptitle('Braccio — Joint Angles (Real-Time)', fontsize=11)

        for i, ax in enumerate(self._axes):
            lo, hi = JOINT_LIMITS[i]

            # Safe-operating-range band (catppuccin red, translucent)
            ax.axhspan(lo, hi, alpha=_LIMIT_ALPHA, color=_LIMIT_COLOR, zorder=0)

            # Per-joint y-limits with margin
            ax.set_ylim(lo - PLOT_Y_MARGIN_DEG, hi + PLOT_Y_MARGIN_DEG)
            ax.set_xlim(0, self._window_s)

            ax.set_title(JOINT_NAMES[i].strip(), fontsize=9)
            ax.set_ylabel('°', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25)

            # X-axis label only on the bottom row (joints 3–5)
            if i >= 3:
                ax.set_xlabel('Time (s)', fontsize=8)

            line, = ax.plot(
                [], [],
                color=JOINT_COLORS[i],
                lw=1.6,
                antialiased=True,
            )
            self._lines.append(line)

        # Logging indicator in the top-left of the first subplot
        self._status_txt = self._axes[0].text(
            0.02, 0.96, '',
            transform=self._axes[0].transAxes,
            fontsize=7, va='top', color='green',
        )

        # Key-binding hint at the very bottom of the figure
        hint = (
            f'[{PLOT_KEY_RESET.upper()}] reset  '
            f'[{PLOT_KEY_SCREENSHOT.upper()}] screenshot  '
            f'[{PLOT_KEY_LOG.upper()}] log  '
            f'[{PLOT_KEY_MAIN_TOGGLE}] toggle main  '
            f'[1-6] pop-out joint'
        )
        fig.text(0.5, 0.002, hint, ha='center', fontsize=7, color='#666666')

        fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _build_ind_figure(self, i: int):
        """
        Build a single-joint pop-out figure for joint i (0-indexed).
        Returns (fig, ax, line).
        """
        plt    = self._plt
        lo, hi = JOINT_LIMITS[i]
        name   = JOINT_NAMES[i].strip()
        color  = JOINT_COLORS[i]

        fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
        fig.suptitle(f'Braccio — {name}', fontsize=10)

        ax.axhspan(lo, hi, alpha=_LIMIT_ALPHA, color=_LIMIT_COLOR, zorder=0)
        ax.set_ylim(lo - PLOT_Y_MARGIN_DEG, hi + PLOT_Y_MARGIN_DEG)
        ax.set_xlim(0, self._window_s)
        ax.set_ylabel('°', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

        line, = ax.plot([], [], color=color, lw=1.8, antialiased=True)

        hint = (
            f'[{PLOT_KEY_RESET.upper()}] reset  '
            f'[{PLOT_KEY_SCREENSHOT.upper()}] screenshot  '
            f'[{PLOT_KEY_LOG.upper()}] log  '
            f'[{PLOT_KEY_MAIN_TOGGLE}] toggle main  '
            f'[1-6] pop-out joint'
        )
        fig.text(0.5, 0.002, hint, ha='center', fontsize=7, color='#666666')

        fig.canvas.mpl_connect('key_press_event', self._on_key)

        return fig, ax, line

    # ── FuncAnimation callback ────────────────────────────────────────────

    def _update(self, _frame) -> list:
        """
        Called by FuncAnimation every 1/PLOT_UPDATE_HZ seconds (inside the
        Qt/Tk event loop).  Updates the main figure and any open pop-outs.
        Pop-outs use canvas.draw_idle() — safe because we're already in the
        event loop.
        """
        with self._data_lock:
            t_data = list(self._times)
            a_data = [list(d) for d in self._angles]

        if not t_data:
            return self._lines

        t_now = t_data[-1]
        x_min = max(0.0, t_now - self._window_s)
        x_max = x_min + self._window_s

        # ── Main combined figure ──────────────────────────────────────────
        for ax, line, angles in zip(self._axes, self._lines, a_data):
            line.set_data(t_data, angles)
            ax.set_xlim(x_min, x_max)

        self._status_txt.set_text('● REC' if self._logging else '')

        # ── Per-joint pop-out figures ─────────────────────────────────────
        for i in range(6):
            fig = self._ind_figs[i]
            if fig is None:
                continue
            if self._plt.fignum_exists(fig.number):
                self._ind_lines[i].set_data(t_data, a_data[i])
                self._ind_axes[i].set_xlim(x_min, x_max)
                fig.canvas.draw_idle()
            else:
                # User closed window via GUI × button — clean up reference
                self._ind_figs[i]  = None
                self._ind_axes[i]  = None
                self._ind_lines[i] = None

        return self._lines

    # ── Window toggle helpers ─────────────────────────────────────────────

    def _toggle_ind_window(self, i: int) -> None:
        """Open or close the pop-out window for joint i (0-indexed)."""
        plt = self._plt
        if plt is None:
            return
        fig = self._ind_figs[i]
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)
            self._ind_figs[i]  = None
            self._ind_axes[i]  = None
            self._ind_lines[i] = None
        else:
            f, a, l = self._build_ind_figure(i)
            self._ind_figs[i]  = f
            self._ind_axes[i]  = a
            self._ind_lines[i] = l

    def _toggle_main_window(self) -> None:
        """
        Hide or show the main combined figure window.
        Tries Tk first, then Qt; silently no-ops if neither is available.
        """
        if self._fig is None:
            return
        try:
            win = self._fig.canvas.manager.window
            # Tk backend
            try:
                self._set_main_window_visible(not win.winfo_ismapped())
                return
            except AttributeError:
                pass
            # Qt backend
            try:
                self._set_main_window_visible(not win.isVisible())
            except AttributeError:
                pass
        except Exception:
            pass

    def _set_main_window_visible(self, visible: bool) -> None:
        """Set main window visibility for both Tk and Qt backends."""
        if self._fig is None:
            return
        try:
            win = self._fig.canvas.manager.window
            # Tk backend
            try:
                if visible:
                    win.deiconify()
                else:
                    win.withdraw()
                return
            except AttributeError:
                pass
            # Qt backend
            try:
                if visible:
                    win.show()
                else:
                    win.hide()
            except AttributeError:
                pass
        except Exception:
            pass

    # ── Keyboard handler ──────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        """Handle key_press_event from any open figure window."""
        key = (event.key or '').lower()

        if key == PLOT_KEY_RESET:
            self.reset()
        elif key == PLOT_KEY_SCREENSHOT:
            self.save_screenshot()
        elif key == PLOT_KEY_LOG:
            self.toggle_logging()
        elif key == PLOT_KEY_MAIN_TOGGLE:
            self._toggle_main_window()
        elif len(key) == 1 and '1' <= key <= '6':
            self._toggle_ind_window(int(key) - 1)
