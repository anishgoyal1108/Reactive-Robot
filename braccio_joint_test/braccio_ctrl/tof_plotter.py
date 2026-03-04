"""
tof_plotter.py — Real-time 4-channel ToF heatmap + 3D surface viewer.

Displays up to 4 VL53L5CX sensor channels as heatmaps and 3D surfaces.
Supports CSV data export, PNG screenshots, and per-channel pop-out windows.

Layout: 2×4 grid (top row = heatmaps, bottom row = 3D surfaces) for all
4 channels, or a single-channel pop-out with heatmap + surface side by side.

Keyboard shortcuts (focus any plot window):
  U       — reset: clear frame counters
  Y       — screenshot: save current figure as PNG
  T       — toggle CSV logging (all 4 channels, min/avg/max per frame)
  0       — toggle main 4-channel combined window
  1–4     — toggle individual channel pop-out window
  +/-     — adjust ToF threshold distance ±50 mm

Usage:
  tof_state = ToFState(num_channels=4)
  plotter   = ToFPlotter(tof_state)
  plotter.start()
  # ... in loop: plotter.pump() ...
  plotter.stop()
"""

import csv
import importlib
import os
import queue
import threading
import time
from datetime import datetime

import numpy as np

from .tof_sensor import ToFState
from .constants import (
    LOG_DIR, SCREENSHOT_DIR,
    TOF_UPSAMPLE_N, TOF_SURFACE_EVERY, TOF_PLOT_INTERVAL_MS,
    TOF_MAX_RANGE_MM,
)

# ── Visual constants ─────────────────────────────────────────────────────────
_CH_COLORS  = ['#89B4FA', '#A6E3A1', '#F9E2AF', '#F38BA8']  # Catppuccin blue/green/yellow/red
_CH_CMAPS   = ['viridis', 'plasma', 'inferno', 'cividis']


def _upsample_bilinear(z8: np.ndarray, N: int) -> np.ndarray:
    """Simple 2-pass bilinear upsampling from 8×8 to N×N."""
    h, w = z8.shape
    x_src = np.linspace(0, w - 1, w)
    y_src = np.linspace(0, h - 1, h)
    x_tgt = np.linspace(0, w - 1, N)
    y_tgt = np.linspace(0, h - 1, N)
    zx = np.vstack([np.interp(x_tgt, x_src, z8[r, :]) for r in range(h)])
    zN = np.vstack([np.interp(y_tgt, y_src, zx[:, c]) for c in range(N)]).T
    return zN


class ToFPlotter:
    """
    Real-time matplotlib plotter for 4 ToF channels + IR status.

    Parameters
    ----------
    state       : shared ToFState instance
    upsample_n  : bilinear upsample resolution (default 40)
    surface_every: redraw 3D surface every N frames (default 5)
    interval_ms : animation interval in ms (default 100)
    """

    def __init__(
        self,
        state: ToFState,
        upsample_n: int   = TOF_UPSAMPLE_N,
        surface_every: int = TOF_SURFACE_EVERY,
        interval_ms: int   = TOF_PLOT_INTERVAL_MS,
    ):
        self._state       = state
        self._upsample_n  = upsample_n
        self._surface_every = surface_every
        self._interval_ms = interval_ms
        self._num_ch      = state.num_channels

        # Meshgrid for 3D surfaces
        self._XN, self._YN = np.meshgrid(
            np.arange(upsample_n), np.arange(upsample_n)
        )

        # ── Lifecycle ─────────────────────────────────────────────────────
        self._stop_event   = threading.Event()
        self._plt          = None
        self._ui_ready     = False
        self._ui_thread    = None
        self._ui_thread_id = None
        self._ui_cmd_q     = queue.SimpleQueue()

        # ── Main combined figure (2×4: heatmaps + surfaces) ──────────────
        self._fig       = None
        self._ax_heat   = []    # list[Axes], length num_ch
        self._ax_surf   = []    # list[Axes], length num_ch
        self._imshows   = []    # list[AxesImage]
        self._surfaces  = []    # list[Poly3DCollection]
        self._anim      = None
        self._status_txt = None  # IR + obstacle status text

        # ── Per-channel pop-out figures ───────────────────────────────────
        self._ind_figs  = [None] * self._num_ch
        self._ind_ax_h  = [None] * self._num_ch
        self._ind_ax_s  = [None] * self._num_ch
        self._ind_ims   = [None] * self._num_ch
        self._ind_surfs = [None] * self._num_ch

        # ── CSV logging ──────────────────────────────────────────────────
        self._logging     = False
        self._log_file    = None
        self._log_writer  = None
        self._log_lock    = threading.Lock()

        # ── Frame counter for surface throttling ─────────────────────────
        self._draw_count = [0] * self._num_ch

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the UI thread."""
        self._stop_event.clear()
        self._ui_thread = threading.Thread(
            target=self._ui_loop,
            daemon=True,
            name='tof-plotter-ui',
        )
        self._ui_thread.start()

    def pump(self) -> None:
        """No-op — UI is pumped by its own thread."""
        return

    def stop(self) -> None:
        """Signal stop, close logs, close figures."""
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
            self._ui_thread.join(timeout=0.5)
        self._ui_ready = False
        self._plt = None

    def toggle_main(self) -> None:
        """Toggle the main combined window visibility."""
        if self._stop_event.is_set():
            return
        if threading.get_ident() == self._ui_thread_id:
            self._toggle_main_window()
        else:
            self._ui_cmd_q.put(('toggle_main', None))

    def toggle_channel(self, ch: int) -> None:
        """Toggle a single-channel pop-out window."""
        if self._stop_event.is_set() or not (0 <= ch < self._num_ch):
            return
        if threading.get_ident() == self._ui_thread_id:
            self._toggle_ind_window(ch)
        else:
            self._ui_cmd_q.put(('toggle_ch', ch))

    def save_screenshot(self) -> str:
        """Save the main figure as PNG."""
        if threading.get_ident() != self._ui_thread_id:
            self._ui_cmd_q.put(('screenshot', None))
            return ''
        if self._fig is None:
            return ''
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(SCREENSHOT_DIR, f'tof_screenshot_{ts}.png')
        self._fig.savefig(path, dpi=150, bbox_inches='tight')
        return path

    def toggle_logging(self) -> bool:
        """Toggle CSV logging for all channels. Returns new state."""
        with self._log_lock:
            if self._logging:
                self._close_log()
                self._logging = False
            else:
                os.makedirs(LOG_DIR, exist_ok=True)
                ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(LOG_DIR, f'tof_{ts}.csv')
                self._log_file   = open(path, 'w', newline='', buffering=1)
                self._log_writer = csv.writer(self._log_file)
                header = ['timestamp', 'channel', 'min_mm', 'avg_mm', 'max_mm',
                           'ir_bits', 'ir_label', 'obstacle_response']
                self._log_writer.writerow(header)
                self._logging = True
            return self._logging

    def export_csv_snapshot(self) -> str:
        """Export current grids of all channels to a single CSV file."""
        os.makedirs(LOG_DIR, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(LOG_DIR, f'tof_grids_{ts}.csv')

        snap = self._state.snapshot()
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['channel', 'row', 'col', 'distance_mm'])
            for ch in range(snap['num_channels']):
                grid = snap['grids'][ch]
                for r in range(8):
                    for c in range(8):
                        w.writerow([ch, r, c, f'{grid[r, c]:.1f}'])
        return path

    # ── Internal helpers ──────────────────────────────────────────────────

    def _close_log(self):
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file   = None
            self._log_writer = None

    def _select_backend(self):
        matplotlib = importlib.import_module('matplotlib')
        for name, mod in [('tkagg', 'matplotlib.backends.backend_tkagg'),
                          ('qtagg', 'matplotlib.backends.backend_qtagg')]:
            try:
                importlib.import_module(mod)
                matplotlib.use(name, force=True)
                return name
            except Exception:
                pass
        raise RuntimeError('No usable matplotlib GUI backend')

    # ── UI thread ─────────────────────────────────────────────────────────

    def _ui_loop(self):
        self._ui_thread_id = threading.get_ident()
        self._init_ui()
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

    def _drain_ui_commands(self):
        while True:
            try:
                cmd, payload = self._ui_cmd_q.get_nowait()
            except Exception:
                return
            if cmd == 'toggle_main':
                self._toggle_main_window()
            elif cmd == 'toggle_ch':
                self._toggle_ind_window(int(payload))
            elif cmd == 'screenshot':
                self.save_screenshot()
            elif cmd == 'export_csv':
                self.export_csv_snapshot()
            elif cmd == 'shutdown':
                return

    def _init_ui(self):
        try:
            self._select_backend()
            plt = importlib.import_module('matplotlib.pyplot')
            FuncAnimation = importlib.import_module('matplotlib.animation').FuncAnimation
        except Exception:
            return

        self._plt = plt
        self._build_main_figure()

        self._anim = FuncAnimation(
            self._fig,
            self._update,
            interval=self._interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=False)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        # Start hidden — user toggles with the key binding
        self._set_main_visible(False)
        self._ui_ready = True

    # ── Figure construction ───────────────────────────────────────────────

    def _build_main_figure(self):
        plt = self._plt
        N   = self._upsample_n

        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('ToF Sensors — Live View (4 Channels)', fontsize=11)

        self._ax_heat   = []
        self._ax_surf   = []
        self._imshows   = []
        self._surfaces  = []

        init_z = np.zeros((N, N), dtype=np.float32)

        for ch in range(self._num_ch):
            # Top row: heatmaps
            ax_h = fig.add_subplot(2, self._num_ch, ch + 1)
            ax_h.set_title(f'CH{ch} Heatmap', fontsize=9)
            im = ax_h.imshow(init_z, origin='lower', aspect='equal',
                             vmin=0, vmax=TOF_MAX_RANGE_MM,
                             cmap=_CH_CMAPS[ch % len(_CH_CMAPS)])
            plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            self._ax_heat.append(ax_h)
            self._imshows.append(im)

            # Bottom row: 3D surfaces
            ax_s = fig.add_subplot(2, self._num_ch, self._num_ch + ch + 1,
                                   projection='3d')
            ax_s.set_title(f'CH{ch} Surface', fontsize=9)
            ax_s.set_xlabel('X', fontsize=7)
            ax_s.set_ylabel('Y', fontsize=7)
            ax_s.set_zlabel('mm', fontsize=7)
            surf = ax_s.plot_surface(self._XN, self._YN, init_z,
                                     rstride=2, cstride=2,
                                     linewidth=0, antialiased=True)
            self._ax_surf.append(ax_s)
            self._surfaces.append(surf)

        # Status text at bottom
        self._status_txt = fig.text(
            0.5, 0.01, '', ha='center', fontsize=8, color='red',
            fontweight='bold',
        )

        # Key hint
        hint = ('[U] reset  [Y] screenshot  [T] log  [C] export CSV  '
                '[0] toggle main  [1-4] pop-out  [+/-] threshold')
        fig.text(0.5, 0.005, hint, ha='center', fontsize=6, color='#888888')

        fig.canvas.mpl_connect('key_press_event', self._on_key)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._fig = fig

    def _build_ind_figure(self, ch: int):
        plt = self._plt
        N   = self._upsample_n
        init_z = np.zeros((N, N), dtype=np.float32)

        fig = plt.figure(figsize=(10, 4))
        fig.suptitle(f'ToF CH{ch} — Heatmap + Surface', fontsize=10)

        ax_h = fig.add_subplot(1, 2, 1)
        im   = ax_h.imshow(init_z, origin='lower', aspect='equal',
                            vmin=0, vmax=TOF_MAX_RANGE_MM,
                            cmap=_CH_CMAPS[ch % len(_CH_CMAPS)])
        plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

        ax_s = fig.add_subplot(1, 2, 2, projection='3d')
        ax_s.set_xlabel('X')
        ax_s.set_ylabel('Y')
        ax_s.set_zlabel('mm')
        surf = ax_s.plot_surface(self._XN, self._YN, init_z,
                                  rstride=2, cstride=2,
                                  linewidth=0, antialiased=True)

        fig.canvas.mpl_connect('key_press_event', self._on_key)
        fig.tight_layout()
        return fig, ax_h, ax_s, im, surf

    # ── Animation callback ────────────────────────────────────────────────

    def _update(self, _frame):
        snap = self._state.snapshot()
        N    = self._upsample_n

        for ch in range(self._num_ch):
            grid = snap['grids'][ch]

            if np.isnan(grid).all():
                zN = np.zeros((N, N), dtype=np.float32)
                suffix = 'NO DATA'
            else:
                zN = _upsample_bilinear(grid, N)
                suffix = 'LIVE'

            # Main figure heatmap
            self._imshows[ch].set_data(zN)

            age = (time.time() - snap['last_rx'][ch]) if snap['last_rx'][ch] > 0 else 999
            self._ax_heat[ch].set_title(
                f'CH{ch} {suffix} | f={snap["frame_cnt"][ch]} | {age:.1f}s',
                fontsize=8,
            )

            # Main figure surface (throttled)
            self._draw_count[ch] += 1
            if self._draw_count[ch] % self._surface_every == 0:
                try:
                    self._surfaces[ch].remove()
                except Exception:
                    pass
                self._surfaces[ch] = self._ax_surf[ch].plot_surface(
                    self._XN, self._YN, zN,
                    rstride=2, cstride=2, linewidth=0, antialiased=True,
                )

            # Pop-out update
            fig = self._ind_figs[ch]
            if fig is not None and self._plt.fignum_exists(fig.number):
                self._ind_ims[ch].set_data(zN)
                if self._draw_count[ch] % self._surface_every == 0:
                    try:
                        self._ind_surfs[ch].remove()
                    except Exception:
                        pass
                    self._ind_surfs[ch] = self._ind_ax_s[ch].plot_surface(
                        self._XN, self._YN, zN,
                        rstride=2, cstride=2, linewidth=0, antialiased=True,
                    )
                fig.canvas.draw_idle()
            elif fig is not None:
                # Window closed by user
                self._ind_figs[ch]  = None
                self._ind_ax_h[ch]  = None
                self._ind_ax_s[ch]  = None
                self._ind_ims[ch]   = None
                self._ind_surfs[ch] = None

        # Status text: obstacle + IR
        obs   = snap['obstacle_response']
        ir    = snap['ir_label']
        dist  = snap['obstacle_dist_mm']
        thresh = snap['tof_threshold_mm']

        if obs == 'back_away':
            status = f'*** IR: {ir} — BACK AWAY (ToF missed!) ***'
            self._status_txt.set_color('red')
        elif obs == 'replan':
            status = f'ToF: {dist:.0f} mm < {thresh:.0f} mm threshold — REPLAN TRAJECTORY'
            self._status_txt.set_color('orange')
        else:
            status = f'Clear | IR: {ir} | Threshold: {thresh:.0f} mm'
            self._status_txt.set_color('green')

        self._status_txt.set_text(status)

        # CSV logging
        with self._log_lock:
            writer = self._log_writer if self._logging else None
        if writer is not None:
            try:
                ts = datetime.now().isoformat()
                for ch in range(self._num_ch):
                    g = snap['grids'][ch]
                    g_valid = g[~np.isnan(g)]
                    if g_valid.size > 0:
                        writer.writerow([
                            ts, ch,
                            f'{np.min(g_valid):.1f}',
                            f'{np.mean(g_valid):.1f}',
                            f'{np.max(g_valid):.1f}',
                            snap['ir_bits'],
                            snap['ir_label'],
                            snap['obstacle_response'],
                        ])
            except Exception:
                with self._log_lock:
                    self._close_log()
                    self._logging = False

        return list(self._imshows)

    # ── Window toggles ────────────────────────────────────────────────────

    def _toggle_ind_window(self, ch: int):
        plt = self._plt
        if plt is None:
            return
        fig = self._ind_figs[ch]
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)
            self._ind_figs[ch]  = None
            self._ind_ax_h[ch]  = None
            self._ind_ax_s[ch]  = None
            self._ind_ims[ch]   = None
            self._ind_surfs[ch] = None
        else:
            f, ah, a_s, im, surf = self._build_ind_figure(ch)
            self._ind_figs[ch]  = f
            self._ind_ax_h[ch]  = ah
            self._ind_ax_s[ch]  = a_s
            self._ind_ims[ch]   = im
            self._ind_surfs[ch] = surf

    def _toggle_main_window(self):
        if self._fig is None:
            return
        try:
            win = self._fig.canvas.manager.window
            try:
                self._set_main_visible(not win.winfo_ismapped())
                return
            except AttributeError:
                pass
            try:
                self._set_main_visible(not win.isVisible())
            except AttributeError:
                pass
        except Exception:
            pass

    def _set_main_visible(self, visible: bool):
        if self._fig is None:
            return
        try:
            win = self._fig.canvas.manager.window
            try:
                if visible:
                    win.deiconify()
                else:
                    win.withdraw()
                return
            except AttributeError:
                pass
            try:
                if visible:
                    win.show()
                else:
                    win.hide()
            except AttributeError:
                pass
        except Exception:
            pass

    # ── Keyboard ──────────────────────────────────────────────────────────

    def _on_key(self, event):
        key = (event.key or '').lower()

        if key == 'u':
            # Reset frame counters
            with self._state._lock:
                for ch in range(self._num_ch):
                    self._state.frame_cnt[ch] = 0
                    self._draw_count[ch] = 0
        elif key == 'y':
            self.save_screenshot()
        elif key == 't':
            self.toggle_logging()
        elif key == 'c':
            self.export_csv_snapshot()
        elif key == '0':
            self._toggle_main_window()
        elif key in ('1', '2', '3', '4'):
            self._toggle_ind_window(int(key) - 1)
        elif key in ('+', '='):
            with self._state._lock:
                self._state.tof_threshold_mm = min(
                    3000.0, self._state.tof_threshold_mm + 50.0)
        elif key in ('-', '_'):
            with self._state._lock:
                self._state.tof_threshold_mm = max(
                    50.0, self._state.tof_threshold_mm - 50.0)
