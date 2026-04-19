#!/usr/bin/env python3
"""
tof_plotter_app.py — Standalone real-time ToF heatmap + 3-D surface viewer.

Receives ToF/IR data over UDP from braccio_ctrl and displays live
matplotlib heatmaps and 3-D surface plots.  Runs entirely on the main
thread — no threading issues with Tk/Qt backends.

Usage:
    python tof_plotter_app.py [--port PORT]

Keyboard shortcuts (focus the plot window):
    U       — reset frame counters
    Y       — save screenshot (PNG)
    T       — toggle CSV logging
    C       — export current grids to CSV snapshot
    0       — toggle main combined window
    1–4     — toggle per-channel pop-out window
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from collections.abc import Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, TextIO

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from braccio_ctrl.mpl_compat import figure_number
from braccio_ctrl.constants import (
    TOF_UPSAMPLE_N, TOF_SURFACE_EVERY, TOF_PLOT_INTERVAL_MS,
    TOF_MAX_RANGE_MM, TOF_DATA_PORT, LOG_DIR, SCREENSHOT_DIR,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from matplotlib.text import Text
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d.axes3d import Axes3D


class _CsvRowWriter(Protocol):
    def writerow(self, row: Iterable[object]) -> object:
        ...


_CH_CMAPS  = ['viridis', 'plasma', 'inferno', 'cividis']
_NUM_CH    = 4


def _upsample_bilinear(z: np.ndarray, N: int) -> np.ndarray:
    h, w = z.shape
    x_src = np.linspace(0, w - 1, w)
    y_src = np.linspace(0, h - 1, h)
    x_tgt = np.linspace(0, w - 1, N)
    y_tgt = np.linspace(0, h - 1, N)
    zx = np.vstack([np.interp(x_tgt, x_src, z[r, :]) for r in range(h)])
    zN = np.vstack([np.interp(y_tgt, y_src, zx[:, c]) for c in range(N)]).T
    return zN


class ToFPlotterApp:
    def __init__(
        self,
        port: int = TOF_DATA_PORT,
        upsample_n: int = TOF_UPSAMPLE_N,
    ) -> None:
        self._port       = port
        self._upsample_n = upsample_n
        self._num_ch     = _NUM_CH

        # Latest received state (updated each animation tick)
        self._grids      = [np.full((4, 4), np.nan, dtype=np.float32)
                            for _ in range(_NUM_CH)]
        self._frame_cnt  = [0] * _NUM_CH
        self._last_rx    = [0.0] * _NUM_CH
        self._ir_label   = 'CLEAR'
        self._obstacle_response = 'clear'
        self._obstacle_dist_mm  = -1.0
        self._tof_threshold_mm  = 300.0

        # Meshgrid for 3-D surfaces
        N = upsample_n
        self._XN, self._YN = np.meshgrid(np.arange(N), np.arange(N))

        # UDP socket (non-blocking)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('127.0.0.1', port))
        self._sock.setblocking(False)

        # Main figure widgets
        self._fig: Figure | None = None
        self._ax_heat: list[Axes] = []
        self._ax_surf: list[Axes3D] = []
        self._imshows: list[AxesImage] = []
        self._surfaces: list[Poly3DCollection] = []
        self._status_txt: Text | None = None
        self._link_txt: Text | None = None

        # Per-channel pop-outs
        self._ind_figs: list[Figure | None] = [None] * _NUM_CH
        self._ind_ax_h: list[Axes | None] = [None] * _NUM_CH
        self._ind_ax_s: list[Axes3D | None] = [None] * _NUM_CH
        self._ind_ims: list[AxesImage | None] = [None] * _NUM_CH
        self._ind_surfs: list[Poly3DCollection | None] = [None] * _NUM_CH

        # CSV logging
        self._logging: bool = False
        self._log_file: TextIO | None = None
        self._log_writer: _CsvRowWriter | None = None

        # Drawing counters
        self._draw_count = [0] * _NUM_CH

        # UDP / link diagnostics (for “no data” vs “masked” vs “wrong process”)
        self._udp_pkts_window = 0
        self._udp_rate_t0    = time.monotonic()
        self._udp_rate_hz    = 0.0
        self._teensy_ok      = False
        self._teensy_port    = ''
        self._diag_vc        = [0] * _NUM_CH
        self._diag_zc        = [0] * _NUM_CH

    def run(self) -> None:
        self._build_main_figure()
        fig = self._fig
        assert fig is not None
        self._anim = FuncAnimation(
            fig, self._update,
            interval=TOF_PLOT_INTERVAL_MS,
            blit=False, cache_frame_data=False,
        )
        plt.show()

    # ── Figure construction ───────────────────────────────────────────────

    def _build_main_figure(self) -> None:
        N = self._upsample_n
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('ToF Sensors — Live View (4 Channels)', fontsize=11)

        self._ax_heat  = []
        self._ax_surf  = []
        self._imshows  = []
        self._surfaces = []
        init_z = np.zeros((N, N), dtype=np.float32)

        for ch in range(self._num_ch):
            ax_h = fig.add_subplot(2, self._num_ch, ch + 1)
            ax_h.set_title(f'CH{ch} Heatmap', fontsize=9)
            im = ax_h.imshow(init_z, origin='lower', aspect='equal',
                             vmin=0, vmax=TOF_MAX_RANGE_MM,
                             cmap=_CH_CMAPS[ch % len(_CH_CMAPS)])
            plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            self._ax_heat.append(ax_h)
            self._imshows.append(im)

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

        self._link_txt = fig.text(
            0.5, 0.97, '', ha='center', fontsize=8, color='#444444',
        )
        self._status_txt = fig.text(
            0.5, 0.01, '', ha='center', fontsize=8, color='red',
            fontweight='bold',
        )
        hint = ('[U] reset  [Y] screenshot  [T] log  [C] export CSV  '
                '[0] toggle  [1-4] pop-out')
        fig.text(0.5, 0.005, hint, ha='center', fontsize=6, color='#888888')
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.92))
        self._fig = fig

    def _build_ind_figure(self, ch: int):
        N = self._upsample_n
        init_z = np.zeros((N, N), dtype=np.float32)

        fig = plt.figure(figsize=(10, 4))
        fig.suptitle(f'ToF CH{ch} — Heatmap + Surface', fontsize=10)

        ax_h = fig.add_subplot(1, 2, 1)
        im   = ax_h.imshow(init_z, origin='lower', aspect='equal',
                            vmin=0, vmax=TOF_MAX_RANGE_MM,
                            cmap=_CH_CMAPS[ch % len(_CH_CMAPS)])
        plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

        ax_s = fig.add_subplot(1, 2, 2, projection='3d')
        ax_s.set_xlabel('X'); ax_s.set_ylabel('Y'); ax_s.set_zlabel('mm')
        surf = ax_s.plot_surface(self._XN, self._YN, init_z,
                                  rstride=2, cstride=2,
                                  linewidth=0, antialiased=True)
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        fig.tight_layout()
        return fig, ax_h, ax_s, im, surf

    # ── Animation callback ────────────────────────────────────────────────

    def _update(self, _frame):
        N = self._upsample_n
        now = time.time()

        # Drain UDP — keep only the latest packet
        latest = None
        while True:
            try:
                data, _ = self._sock.recvfrom(65536)
                latest = json.loads(data)
                self._udp_pkts_window += 1
            except BlockingIOError:
                break
            except Exception:
                break

        dt = now - self._udp_rate_t0
        if dt >= 1.0:
            self._udp_rate_hz = self._udp_pkts_window / dt
            self._udp_pkts_window = 0
            self._udp_rate_t0 = now

        if latest is not None:
            for ch in range(self._num_ch):
                raw = latest['grids'][ch]
                self._grids[ch] = np.array(raw, dtype=np.float32)
                self._last_rx[ch] = now
            self._frame_cnt = latest.get('frame_cnt', self._frame_cnt)
            self._ir_label  = latest.get('ir_label', self._ir_label)
            self._obstacle_response = latest.get('obstacle_response',
                                                  self._obstacle_response)
            self._obstacle_dist_mm  = latest.get('obstacle_dist_mm',
                                                  self._obstacle_dist_mm)
            self._tof_threshold_mm  = latest.get('tof_threshold_mm',
                                                  self._tof_threshold_mm)
            self._teensy_ok = latest.get('teensy_connected', False)
            self._teensy_port = latest.get('teensy_port', '')
            self._diag_vc = latest.get('diag_valid_cells', self._diag_vc)
            self._diag_zc = latest.get('diag_zone_count', self._diag_zc)

        if self._udp_rate_hz < 0.05:
            link = ('UDP: no packets — run braccio_ctrl with '
                    f'--teensy-port (this app listens on :{self._port})')
        else:
            ts = ('Teensy serial OK' if self._teensy_ok
                  else 'Teensy serial NOT OPEN in runner')
            port = self._teensy_port or '(port ?)'
            link = f'UDP ~{self._udp_rate_hz:.1f} pkt/s | {ts} {port}'
        lt = self._link_txt
        if lt is not None:
            lt.set_text(link)

        for ch in range(self._num_ch):
            grid = self._grids[ch]
            fc   = self._frame_cnt[ch]
            vc = self._diag_vc[ch] if ch < len(self._diag_vc) else 0
            zc = self._diag_zc[ch] if ch < len(self._diag_zc) else 0
            if np.isnan(grid).all():
                zN = np.zeros((N, N), dtype=np.float32)
                if fc > 0 and zc > 0 and vc == 0:
                    suffix = 'MASKED'
                elif fc > 0:
                    suffix = 'NO CELLS'
                else:
                    suffix = 'NO DATA'
            else:
                zN     = _upsample_bilinear(grid, N)
                suffix = 'LIVE'

            self._imshows[ch].set_data(zN)
            age = now - self._last_rx[ch] if self._last_rx[ch] > 0 else 999
            self._ax_heat[ch].set_title(
                f'CH{ch} {suffix} | f={self._frame_cnt[ch]} | {age:.1f}s',
                fontsize=8,
            )

            self._draw_count[ch] += 1
            if self._draw_count[ch] % TOF_SURFACE_EVERY == 0:
                try:
                    self._surfaces[ch].remove()
                except Exception:
                    pass
                self._surfaces[ch] = self._ax_surf[ch].plot_surface(
                    self._XN, self._YN, zN,
                    rstride=2, cstride=2, linewidth=0, antialiased=True,
                )

            # Pop-out
            fig = self._ind_figs[ch]
            if fig is not None and plt.fignum_exists(figure_number(fig)):
                im_ind = self._ind_ims[ch]
                if im_ind is not None:
                    im_ind.set_data(zN)
                if self._draw_count[ch] % TOF_SURFACE_EVERY == 0:
                    surf_ind = self._ind_surfs[ch]
                    if surf_ind is not None:
                        try:
                            surf_ind.remove()
                        except Exception:
                            pass
                    ax_s = self._ind_ax_s[ch]
                    if ax_s is not None:
                        self._ind_surfs[ch] = ax_s.plot_surface(
                            self._XN, self._YN, zN,
                            rstride=2, cstride=2, linewidth=0,
                            antialiased=True,
                        )
                fig.canvas.draw_idle()
            elif fig is not None:
                self._ind_figs[ch]  = None
                self._ind_ax_h[ch]  = None
                self._ind_ax_s[ch]  = None
                self._ind_ims[ch]   = None
                self._ind_surfs[ch] = None

        # Status bar
        obs    = self._obstacle_response
        ir     = self._ir_label
        dist   = self._obstacle_dist_mm
        thresh = self._tof_threshold_mm
        st = self._status_txt
        if st is not None:
            if obs == 'back_away':
                status = f'*** IR: {ir} — BACK AWAY (ToF missed!) ***'
                st.set_color('red')
            elif obs == 'replan':
                status = (
                    f'ToF: {dist:.0f} mm < {thresh:.0f} mm threshold '
                    f'— REPLAN TRAJECTORY'
                )
                st.set_color('orange')
            else:
                status = f'Clear | IR: {ir} | Threshold: {thresh:.0f} mm'
                st.set_color('green')
            st.set_text(status)

        # CSV streaming log
        writer = self._log_writer
        if self._logging and writer is not None:
            try:
                ts = datetime.now().isoformat()
                for ch in range(self._num_ch):
                    g = self._grids[ch]
                    gv = g[~np.isnan(g)]
                    if gv.size > 0:
                        writer.writerow([
                            ts, ch,
                            f'{np.min(gv):.1f}',
                            f'{np.mean(gv):.1f}',
                            f'{np.max(gv):.1f}',
                            self._ir_label,
                            self._obstacle_response,
                        ])
            except Exception:
                self._close_log()

        return list(self._imshows)

    # ── Keyboard ──────────────────────────────────────────────────────────

    def _on_key(self, event):
        key = (event.key or '').lower()
        if key == 'u':
            self._draw_count = [0] * self._num_ch
        elif key == 'y':
            self._screenshot()
        elif key == 't':
            self._toggle_logging()
        elif key == 'c':
            self._export_csv()
        elif key == '0':
            self._toggle_main()
        elif key in ('1', '2', '3', '4'):
            self._toggle_ind(int(key) - 1)

    def _screenshot(self):
        if self._fig is None:
            return
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(SCREENSHOT_DIR, f'tof_screenshot_{ts}.png')
        self._fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Screenshot → {path}')

    def _toggle_logging(self):
        if self._logging:
            self._close_log()
            self._logging = False
            print('Logging stopped.')
        else:
            os.makedirs(LOG_DIR, exist_ok=True)
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(LOG_DIR, f'tof_{ts}.csv')
            log_f = open(path, 'w', newline='', buffering=1)
            self._log_file = log_f
            self._log_writer = csv.writer(log_f)
            self._log_writer.writerow([
                'timestamp', 'channel', 'min_mm', 'avg_mm', 'max_mm',
                'ir_label', 'obstacle_response',
            ])
            self._logging = True
            print(f'Logging → {path}')

    def _close_log(self):
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
            self._log_file   = None
            self._log_writer = None

    def _export_csv(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(LOG_DIR, f'tof_grids_{ts}.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['channel', 'row', 'col', 'distance_mm'])
            for ch in range(self._num_ch):
                grid = self._grids[ch]
                rows, cols = grid.shape
                for r in range(rows):
                    for c in range(cols):
                        w.writerow([ch, r, c, f'{grid[r, c]:.1f}'])
        print(f'Grid CSV → {path}')

    def _toggle_main(self):
        if self._fig is None:
            return
        try:
            win = getattr(self._fig.canvas.manager, "window", None)
            if win is None:
                return
            try:
                if win.winfo_ismapped():
                    win.withdraw()
                else:
                    win.deiconify()
                return
            except AttributeError:
                pass
            try:
                if win.isVisible():
                    win.hide()
                else:
                    win.show()
            except AttributeError:
                pass
        except Exception:
            pass

    def _toggle_ind(self, ch: int):
        fig = self._ind_figs[ch]
        if fig is not None and plt.fignum_exists(figure_number(fig)):
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


def main():
    parser = argparse.ArgumentParser(
        description='Standalone real-time ToF heatmap + 3-D surface viewer')
    parser.add_argument(
        '--port', type=int, default=TOF_DATA_PORT,
        help=(
            f'UDP listen port from braccio_ctrl (default {TOF_DATA_PORT}). '
            'This is not a /dev/tty device; use tof_serial_diagnose.py for serial.'
        ),
    )
    args = parser.parse_args()

    print(f'Listening for ToF data on UDP :{args.port}')
    print('Keyboard: [U] reset  [Y] screenshot  [T] log  [C] CSV  '
          '[0] toggle  [1-4] pop-out')
    app = ToFPlotterApp(port=args.port)
    app.run()


if __name__ == '__main__':
    main()
