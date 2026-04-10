#!/usr/bin/env python3
"""
arm_plotter_app.py — Standalone real-time joint-angle plotter.

Receives arm data over UDP from braccio_ctrl and displays live
matplotlib graphs.  Runs entirely on the main thread — no threading
issues with Tk/Qt backends.

Usage:
    python arm_plotter_app.py [--port PORT]

Keyboard shortcuts (focus the plot window):
    U       — reset: clear data, restart t = 0
    Y       — save screenshot (PNG)
    T       — toggle CSV logging
    0       — toggle main combined window
    1–6     — toggle per-joint pop-out window
"""

import argparse
import csv
import json
import os
import socket
import time
from collections import deque
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from braccio_ctrl.constants import (
    JOINT_NAMES, JOINT_COLORS, JOINT_LIMITS,
    PLOT_WINDOW_S, PLOT_UPDATE_HZ, PLOT_Y_MARGIN_DEG,
    LOG_DIR, SCREENSHOT_DIR, ARM_DATA_PORT,
)

_LIMIT_COLOR = '#F38BA8'
_LIMIT_ALPHA = 0.18


class ArmPlotterApp:
    def __init__(self, port: int = ARM_DATA_PORT, window_s: float = PLOT_WINDOW_S):
        self._port     = port
        self._window_s = window_s
        self._t0       = time.monotonic()

        maxlen = int(window_s * 20)
        self._times  = deque(maxlen=maxlen)
        self._angles = [deque(maxlen=maxlen) for _ in range(6)]

        # UDP socket (non-blocking)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(('127.0.0.1', port))
        self._sock.setblocking(False)

        # CSV logging
        self._logging    = False
        self._log_file   = None
        self._log_writer = None

        # Main figure
        self._fig   = None
        self._axes  = []
        self._lines = []
        self._status_txt = None

        # Pop-out figures
        self._ind_figs  = [None] * 6
        self._ind_axes  = [None] * 6
        self._ind_lines = [None] * 6

    def run(self):
        self._build_main_figure()

        self._anim = FuncAnimation(
            self._fig, self._update,
            interval=int(1000 / PLOT_UPDATE_HZ),
            blit=False, cache_frame_data=False,
        )
        plt.show()

    # ── Figure construction ───────────────────────────────────────────────

    def _build_main_figure(self):
        fig, axes_arr = plt.subplots(2, 3, figsize=(14, 7), sharex=True,
                                     constrained_layout=True)
        self._fig  = fig
        self._axes = axes_arr.flatten().tolist()
        self._lines = []

        fig.suptitle('Braccio — Joint Angles (Real-Time)', fontsize=11)

        for i, ax in enumerate(self._axes):
            lo, hi = JOINT_LIMITS[i]
            ax.axhspan(lo, hi, alpha=_LIMIT_ALPHA, color=_LIMIT_COLOR, zorder=0)
            ax.set_ylim(lo - PLOT_Y_MARGIN_DEG, hi + PLOT_Y_MARGIN_DEG)
            ax.set_xlim(0, self._window_s)
            ax.set_title(JOINT_NAMES[i].strip(), fontsize=9)
            ax.set_ylabel('°', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25)
            if i >= 3:
                ax.set_xlabel('Time (s)', fontsize=8)
            line, = ax.plot([], [], color=JOINT_COLORS[i], lw=1.6,
                            antialiased=True)
            self._lines.append(line)

        self._status_txt = self._axes[0].text(
            0.02, 0.96, '', transform=self._axes[0].transAxes,
            fontsize=7, va='top', color='green',
        )
        hint = '[U] reset  [Y] screenshot  [T] log  [0] toggle  [1-6] pop-out'
        fig.text(0.5, 0.002, hint, ha='center', fontsize=7, color='#666666')
        fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _build_ind_figure(self, i: int):
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
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        return fig, ax, line

    # ── Animation callback ────────────────────────────────────────────────

    def _update(self, _frame):
        # Drain all pending UDP packets, keep latest data
        while True:
            try:
                data, _ = self._sock.recvfrom(4096)
                msg = json.loads(data)
                t = time.monotonic() - self._t0
                self._times.append(t)
                for i, a in enumerate(msg['joints']):
                    self._angles[i].append(a)
            except BlockingIOError:
                break
            except Exception:
                break

        if not self._times:
            return self._lines

        t_data = list(self._times)
        a_data = [list(d) for d in self._angles]
        t_now  = t_data[-1]
        x_min  = max(0.0, t_now - self._window_s)
        x_max  = x_min + self._window_s

        for ax, line, angles in zip(self._axes, self._lines, a_data):
            line.set_data(t_data, angles)
            ax.set_xlim(x_min, x_max)

        self._status_txt.set_text('● REC' if self._logging else '')

        # CSV logging
        if self._logging and self._log_writer and t_data:
            try:
                self._log_writer.writerow(
                    [f'{t_data[-1]:.4f}'] + [str(a[-1]) for a in a_data if a]
                )
            except Exception:
                self._close_log()

        # Pop-out windows
        for i in range(6):
            fig = self._ind_figs[i]
            if fig is None:
                continue
            if plt.fignum_exists(fig.number):
                self._ind_lines[i].set_data(t_data, a_data[i])
                self._ind_axes[i].set_xlim(x_min, x_max)
                fig.canvas.draw_idle()
            else:
                self._ind_figs[i]  = None
                self._ind_axes[i]  = None
                self._ind_lines[i] = None

        return self._lines

    # ── Keyboard ──────────────────────────────────────────────────────────

    def _on_key(self, event):
        key = (event.key or '').lower()
        if key == 'u':
            self._reset()
        elif key == 'y':
            self._screenshot()
        elif key == 't':
            self._toggle_logging()
        elif key == '0':
            self._toggle_main()
        elif len(key) == 1 and '1' <= key <= '6':
            self._toggle_ind(int(key) - 1)

    def _reset(self):
        self._t0 = time.monotonic()
        self._times.clear()
        for d in self._angles:
            d.clear()

    def _screenshot(self):
        if self._fig is None:
            return
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(SCREENSHOT_DIR, f'screenshot_{ts}.png')
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
            path = os.path.join(LOG_DIR, f'braccio_{ts}.csv')
            self._log_file   = open(path, 'w', newline='', buffering=1)
            self._log_writer = csv.writer(self._log_file)
            self._log_writer.writerow(
                ['time_s'] + [n.strip() for n in JOINT_NAMES]
            )
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

    def _toggle_main(self):
        if self._fig is None:
            return
        try:
            win = self._fig.canvas.manager.window
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

    def _toggle_ind(self, i: int):
        fig = self._ind_figs[i]
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)
            self._ind_figs[i]  = None
            self._ind_axes[i]  = None
            self._ind_lines[i] = None
        else:
            f, a, ln = self._build_ind_figure(i)
            self._ind_figs[i]  = f
            self._ind_axes[i]  = a
            self._ind_lines[i] = ln


def main():
    parser = argparse.ArgumentParser(
        description='Standalone real-time arm joint-angle plotter')
    parser.add_argument(
        '--port', type=int, default=ARM_DATA_PORT,
        help=(
            f'UDP listen port from braccio_ctrl (default {ARM_DATA_PORT}); '
            'not a serial device path.'
        ),
    )
    args = parser.parse_args()

    print(f'Listening for arm data on UDP :{args.port}')
    print('Keyboard: [U] reset  [Y] screenshot  [T] log  '
          '[0] toggle  [1-6] pop-out')
    app = ArmPlotterApp(port=args.port)
    app.run()


if __name__ == '__main__':
    main()
