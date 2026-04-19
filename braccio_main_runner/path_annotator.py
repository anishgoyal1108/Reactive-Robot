"""
path_annotator.py — Standalone matplotlib tool for annotating ideal arm paths.

Loads an rl_transitions_TIMESTAMP.npz file produced by RLRecorder, lets the
user draw an ideal path in theta-Z space (with r encoded as viridis colour
on the actual path), and writes DTW-derived rewards back to the NPZ.

Usage
-----
  python path_annotator.py logs/rl_transitions_20260419_120000.npz

Controls
--------
  D        — toggle Waypoint / Freehand mode
  Z        — undo last control point
  Shift+Z  — clear all control points (incl. r-θ)
  S        — increase spline smoothing (Gaussian pass on output curve)
  Shift+S  — decrease spline smoothing
  Enter    — accept drawn path, compute DTW rewards
  Escape   — discard drawing
  R        — toggle r-vs-theta subplot visibility
  Export   — button in right panel: writes rewards to NPZ

The r-vs-θ subplot is **draggable** when the session r varies: left-click
empty space to add a control point, drag to move, double-click to delete.
When r is effectively constant (sweep mode, std < 1 mm) the subplot
renders the seeded path but is non-interactive.

Export writes three things:
  * `rewards[]`          — DTW-derived rewards in-place
  * `drawn_theta_r_z[]`  — (M, 3) ideal path triples for downstream use
  * `_annotated.npz`     — sibling copy for archival / training pipelines
"""

import sys
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.patches import FancyBboxPatch

from scipy.interpolate import CubicSpline
from scipy.signal     import savgol_filter


# ── Constants ─────────────────────────────────────────────────────────────────
THETA_MIN, THETA_MAX = 0.0,   180.0
Z_MIN,     Z_MAX     = -250.0, 200.0
R_MIN,     R_MAX     =   10.0, 240.0
FREEHAND_SAVGOL_WIN  = 11
FREEHAND_SAVGOL_POLY = 3
DOUGLAS_PEUCKER_EPS  = 3.0     # degrees (theta-space)
DTW_REWARD_RANGE     = 2.0     # rewards will be in [-DTW_REWARD_RANGE, 0]


# ── Spline / smoothing helpers ────────────────────────────────────────────────

def catmull_rom(control_pts: np.ndarray, n_out: int = 300) -> np.ndarray:
    """Smooth chord-length-parameterised cubic spline through control_pts (N,2)."""
    if len(control_pts) < 2:
        return control_pts
    if len(control_pts) == 2:
        return np.linspace(control_pts[0], control_pts[1], n_out)
    d = np.cumsum(np.r_[0.0, np.sqrt(
        ((np.diff(control_pts, axis=0)) ** 2).sum(axis=1))])
    if d[-1] < 1e-9:
        return control_pts
    d /= d[-1]
    cs = CubicSpline(d, control_pts, bc_type='not-a-knot')
    return cs(np.linspace(0.0, 1.0, n_out))


def douglas_peucker(pts: np.ndarray, eps: float) -> np.ndarray:
    """Iterative Douglas-Peucker simplification (keeps endpoints)."""
    if len(pts) < 3:
        return pts
    keep  = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        s, e = stack.pop()
        if e - s < 2:
            continue
        a, b = pts[s], pts[e]
        ab = b - a
        ab_len2 = float(ab @ ab)
        if ab_len2 < 1e-9:
            continue
        # perpendicular distance of each intermediate point to line ab
        segs = pts[s + 1:e] - a
        t = (segs @ ab) / ab_len2
        proj = a + np.outer(t, ab)
        d = np.sqrt(((pts[s + 1:e] - proj) ** 2).sum(axis=1))
        if len(d) == 0:
            continue
        i = int(np.argmax(d))
        if d[i] > eps:
            keep[s + 1 + i] = True
            stack.append((s, s + 1 + i))
            stack.append((s + 1 + i, e))
    return pts[keep]


def smooth_freehand(raw: np.ndarray) -> np.ndarray:
    """Savitzky-Golay filter a raw drag trace (N,2)."""
    if len(raw) < FREEHAND_SAVGOL_WIN:
        return raw
    xs = savgol_filter(raw[:, 0], FREEHAND_SAVGOL_WIN, FREEHAND_SAVGOL_POLY)
    ys = savgol_filter(raw[:, 1], FREEHAND_SAVGOL_WIN, FREEHAND_SAVGOL_POLY)
    return np.column_stack([xs, ys])


def gaussian_smooth(curve: np.ndarray, sigma: float) -> np.ndarray:
    """
    Post-process a spline curve with a 1D Gaussian along each axis.

    `sigma` is in samples.  sigma <= 0 returns the input unchanged, matching
    the "0 smoothing = raw Catmull-Rom" behaviour of the UI slider.
    """
    if sigma <= 0.0 or len(curve) < 3:
        return curve
    from scipy.ndimage import gaussian_filter1d
    out = np.empty_like(curve)
    out[:, 0] = gaussian_filter1d(curve[:, 0], sigma=sigma, mode='nearest')
    out[:, 1] = gaussian_filter1d(curve[:, 1], sigma=sigma, mode='nearest')
    # Pin endpoints so smoothing doesn't walk them off the user's click
    out[0]  = curve[0]
    out[-1] = curve[-1]
    return out


# ── Dynamic Time Warping (pure NumPy; replaces unbuildable fastdtw) ───────────

def dtw_path(a: np.ndarray, b: np.ndarray) -> tuple:
    """
    Classic O(NM) DTW. a:(N,D), b:(M,D). Returns (distance, warping_path).
    warping_path is a list of (i, j) pairs from (0,0) to (N-1, M-1).
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, []
    inf  = np.inf
    D = np.full((n + 1, m + 1), inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            c = float(np.linalg.norm(ai - b[j - 1]))
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i -= 1; j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        path.append((i - 1, j - 1))
    path.reverse()
    return float(D[n, m]), path


def compute_dtw_rewards(actual_theta_z: np.ndarray,
                        drawn_theta_z:  np.ndarray) -> np.ndarray:
    """
    Per-timestep reward array (N,) in [-DTW_REWARD_RANGE, 0].
    actual_theta_z: (N, 2) normalised to [0,1]×[0,1].
    drawn_theta_z:  (M, 2) same normalisation.
    """
    if len(actual_theta_z) == 0 or len(drawn_theta_z) == 0:
        return np.zeros(len(actual_theta_z), dtype=np.float32)
    _, path = dtw_path(actual_theta_z, drawn_theta_z)
    per_step = np.zeros(len(actual_theta_z), dtype=np.float64)
    counts   = np.zeros(len(actual_theta_z), dtype=np.int64)
    for i, j in path:
        per_step[i] += float(np.linalg.norm(
            actual_theta_z[i] - drawn_theta_z[j]))
        counts[i]   += 1
    counts[counts == 0] = 1
    per_step /= counts
    max_cost = per_step.max() if per_step.max() > 0 else 1.0
    return (-DTW_REWARD_RANGE * (per_step / max_cost)).astype(np.float32)


# ── Coordinate normalisation helpers ──────────────────────────────────────────

def norm_theta(t):   return (t - THETA_MIN) / (THETA_MAX - THETA_MIN)
def norm_z(z):       return (z - Z_MIN)     / (Z_MAX     - Z_MIN)
def norm_theta_z(theta, z):
    return np.column_stack([norm_theta(theta), norm_z(z)])


# ── Session loading ───────────────────────────────────────────────────────────

class SessionData:
    """Loaded NPZ file: arrays + obstacle overlay points."""

    def __init__(self, path: str):
        self.path = path
        z = np.load(path, allow_pickle=False)
        self.obs       = z['obs']
        self.actions   = z['actions']
        self.rewards   = z['rewards'].astype(np.float32)
        self.next_obs  = z['next_obs']
        self.dones     = z['dones']
        self.theta     = z['theta'].astype(np.float32)
        self.r         = z['r'].astype(np.float32)
        self.z         = z['z'].astype(np.float32)
        self.timestamps = z['timestamps']
        self.n = len(self.theta)

        # Derive IR events and "close ToF" readings from obs vector if present
        # obs layout: [0:5]=arm, [5:21]=ch0, [21:37]=ch1, ..., [53]=ir_n
        if self.obs.shape[1] >= 54:
            self.ir_series = (self.obs[:, 53] * 3.0).round().astype(np.int8)
            ch0_grid = self.obs[:, 5:21]
            ch1_grid = self.obs[:, 21:37]
            # Un-normalize (obs was clip(d/250, 0, 2); 1.0 => NaN/clear)
            with np.errstate(invalid='ignore'):
                ch0_mm = np.where(ch0_grid >= 0.999, np.nan, ch0_grid * 250.0)
                ch1_mm = np.where(ch1_grid >= 0.999, np.nan, ch1_grid * 250.0)
            # nanmin warns for all-NaN rows (no obstacle this tick) — that's
            # the normal "no threat" case; silence it and keep NaN as output.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                self.ch0_min_mm = np.nanmin(ch0_mm, axis=1)
                self.ch1_min_mm = np.nanmin(ch1_mm, axis=1)
        else:
            self.ir_series  = np.zeros(self.n, dtype=np.int8)
            self.ch0_min_mm = np.full(self.n, np.nan, dtype=np.float32)
            self.ch1_min_mm = np.full(self.n, np.nan, dtype=np.float32)

        # Theta-direction reversals for vertical reference lines
        if self.n > 2:
            d = np.diff(self.theta)
            sign = np.sign(d)
            self.reversals = np.where(np.diff(sign) != 0)[0] + 1
        else:
            self.reversals = np.empty(0, dtype=np.int64)

    def obstacle_theta_z(self) -> np.ndarray:
        """Return (K, 2) array of obstacle events projected to theta-Z space."""
        mask = (self.ir_series > 0) | (self.ch0_min_mm < 200) | (self.ch1_min_mm < 200)
        return np.column_stack([self.theta[mask], self.z[mask]])


# ── Main annotator app ────────────────────────────────────────────────────────

class PathAnnotator:
    """Matplotlib-backed path annotation UI. Call run() to start the event loop."""

    MODE_WAYPOINT = 'waypoint'
    MODE_FREEHAND = 'freehand'

    def __init__(self, session: SessionData):
        self.session = session
        self.mode    = self.MODE_WAYPOINT
        # Primary (theta-Z) drawing
        self.control_pts: list = []        # list of [theta, z]
        self.freehand_raw: list = []       # accumulated drag samples
        self.freehand_drawing = False
        # r-vs-theta secondary drawing (draggable 1D spline, plan Stage 3)
        # If session r is effectively constant (std < threshold), the subplot
        # is rendered but frozen — matches "collapses to horizontal line and
        # is non-interactive" in the plan.
        self.r_control_pts: list = []      # list of [theta, r]
        self._r_constant = bool(self.session.r.std() < 1.0)
        # Smoothing factor (0 = raw Catmull-Rom; >0 adds a Gaussian smoothing
        # pass over the final spline curve — adjustable via slider or S keys)
        self.smoothing = 0.0
        self.show_r_subplot = True

        self._setup_figure()
        self._seed_r_control_pts()
        self._redraw_actual()
        self._redraw_drawn()
        self._connect_events()

    # ── Figure setup ──────────────────────────────────────────────────────

    def _setup_figure(self) -> None:
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title(
            f"Path Annotator — {os.path.basename(self.session.path)}")

        # Main theta-Z canvas (left 70%, upper 75%)
        self.ax_main = self.fig.add_axes([0.06, 0.25, 0.62, 0.70])
        self.ax_main.set_xlabel('theta (deg)')
        self.ax_main.set_ylabel('z (mm)')
        self.ax_main.set_xlim(THETA_MIN - 5, THETA_MAX + 5)
        self.ax_main.set_ylim(Z_MIN - 10, Z_MAX + 10)
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_title(
            'Draw ideal path (L-click=add, drag=move, dbl-click=delete)',
            fontsize=11)

        # Secondary r vs theta subplot (below main)
        self.ax_r = self.fig.add_axes([0.06, 0.07, 0.62, 0.14])
        self.ax_r.set_xlabel('theta (deg)')
        self.ax_r.set_ylabel('r (mm)')
        self.ax_r.set_xlim(THETA_MIN - 5, THETA_MAX + 5)
        self.ax_r.set_ylim(R_MIN - 10, R_MAX + 10)
        self.ax_r.grid(True, alpha=0.3)

        # Control panel (right 30%)
        self.ax_mode  = self.fig.add_axes([0.72, 0.84, 0.22, 0.09])
        self.radio    = RadioButtons(
            self.ax_mode, [self.MODE_WAYPOINT, self.MODE_FREEHAND], active=0)

        self.ax_undo  = self.fig.add_axes([0.72, 0.76, 0.10, 0.05])
        self.btn_undo = Button(self.ax_undo, 'Undo (Z)')

        self.ax_clear = self.fig.add_axes([0.84, 0.76, 0.10, 0.05])
        self.btn_clear = Button(self.ax_clear, 'Clear')

        # Smoothing slider — adds Gaussian smoothing to the Catmull-Rom
        # spline output.  S / Shift+S keys also adjust this.
        self.ax_smooth = self.fig.add_axes([0.74, 0.69, 0.20, 0.03])
        self.slider_smooth = Slider(
            self.ax_smooth, 'Smooth', 0.0, 10.0,
            valinit=self.smoothing, valstep=0.5,
        )

        self.ax_accept = self.fig.add_axes([0.72, 0.60, 0.22, 0.06])
        self.btn_accept = Button(self.ax_accept, 'Accept (Enter)  →  DTW')

        self.ax_export = self.fig.add_axes([0.72, 0.52, 0.22, 0.06])
        self.btn_export = Button(self.ax_export, 'Export NPZ')

        self.ax_status = self.fig.add_axes([0.72, 0.25, 0.22, 0.25])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(
            0.0, 1.0,
            f"N transitions : {self.session.n}\n"
            f"Mode          : {self.mode}\n"
            f"Smoothing     : 0.0\n"
            f"θ-z pts       : 0\n"
            f"r-θ pts       : 0\n"
            f"DTW rewards   : (not computed)\n",
            fontsize=10, family='monospace', va='top',
        )

        # Persistent artist handles (set during redraw)
        self._actual_line    = None
        self._actual_scatter = None
        self._obstacle_scat  = None
        self._reversal_lines = []
        self._drawn_line     = None
        self._drawn_pts      = None
        self._r_actual_line  = None
        self._r_drawn_line   = None
        self._r_drawn_pts    = None     # draggable diamonds on r subplot

        self._last_rewards: np.ndarray = None

    # ── r-subplot seeding ─────────────────────────────────────────────────

    def _seed_r_control_pts(self) -> None:
        """
        Seed r-θ control points from the actual session.

        If r is effectively constant (sweep mode), place exactly two points
        at the endpoints — the subplot renders but is non-interactive.
        Otherwise place 5 evenly-spaced sample points along the actual path.
        """
        if self.session.n == 0:
            return
        theta_sorted = np.sort(self.session.theta)
        if self._r_constant:
            r_const = float(np.median(self.session.r))
            self.r_control_pts = [[float(theta_sorted[0]),  r_const],
                                  [float(theta_sorted[-1]), r_const]]
            return
        # Sample 5 points along actual trajectory, sorted by theta
        order = np.argsort(self.session.theta)
        idx   = np.linspace(0, len(order) - 1, 5).astype(int)
        pts   = [[float(self.session.theta[order[i]]),
                  float(self.session.r[order[i]])] for i in idx]
        self.r_control_pts = pts

    # ── Redraw helpers ────────────────────────────────────────────────────

    def _redraw_actual(self) -> None:
        """Draw the recorded actual arm path with r as viridis colour + overlays."""
        ax = self.ax_main

        # Remove stale artists
        if self._actual_line is not None:
            self._actual_line.remove()
        if self._actual_scatter is not None:
            self._actual_scatter.remove()
        if self._obstacle_scat is not None:
            self._obstacle_scat.remove()
        for l in self._reversal_lines:
            l.remove()
        self._reversal_lines.clear()

        # Line (thin, light) + scatter (coloured by r)
        self._actual_line, = ax.plot(
            self.session.theta, self.session.z,
            color='steelblue', linewidth=0.8, alpha=0.6, label='actual path',
        )
        r_norm = (self.session.r - R_MIN) / max(1.0, (R_MAX - R_MIN))
        self._actual_scatter = ax.scatter(
            self.session.theta, self.session.z,
            c=r_norm, cmap='viridis', s=6, vmin=0.0, vmax=1.0,
            label='r value (viridis)', zorder=3,
        )

        obs_pts = self.session.obstacle_theta_z()
        if len(obs_pts) > 0:
            self._obstacle_scat = ax.scatter(
                obs_pts[:, 0], obs_pts[:, 1],
                color='crimson', marker='x', s=40, alpha=0.7,
                label='obstacle event', zorder=4,
            )

        for idx in self.session.reversals:
            if idx < len(self.session.theta):
                l = ax.axvline(self.session.theta[idx],
                               color='gray', linestyle='--', alpha=0.3)
                self._reversal_lines.append(l)

        if self.session.n > 0:
            ax.scatter([self.session.theta[0]],  [self.session.z[0]],
                       color='lime',    s=60, zorder=5, label='start')
            ax.scatter([self.session.theta[-1]], [self.session.z[-1]],
                       color='darkred', s=60, zorder=5, label='end')

        ax.legend(loc='upper right', fontsize=8)

        # r subplot — actual r over theta
        if self._r_actual_line is not None:
            self._r_actual_line.remove()
        self._r_actual_line, = self.ax_r.plot(
            self.session.theta, self.session.r,
            color='steelblue', linewidth=1.0, alpha=0.7,
        )

    def _redraw_drawn(self) -> None:
        """Re-render the user's annotation layer."""
        ax = self.ax_main
        if self._drawn_line is not None:
            self._drawn_line.remove();  self._drawn_line = None
        if self._drawn_pts is not None:
            self._drawn_pts.remove();   self._drawn_pts  = None
        if self._r_drawn_line is not None:
            self._r_drawn_line.remove(); self._r_drawn_line = None

        if len(self.control_pts) >= 2:
            pts = np.asarray(self.control_pts)
            curve = catmull_rom(pts, n_out=400)
            curve = gaussian_smooth(curve, sigma=self.smoothing)
            self._drawn_line, = ax.plot(
                curve[:, 0], curve[:, 1],
                color='darkorange', linewidth=2.2, label='ideal path',
            )

        if self.control_pts:
            pts = np.asarray(self.control_pts)
            self._drawn_pts = ax.scatter(
                pts[:, 0], pts[:, 1],
                color='darkorange', marker='D', s=60,
                edgecolors='black', zorder=6,
            )

        # ── r-subplot: draggable 1D spline through r_control_pts ─────────
        if self._r_drawn_pts is not None:
            self._r_drawn_pts.remove()
            self._r_drawn_pts = None
        if self.r_control_pts and len(self.r_control_pts) >= 2:
            rpts = np.asarray(self.r_control_pts)
            rcurve = catmull_rom(rpts, n_out=300)
            rcurve = gaussian_smooth(rcurve, sigma=self.smoothing)
            self._r_drawn_line, = self.ax_r.plot(
                rcurve[:, 0], rcurve[:, 1],
                color='darkorange', linewidth=1.8, label='ideal r',
            )
            # Only render draggable handles if r isn't frozen-constant
            if not self._r_constant:
                self._r_drawn_pts = self.ax_r.scatter(
                    rpts[:, 0], rpts[:, 1],
                    color='darkorange', marker='D', s=45,
                    edgecolors='black', zorder=6,
                )

        self._refresh_status()
        self.fig.canvas.draw_idle()

    def _refresh_status(self) -> None:
        if self._last_rewards is not None:
            rew_line = (f"DTW rewards   : min={self._last_rewards.min():.3f} "
                        f"max={self._last_rewards.max():.3f}")
        else:
            rew_line = "DTW rewards   : (not computed)"
        r_note = " (frozen)" if self._r_constant else ""
        self.status_text.set_text(
            f"N transitions : {self.session.n}\n"
            f"Mode          : {self.mode}\n"
            f"Smoothing     : {self.smoothing:.1f}\n"
            f"θ-z pts       : {len(self.control_pts)}\n"
            f"r-θ pts       : {len(self.r_control_pts)}{r_note}\n"
            f"{rew_line}\n"
        )

    # ── Event wiring ──────────────────────────────────────────────────────

    def _connect_events(self) -> None:
        cid = self.fig.canvas.mpl_connect
        cid('button_press_event',   self._on_press)
        cid('motion_notify_event',  self._on_motion)
        cid('button_release_event', self._on_release)
        cid('key_press_event',      self._on_key)

        self.radio.on_clicked(self._on_mode_change)
        self.btn_undo.on_clicked(lambda _e: self._undo())
        self.btn_clear.on_clicked(lambda _e: self._clear())
        self.btn_accept.on_clicked(lambda _e: self._accept())
        self.btn_export.on_clicked(lambda _e: self._export())
        self.slider_smooth.on_changed(self._on_smooth_change)

        # Drag state: index into control_pts (main) or r_control_pts (r subplot)
        self._drag_idx:    int  = -1
        self._drag_target: str  = 'main'    # 'main' or 'r'

    # ── Mode / button handlers ────────────────────────────────────────────

    def _on_mode_change(self, label: str) -> None:
        self.mode = label
        self.freehand_raw.clear()
        self.freehand_drawing = False
        self._refresh_status()

    def _undo(self) -> None:
        if self.control_pts:
            self.control_pts.pop()
            self._last_rewards = None
            self._redraw_drawn()

    def _clear(self) -> None:
        self.control_pts.clear()
        self.freehand_raw.clear()
        self.freehand_drawing = False
        # Reset r-subplot control points to their seeded defaults
        self.r_control_pts.clear()
        self._seed_r_control_pts()
        self._last_rewards = None
        self._redraw_drawn()

    # ── Mouse handlers ────────────────────────────────────────────────────

    def _on_press(self, ev) -> None:
        # ── r subplot: click+drag a point (only if r isn't frozen) ───────
        if ev.inaxes is self.ax_r and ev.xdata is not None \
                and not self._r_constant:
            x, y = float(ev.xdata), float(ev.ydata)
            idx = self._nearest_r_pt_idx(x, y, tol_theta=4.0, tol_r=15.0)
            if ev.dblclick and idx >= 0:
                del self.r_control_pts[idx]
                self._last_rewards = None
                self._redraw_drawn()
                return
            if idx >= 0:
                self._drag_idx    = idx
                self._drag_target = 'r'
                return
            # Add new r control pt (left-click empty space)
            if ev.button == 1:
                self.r_control_pts.append([x, y])
                self.r_control_pts.sort(key=lambda p: p[0])
                self._last_rewards = None
                self._redraw_drawn()
            return

        if ev.inaxes is not self.ax_main or ev.xdata is None:
            return
        x, y = float(ev.xdata), float(ev.ydata)

        if self.mode == self.MODE_WAYPOINT:
            idx = self._nearest_pt_idx(x, y, tol_theta=4.0, tol_z=10.0)
            # Double-click → delete
            if ev.dblclick and idx >= 0:
                del self.control_pts[idx]
                self._last_rewards = None
                self._redraw_drawn()
                return
            # Existing point → start drag
            if idx >= 0:
                self._drag_idx    = idx
                self._drag_target = 'main'
                return
            # Empty space → add new point
            if ev.button == 1:
                self.control_pts.append([x, y])
                self._last_rewards = None
                self._redraw_drawn()

        elif self.mode == self.MODE_FREEHAND:
            if ev.button == 1:
                self.freehand_drawing = True
                self.freehand_raw = [[x, y]]

    def _on_motion(self, ev) -> None:
        # Dragging r-subplot control point
        if self._drag_target == 'r' and self._drag_idx >= 0 \
                and ev.inaxes is self.ax_r and ev.xdata is not None:
            x, y = float(ev.xdata), float(ev.ydata)
            self.r_control_pts[self._drag_idx] = [x, y]
            self._last_rewards = None
            self._redraw_drawn()
            return

        if ev.inaxes is not self.ax_main or ev.xdata is None:
            return
        x, y = float(ev.xdata), float(ev.ydata)

        if self.mode == self.MODE_WAYPOINT \
                and self._drag_target == 'main' and self._drag_idx >= 0:
            self.control_pts[self._drag_idx] = [x, y]
            self._last_rewards = None
            self._redraw_drawn()

        elif self.mode == self.MODE_FREEHAND and self.freehand_drawing:
            self.freehand_raw.append([x, y])
            raw = np.asarray(self.freehand_raw)
            if self._drawn_line is not None:
                self._drawn_line.remove()
            self._drawn_line, = self.ax_main.plot(
                raw[:, 0], raw[:, 1], color='darkorange',
                linewidth=1.8, alpha=0.8,
            )
            self.fig.canvas.draw_idle()

    def _on_release(self, ev) -> None:
        # Terminate any drag (main or r)
        if self._drag_idx >= 0:
            if self._drag_target == 'r':
                self.r_control_pts.sort(key=lambda p: p[0])
                self._redraw_drawn()
            self._drag_idx    = -1
            self._drag_target = 'main'

        if self.mode == self.MODE_WAYPOINT:
            return

        if self.mode == self.MODE_FREEHAND and self.freehand_drawing:
            self.freehand_drawing = False
            if len(self.freehand_raw) < 3:
                self.freehand_raw.clear()
                return
            raw = np.asarray(self.freehand_raw)
            smoothed  = smooth_freehand(raw)
            simplified = douglas_peucker(smoothed, DOUGLAS_PEUCKER_EPS)
            self.control_pts = [list(p) for p in simplified]
            self.freehand_raw.clear()
            self._last_rewards = None
            self._redraw_drawn()

    # ── Keyboard handlers ─────────────────────────────────────────────────

    def _on_key(self, ev) -> None:
        k = (ev.key or '').lower()
        if k == 'd':
            new_mode = (self.MODE_FREEHAND if self.mode == self.MODE_WAYPOINT
                        else self.MODE_WAYPOINT)
            self.radio.set_active(
                [self.MODE_WAYPOINT, self.MODE_FREEHAND].index(new_mode))
        elif ev.key == 'Z':      # shift+z = clear all
            self._clear()
        elif k == 'z':
            self._undo()
        elif ev.key == 'S':      # shift+s = decrease smoothing
            self._bump_smoothing(-0.5)
        elif k == 's':           # s = increase smoothing
            self._bump_smoothing(+0.5)
        elif k == 'enter':
            self._accept()
        elif k == 'escape':
            self._clear()
        elif k == 'r':
            self.show_r_subplot = not self.show_r_subplot
            self.ax_r.set_visible(self.show_r_subplot)
            self.fig.canvas.draw_idle()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _nearest_pt_idx(self, x: float, y: float,
                        tol_theta: float, tol_z: float) -> int:
        """Return index of control point near (x,y), or -1."""
        if not self.control_pts:
            return -1
        pts = np.asarray(self.control_pts)
        dx = np.abs(pts[:, 0] - x) / tol_theta
        dy = np.abs(pts[:, 1] - y) / tol_z
        d  = np.sqrt(dx * dx + dy * dy)
        i = int(np.argmin(d))
        return i if d[i] < 1.0 else -1

    def _nearest_r_pt_idx(self, x: float, y: float,
                          tol_theta: float, tol_r: float) -> int:
        """Return index of r-subplot control point near (x,y), or -1."""
        if not self.r_control_pts:
            return -1
        pts = np.asarray(self.r_control_pts)
        dx = np.abs(pts[:, 0] - x) / tol_theta
        dy = np.abs(pts[:, 1] - y) / tol_r
        d  = np.sqrt(dx * dx + dy * dy)
        i = int(np.argmin(d))
        return i if d[i] < 1.0 else -1

    # ── Smoothing ────────────────────────────────────────────────────────

    def _on_smooth_change(self, val: float) -> None:
        self.smoothing = float(val)
        self._last_rewards = None
        self._redraw_drawn()

    def _bump_smoothing(self, delta: float) -> None:
        new_val = max(0.0, min(10.0, self.smoothing + delta))
        self.slider_smooth.set_val(new_val)   # triggers _on_smooth_change

    # ── Accept / Export ───────────────────────────────────────────────────

    def _accept(self) -> None:
        """Compute DTW rewards from current drawn path and store internally."""
        if len(self.control_pts) < 2:
            print("[path_annotator] Need at least 2 control points before Accept")
            return
        if self.session.n == 0:
            return

        drawn_curve = catmull_rom(np.asarray(self.control_pts), n_out=400)
        drawn_n = np.column_stack([
            norm_theta(drawn_curve[:, 0]),
            norm_z    (drawn_curve[:, 1]),
        ])
        actual_n = norm_theta_z(self.session.theta, self.session.z)
        rewards = compute_dtw_rewards(actual_n, drawn_n)

        # 50/50 blend with existing placeholder rewards so the annotator can be
        # layered over a pre-existing reward signal without blowing it away.
        blended = 0.5 * rewards + 0.5 * self.session.rewards
        self._last_rewards = blended.astype(np.float32)
        self._refresh_status()
        print(f"[path_annotator] DTW rewards computed "
              f"(min={self._last_rewards.min():.3f}, "
              f"max={self._last_rewards.max():.3f})")

    def _build_drawn_triples(self, n_out: int = 400) -> np.ndarray:
        """
        Combine the θ-z and r-θ control points into an (n_out, 3) array
        of (theta, r, z) ideal-path triples.

        The θ-z spline parameterises the path; r is sampled from the
        r-θ spline at each curve theta (with extrapolation via np.interp).
        If the user never edited the r subplot, r defaults to the session
        median (or the seeded actual-r curve for varying-r sessions).
        """
        if len(self.control_pts) < 2:
            return np.zeros((0, 3), dtype=np.float32)
        theta_z = catmull_rom(np.asarray(self.control_pts), n_out=n_out)
        theta_z = gaussian_smooth(theta_z, sigma=self.smoothing)

        if self.r_control_pts and len(self.r_control_pts) >= 2:
            rpts   = np.asarray(sorted(self.r_control_pts, key=lambda p: p[0]))
            rcurve = catmull_rom(rpts, n_out=n_out)
            rcurve = gaussian_smooth(rcurve, sigma=self.smoothing)
            r_samples = np.interp(theta_z[:, 0], rcurve[:, 0], rcurve[:, 1])
        else:
            r_samples = np.full(len(theta_z),
                                float(np.median(self.session.r)),
                                dtype=np.float32)
        return np.column_stack([
            theta_z[:, 0],   # theta
            r_samples,       # r
            theta_z[:, 1],   # z
        ]).astype(np.float32)

    def _export(self) -> None:
        """Write DTW rewards back into NPZ + an annotated sidecar."""
        if self._last_rewards is None:
            print("[path_annotator] Press Accept before Export (no rewards yet)")
            return

        base, ext = os.path.splitext(self.session.path)
        annotated_path = base + "_annotated" + ext
        triples = self._build_drawn_triples()

        arrays = {
            'obs':        self.session.obs,
            'actions':    self.session.actions,
            'rewards':    self._last_rewards,
            'next_obs':   self.session.next_obs,
            'dones':      self.session.dones,
            'theta':      self.session.theta,
            'r':          self.session.r,
            'z':          self.session.z,
            'timestamps': self.session.timestamps,
            'drawn_theta_r_z': triples,   # (M, 3) ideal-path triples
        }

        np.savez_compressed(annotated_path, **arrays)
        # Overwrite original rewards field in place (keep drawn_theta_r_z too)
        np.savez_compressed(self.session.path, **arrays)
        print(f"[path_annotator] Exported ({len(triples)} triples):\n"
              f"  {annotated_path}\n  {self.session.path}")

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        plt.show()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Annotate ideal arm paths on recorded RL sessions.')
    parser.add_argument(
        'npz_path',
        help='Path to an rl_transitions_*.npz produced by RLRecorder')
    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"error: file not found: {args.npz_path}", file=sys.stderr)
        return 1

    session   = SessionData(args.npz_path)
    annotator = PathAnnotator(session)
    annotator.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
