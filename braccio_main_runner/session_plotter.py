#!/usr/bin/env python3
"""Session playback and export tool for Braccio recordings."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import matplotlib
if "--export-mp4" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from scipy.spatial import ConvexHull
except Exception:
    ConvexHull = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from braccio_ctrl.constants import JOINT_NAMES
from planning.sensor_config import SENSOR_CONFIG, sensor_urdf_path
from planning.urdf_kinematics import BraccioURDFKinematics


@dataclass
class SessionData:
    session_dir: Path
    records: list[dict]
    times_s: np.ndarray
    meta: dict


@dataclass
class LaunchConfig:
    session_path: Path
    action: str
    resolution_px: tuple[int, int]
    view_name: str
    view_elev: float
    view_azim: float


@dataclass
class VisualSpec:
    link_name: str
    vertices: np.ndarray
    faces: np.ndarray
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    color_rgba: tuple[float, float, float, float]


class SessionPlotter:
    _REPLAN_TRAJECTORY_MODES = frozenset(
        {
            "confirm_stop",
            "vertical_probe",
            "planning_hold",
            "avoid_execute",
            "hold_no_path",
        }
    )

    def __init__(
        self,
        data: SessionData,
        resolution_px: tuple[int, int] = (1280, 720),
        view_elev: float = 24.0,
        view_azim: float = -55.0,
    ):
        self.data = data
        self.records = data.records
        self.times = data.times_s
        self._resolution_px = (int(resolution_px[0]), int(resolution_px[1]))
        self._view_elev = float(view_elev)
        self._view_azim = float(view_azim)

        urdf_path = Path(data.meta.get("urdf_path", ""))
        preferred_urdf = sensor_urdf_path()
        if (not urdf_path.exists()) or ("with_tof" not in urdf_path.name.lower() and preferred_urdf.exists()):
            urdf_path = preferred_urdf
        self.urdf_path = urdf_path
        self.kin = BraccioURDFKinematics(urdf_path)
        self._visual_specs = self._load_visual_specs(urdf_path)

        supported = list(SENSOR_CONFIG.get("supported_channels", [0, 1, 2, 3]))
        n_cfg = _infer_session_channel_count(data)
        n_cfg = max(1, min(n_cfg, len(supported)))
        self._tof_channels = supported[:n_cfg]
        self._sensor_layout_label = "Hyperion 4-ToF" if n_cfg > 2 else "Dual 2-ToF"

        threshold_mm = float(data.meta.get("threshold_mm", 200.0))
        self._tof_cone_range_m = max(0.12, min(1.0, 1.15 * (threshold_mm / 1000.0)))

        self._channel_colors = {
            0: "#3B82F6",  # west / -Y boresight
            1: "#F97316",  # east / +Y boresight
            2: "#22C55E",  # north / +Z boresight
            3: "#A855F7",  # south / -Z boresight
        }

        imu_rot = np.asarray(data.meta.get("imu_rot_wrist_from_imu", []), dtype=float)
        if imu_rot.shape == (3, 3):
            self._imu_rot_wrist_from_imu = imu_rot
        else:
            self._imu_rot_wrist_from_imu = np.eye(3, dtype=float) if bool(data.meta.get("imu_calibrated", False)) else None
        self._imu_calibrated = bool(data.meta.get("imu_calibrated", False)) and (self._imu_rot_wrist_from_imu is not None)
        self._imu_calibration_source = "meta" if imu_rot.shape == (3, 3) else ("accel_fallback" if self._imu_calibrated else "off")
        self._imu_fuse_weight = 0.8
        self._imu_max_tilt_deg = 35.0
        self._planner_clearance_mm = float(data.meta.get("planner_clearance_mm", 10.0))

        self._idx = 0
        self._paused = False
        self._obs_scatter = None
        self._blocked_pocket_scatter = None
        self._obs_geom_scatters: list = []
        self._obs_geom_hulls: list = []
        self.tool_points = self._precompute_tool_points()
        self._timeseries = self._build_timeseries()

        width_px, height_px = self._resolution_px
        dpi = 100
        self.fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.7, 1.35], height_ratios=[1.0, 0.92])
        self.ax = self.fig.add_subplot(gs[:, 0], projection="3d")
        self.joints_ax = self.fig.add_subplot(gs[0, 1])
        self.imu_ax = self.fig.add_subplot(gs[1, 1])
        self.fig.subplots_adjust(left=0.03, right=0.98, bottom=0.06, top=0.94, wspace=0.18, hspace=0.32)

        self._setup_axes()
        self._setup_side_axes()
        self.ax.view_init(elev=self._view_elev, azim=self._view_azim)

        self.robot_line, = self.ax.plot([], [], [], "-o", color="#ff8f2a", linewidth=1.2, markersize=3, alpha=0.25)
        self.tool_marker, = self.ax.plot([], [], [], "o", color="#00c36b", markersize=6)
        self.breadcrumb_line, = self.ax.plot([], [], [], color="#00c36b", linewidth=2.0)
        self.replanned_line, = self.ax.plot([], [], [], "-", color="#2563EB", linewidth=2.8, alpha=0.95)
        self.planned_line, = self.ax.plot([], [], [], "--", color="#38BDF8", linewidth=2.6, alpha=1.0)
        self._planned_scatter = None
        self._mesh_artists = self._init_robot_mesh_artists()
        self._replan_history_points = self._build_replan_history_points(self.records, self.tool_points)
        self.fig.patch.set_facecolor("#0F172A")
        self.ax.set_facecolor("#111827")
        self.joints_ax.set_facecolor("#111827")
        self.imu_ax.set_facecolor("#111827")
        self.ax.legend(
            handles=[
                Line2D([0], [0], color="#00c36b", lw=2.0, label="Recent tool path"),
                Line2D([0], [0], color="#2563EB", lw=2.8, label="Replanned path taken"),
                Line2D([0], [0], color="#38BDF8", lw=2.6, ls="--", label="Remaining future plan"),
            ],
            loc="upper right",
            fontsize=7.0,
            facecolor="#0F172A",
            edgecolor="#334155",
            labelcolor="#E5E7EB",
        )

        self._draw_feasible_cloud()
        self._init_tof_coverage_artists()
        self._init_side_plot_artists()

        dt = np.diff(self.times)
        dt_med = float(np.median(dt)) if dt.size > 0 else 0.04
        self._fps = max(1, min(60, int(round(1.0 / max(1e-3, dt_med)))))
        self._interval_ms = int(max(10, round(1000.0 / self._fps)))

        self.anim = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=self._interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _setup_axes(self) -> None:
        lo, hi, aspect = self._workspace_limits_for_points(self._workspace_bounds_points())
        self.ax.set_xlim(float(lo[0]), float(hi[0]))
        self.ax.set_ylim(float(lo[1]), float(hi[1]))
        self.ax.set_zlim(float(lo[2]), float(hi[2]))
        try:
            self.ax.set_box_aspect(aspect)
        except Exception:
            pass
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title("Braccio Session Playback")
        self.ax.xaxis.pane.set_facecolor((0.08, 0.11, 0.17, 0.12))
        self.ax.yaxis.pane.set_facecolor((0.08, 0.11, 0.17, 0.12))
        self.ax.zaxis.pane.set_facecolor((0.08, 0.11, 0.17, 0.12))
        self.ax.grid(True, alpha=0.18)

    def _workspace_bounds_points(self) -> np.ndarray:
        pts: list[np.ndarray] = [np.array([[0.0, 0.0, 0.0]], dtype=float)]
        if self.tool_points.size > 0:
            pts.append(np.asarray(self.tool_points, dtype=float).reshape(-1, 3))

        stride = max(1, len(self.records) // 160)
        indices = list(range(0, len(self.records), stride))
        if self.records and indices[-1] != len(self.records) - 1:
            indices.append(len(self.records) - 1)

        for idx in indices:
            rec = self.records[idx]
            joints = [float(x) for x in rec.get("joints_deg", [90, 90, 90, 90, 90, 73])]
            try:
                pts.append(np.asarray(self.kin.chain_positions_m(joints), dtype=float).reshape(-1, 3))
            except Exception:
                pass
            for obs in rec.get("obstacles", []):
                try:
                    radius_m = max(0.0, float(obs.get("radius_mm", 0.0)) / 1000.0)
                    center = np.array(
                        [[float(obs.get("x_m", 0.0)), float(obs.get("y_m", 0.0)), float(obs.get("z_m", 0.0))]],
                        dtype=float,
                    )
                    offsets = np.array(
                        [
                            [0.0, 0.0, 0.0],
                            [radius_m, 0.0, 0.0],
                            [-radius_m, 0.0, 0.0],
                            [0.0, radius_m, 0.0],
                            [0.0, -radius_m, 0.0],
                            [0.0, 0.0, radius_m],
                            [0.0, 0.0, -radius_m],
                        ],
                        dtype=float,
                    )
                    pts.append(center + offsets)
                except Exception:
                    continue
            plan = rec.get("future_plan", [])
            if plan:
                plan_pts = []
                for point in plan:
                    xyz = self._future_plan_point_m(point)
                    if xyz is not None:
                        plan_pts.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                if plan_pts:
                    pts.append(np.asarray(plan_pts, dtype=float))

        if not pts:
            return np.zeros((0, 3), dtype=float)
        arr = np.vstack(pts)
        finite = np.all(np.isfinite(arr), axis=1)
        return arr[finite]

    @staticmethod
    def _workspace_limits_for_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
        arr = np.asarray(points, dtype=float).reshape(-1, 3) if np.asarray(points).size else np.zeros((0, 3), dtype=float)
        finite = np.all(np.isfinite(arr), axis=1) if arr.size else np.zeros((0,), dtype=bool)
        arr = arr[finite]
        if arr.size <= 0:
            arr = np.array([[-0.22, -0.22, 0.0], [0.22, 0.22, 0.28]], dtype=float)

        lo = np.min(arr, axis=0)
        hi = np.max(arr, axis=0)
        center = 0.5 * (lo + hi)
        ranges = np.maximum(hi - lo, 1e-6)

        xy_span = max(0.62, 1.25 * float(max(ranges[0], ranges[1])) + 0.10)
        z_span = max(0.42, 1.25 * float(ranges[2]) + 0.08)
        x_lo = float(center[0] - 0.5 * xy_span)
        x_hi = float(center[0] + 0.5 * xy_span)
        y_lo = float(center[1] - 0.5 * xy_span)
        y_hi = float(center[1] + 0.5 * xy_span)

        z_lo = min(float(lo[2] - 0.05), -0.04)
        z_hi = max(float(hi[2] + 0.06), z_lo + z_span)
        z_span = float(z_hi - z_lo)
        return (
            np.array([x_lo, y_lo, z_lo], dtype=float),
            np.array([x_hi, y_hi, z_hi], dtype=float),
            (float(xy_span), float(xy_span), float(z_span)),
        )

    def _setup_side_axes(self) -> None:
        for axis in (self.joints_ax, self.imu_ax):
            axis.grid(True, alpha=0.18, color="#334155")
            axis.tick_params(colors="#CBD5E1", labelsize=8)
            for spine in axis.spines.values():
                spine.set_color("#334155")
            axis.xaxis.label.set_color("#E5E7EB")
            axis.yaxis.label.set_color("#E5E7EB")
            axis.title.set_color("#E5E7EB")

        self.joints_ax.set_title("Joint States")
        self.joints_ax.set_xlabel("Time [s]")
        self.joints_ax.set_ylabel("Angle [deg]")
        self.imu_ax.set_title("IMU / ToF State")
        self.imu_ax.set_xlabel("Time [s]")
        self.imu_ax.set_ylabel("Scaled value")

    def _mode_color(self, mode: str) -> str:
        m = str(mode)
        return {
            "detect_sweep": "#22C55E",
            "confirm_stop": "#EAB308",
            "vertical_probe": "#A78BFA",
            "planning_hold": "#38BDF8",
            "avoid_execute": "#F59E0B",
            "tracking_sweep": "#14B8A6",
            "hold_no_path": "#EF4444",
        }.get(m, "#64748B")

    def _build_timeseries(self) -> dict[str, np.ndarray]:
        t = np.asarray(self.times, dtype=float)
        severity = np.array(
            [float(rec.get("planner_debug", {}).get("debug", {}).get("severity", 0.0)) for rec in self.records],
            dtype=float,
        )
        clearance = np.array(
            [float(rec.get("planner_debug", {}).get("debug", {}).get("min_clearance_mm", -1.0)) for rec in self.records],
            dtype=float,
        )
        clearance[~np.isfinite(clearance)] = -1.0
        speed = np.array(
            [float(rec.get("planner_debug", {}).get("debug", {}).get("speed_scale", 1.0)) for rec in self.records],
            dtype=float,
        )
        objective = []
        joints = np.zeros((len(self.records), 6), dtype=float)
        tof_min = np.full((len(self.records), max(1, len(self._tof_channels))), np.nan, dtype=float)
        tof_active = np.zeros(len(self.records), dtype=float)
        imu_accel_norm = np.zeros(len(self.records), dtype=float)
        imu_gyro_norm = np.zeros(len(self.records), dtype=float)
        imu_quality = np.zeros(len(self.records), dtype=float)
        imu_rms = np.zeros(len(self.records), dtype=float)
        modes = []
        for idx, rec in enumerate(self.records):
            planner_root = rec.get("planner_debug", {})
            goal = planner_root.get("goal", {})
            command = planner_root.get("command", {})
            theta_err = abs(float(goal.get("theta", 0.0)) - float(command.get("theta", 0.0))) / 180.0
            r_err = abs(float(goal.get("r_mm", 0.0)) - float(command.get("r_mm", 0.0))) / 240.0
            z_err = abs(float(goal.get("effective_z_mm", goal.get("z_mm", 0.0))) - float(command.get("z_mm", 0.0))) / 250.0
            clearance_term = 0.0
            if np.isfinite(clearance[idx]) and clearance[idx] >= 0.0:
                clearance_term = max(0.0, (self._planner_clearance_mm - clearance[idx]) / max(1.0, self._planner_clearance_mm))
            objective.append(theta_err + r_err + z_err + clearance_term + 0.35 * float(severity[idx]))
            joints[idx, : len(rec.get("joints_deg", []))] = np.asarray(rec.get("joints_deg", [0, 0, 0, 0, 0, 0]), dtype=float)[:6]
            mins = rec.get("tof_min_by_ch_mm", [])
            for ch_idx in range(min(len(mins), tof_min.shape[1])):
                value = mins[ch_idx]
                if value is not None:
                    tof_min[idx, ch_idx] = float(value)
            tof_active[idx] = float(rec.get("tof_active_count", 0.0))
            imu = rec.get("imu", {})
            ax = float(imu.get("ax_g", 0.0))
            ay = float(imu.get("ay_g", 0.0))
            az = float(imu.get("az_g", 0.0))
            gx = float(imu.get("gx_dps", 0.0))
            gy = float(imu.get("gy_dps", 0.0))
            gz = float(imu.get("gz_dps", 0.0))
            imu_accel_norm[idx] = float(math.sqrt(ax * ax + ay * ay + az * az))
            imu_gyro_norm[idx] = float(math.sqrt(gx * gx + gy * gy + gz * gz))
            imu_rms[idx] = float(imu.get("rms_deg", self.data.meta.get("imu_calibration_rms_deg", 0.0)))
            imu_quality[idx] = 1.0 if bool(imu.get("online", False)) and bool(imu.get("calibrated", False)) else 0.0
            modes.append(str(rec.get("mode", "")))
        return {
            "t": t,
            "severity": severity,
            "clearance": clearance,
            "speed": speed,
            "objective": np.asarray(objective, dtype=float),
            "joints": joints,
            "tof_min": tof_min,
            "tof_active": tof_active,
            "imu_accel_norm": imu_accel_norm,
            "imu_gyro_norm": imu_gyro_norm,
            "imu_quality": imu_quality,
            "imu_rms": imu_rms,
            "modes": np.asarray(modes, dtype=object),
        }

    def _mode_segments(self) -> list[tuple[float, float, str]]:
        if len(self.records) == 0:
            return []
        segments: list[tuple[float, float, str]] = []
        modes = self._timeseries["modes"]
        start_idx = 0
        current = str(modes[0])
        for idx in range(1, len(modes)):
            if str(modes[idx]) != current:
                segments.append((float(self.times[start_idx]), float(self.times[idx - 1]), current))
                start_idx = idx
                current = str(modes[idx])
        segments.append((float(self.times[start_idx]), float(self.times[-1]), current))
        return segments

    def _init_side_plot_artists(self) -> None:
        t = self._timeseries["t"]
        t_end = float(t[-1]) if t.size else 1.0

        for axis in (self.joints_ax, self.imu_ax):
            for t0, t1, mode in self._mode_segments():
                axis.axvspan(t0, t1, color=self._mode_color(mode), alpha=0.06, linewidth=0.0)
            axis.set_xlim(0.0, max(1.0, t_end))

        joint_colors = ["#60A5FA", "#F97316", "#22C55E", "#EAB308", "#A78BFA", "#F472B6"]
        joint_labels = [str(name).strip() for name in JOINT_NAMES[:6]]
        self._joint_lines = []
        for idx, (color, label) in enumerate(zip(joint_colors, joint_labels)):
            line, = self.joints_ax.plot([], [], color=color, linewidth=1.25, label=label)
            self._joint_lines.append(line)
        self._joints_cursor = self.joints_ax.axvline(0.0, color="#E5E7EB", linewidth=1.0, alpha=0.7)
        joint_vals = self._timeseries["joints"][:, :6] if self._timeseries["joints"].size else np.zeros((1, 6), dtype=float)
        j_lo = float(np.nanmin(joint_vals)) if joint_vals.size else 0.0
        j_hi = float(np.nanmax(joint_vals)) if joint_vals.size else 180.0
        pad = max(5.0, 0.08 * (j_hi - j_lo + 1e-6))
        self.joints_ax.set_ylim(j_lo - pad, j_hi + pad)
        self.joints_ax.legend(loc="upper right", fontsize=6.6, facecolor="#0F172A", edgecolor="#334155", labelcolor="#E5E7EB", ncol=2)

        self._imu_accel_line, = self.imu_ax.plot([], [], color="#22C55E", linewidth=1.2, label="|accel| g")
        self._imu_gyro_line, = self.imu_ax.plot([], [], color="#F97316", linewidth=1.2, label="|gyro| /100")
        self._imu_quality_line, = self.imu_ax.plot([], [], color="#38BDF8", linewidth=1.2, label="calibrated")
        self._imu_rms_line, = self.imu_ax.plot([], [], color="#A78BFA", linewidth=1.1, label="rms /20")
        self._imu_cursor = self.imu_ax.axvline(0.0, color="#E5E7EB", linewidth=1.0, alpha=0.7)
        imu_max = max(
            1.2,
            float(np.nanmax(self._timeseries["imu_accel_norm"])) if self._timeseries["imu_accel_norm"].size else 0.0,
            float(np.nanmax(self._timeseries["imu_gyro_norm"] / 100.0)) if self._timeseries["imu_gyro_norm"].size else 0.0,
            float(np.nanmax(self._timeseries["imu_rms"] / 20.0)) if self._timeseries["imu_rms"].size else 0.0,
        )
        self.imu_ax.set_ylim(-0.05, imu_max * 1.1)
        self.imu_ax.legend(loc="upper right", fontsize=6.6, facecolor="#0F172A", edgecolor="#334155", labelcolor="#E5E7EB", ncol=2)

    def _update_side_plots(self, idx: int) -> None:
        end = max(1, int(idx) + 1)
        t = self._timeseries["t"][:end]
        t_now = float(self._timeseries["t"][min(idx, len(self._timeseries["t"]) - 1)])
        self._joints_cursor.set_xdata([t_now, t_now])
        self._imu_cursor.set_xdata([t_now, t_now])
        for joint_idx, line in enumerate(self._joint_lines):
            line.set_data(t, self._timeseries["joints"][:end, joint_idx])
        self._imu_accel_line.set_data(t, self._timeseries["imu_accel_norm"][:end])
        self._imu_gyro_line.set_data(t, self._timeseries["imu_gyro_norm"][:end] / 100.0)
        self._imu_quality_line.set_data(t, self._timeseries["imu_quality"][:end])
        self._imu_rms_line.set_data(t, self._timeseries["imu_rms"][:end] / 20.0)

    @staticmethod
    def _nearest_tof_m(tof_min_mm: np.ndarray) -> np.ndarray:
        arr = np.asarray(tof_min_mm, dtype=float)
        if arr.ndim != 2 or arr.size <= 0:
            return np.zeros((0,), dtype=float)
        out = np.full((arr.shape[0],), np.nan, dtype=float)
        finite = np.isfinite(arr)
        for idx in range(arr.shape[0]):
            if np.any(finite[idx]):
                out[idx] = float(np.min(arr[idx, finite[idx]])) / 1000.0
        return out

    def _load_feasible_cloud(self) -> np.ndarray | None:
        return None

    def _precompute_tool_points(self) -> np.ndarray:
        out = np.zeros((len(self.records), 3), dtype=float)
        for i, rec in enumerate(self.records):
            joints = [float(x) for x in rec.get("joints_deg", [90, 90, 90, 90, 90, 73])]
            try:
                tf = self.kin.joint_transform(joints, "wrist_roll")
                out[i, :] = tf[:3, 3]
            except Exception:
                out[i, :] = np.asarray(rec.get("eef_m", [0.0, 0.0, 0.0]), dtype=float)
        return out

    @staticmethod
    def _box_mesh(size_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sx, sy, sz = [0.5 * float(v) for v in np.asarray(size_xyz, dtype=float).reshape(3)]
        verts = np.array(
            [
                [-sx, -sy, -sz],
                [sx, -sy, -sz],
                [sx, sy, -sz],
                [-sx, sy, -sz],
                [-sx, -sy, sz],
                [sx, -sy, sz],
                [sx, sy, sz],
                [-sx, sy, sz],
            ],
            dtype=float,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [1, 2, 6], [1, 6, 5],
                [2, 3, 7], [2, 7, 6],
                [3, 0, 4], [3, 4, 7],
            ],
            dtype=int,
        )
        return verts, faces

    @staticmethod
    def _load_stl_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
        raw = path.read_bytes()
        if len(raw) >= 84:
            tri_count = int.from_bytes(raw[80:84], byteorder="little", signed=False)
            expected = 84 + (50 * tri_count)
            if expected == len(raw):
                data = np.frombuffer(
                    raw[84:],
                    dtype=np.dtype([("normal", "<f4", (3,)), ("verts", "<f4", (3, 3)), ("attr", "<u2")]),
                )
                verts = np.asarray(data["verts"], dtype=float).reshape(-1, 3)
                faces = np.arange(verts.shape[0], dtype=int).reshape(-1, 3)
                return verts, faces

        verts = []
        for line in raw.decode("utf-8", errors="ignore").splitlines():
            txt = line.strip()
            if txt.lower().startswith("vertex"):
                parts = txt.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not verts:
            raise RuntimeError(f"unsupported STL format: {path}")
        arr = np.asarray(verts, dtype=float)
        face_count = arr.shape[0] // 3
        arr = arr[: face_count * 3, :]
        faces = np.arange(arr.shape[0], dtype=int).reshape(-1, 3)
        return arr, faces

    @staticmethod
    def _parse_color_rgba(text: str | None, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if not text:
            return default
        parts = text.strip().split()
        if len(parts) != 4:
            return default
        try:
            vals = tuple(float(x) for x in parts)
        except Exception:
            return default
        return vals

    def _load_visual_specs(self, urdf_path: Path) -> list[VisualSpec]:
        root = ElementTree.parse(urdf_path).getroot()
        materials: dict[str, tuple[float, float, float, float]] = {}
        for mat in root.findall("material"):
            name = mat.attrib.get("name", "")
            color_elem = mat.find("color")
            if name and color_elem is not None:
                materials[name] = self._parse_color_rgba(color_elem.attrib.get("rgba"), (0.85, 0.55, 0.18, 1.0))

        specs: list[VisualSpec] = []
        for link in root.findall("link"):
            link_name = link.attrib.get("name", "")
            for vis in link.findall("visual"):
                origin = vis.find("origin")
                origin_xyz = np.asarray([0.0, 0.0, 0.0], dtype=float)
                origin_rpy = np.asarray([0.0, 0.0, 0.0], dtype=float)
                if origin is not None:
                    xyz_txt = origin.attrib.get("xyz", "0 0 0").split()
                    rpy_txt = origin.attrib.get("rpy", "0 0 0").split()
                    if len(xyz_txt) == 3:
                        origin_xyz = np.asarray([float(x) for x in xyz_txt], dtype=float)
                    if len(rpy_txt) == 3:
                        origin_rpy = np.asarray([float(x) for x in rpy_txt], dtype=float)

                mat_elem = vis.find("material")
                color = (0.85, 0.55, 0.18, 0.95)
                if mat_elem is not None:
                    if "name" in mat_elem.attrib and mat_elem.attrib["name"] in materials:
                        color = materials[mat_elem.attrib["name"]]
                    color_elem = mat_elem.find("color")
                    if color_elem is not None:
                        color = self._parse_color_rgba(color_elem.attrib.get("rgba"), color)

                geom = vis.find("geometry")
                if geom is None:
                    continue
                mesh_elem = geom.find("mesh")
                box_elem = geom.find("box")

                if mesh_elem is not None:
                    mesh_path = (urdf_path.parent / mesh_elem.attrib.get("filename", "")).resolve()
                    if not mesh_path.exists():
                        continue
                    try:
                        verts, faces = self._load_stl_mesh(mesh_path)
                    except Exception:
                        continue
                elif box_elem is not None:
                    size_txt = box_elem.attrib.get("size", "0.1 0.1 0.1").split()
                    size = np.asarray([float(x) for x in size_txt[:3]], dtype=float)
                    verts, faces = self._box_mesh(size)
                else:
                    continue

                specs.append(
                    VisualSpec(
                        link_name=link_name,
                        vertices=np.asarray(verts, dtype=float),
                        faces=np.asarray(faces, dtype=int),
                        origin_xyz=origin_xyz,
                        origin_rpy=origin_rpy,
                        color_rgba=tuple(float(x) for x in color),
                    )
                )

        return specs

    def _init_robot_mesh_artists(self) -> list[Poly3DCollection]:
        artists: list[Poly3DCollection] = []
        for spec in self._visual_specs:
            artist = Poly3DCollection(
                [],
                facecolors=[spec.color_rgba],
                edgecolors=[(*spec.color_rgba[:3], min(1.0, spec.color_rgba[3] * 0.55))],
                linewidths=0.08,
                alpha=spec.color_rgba[3],
            )
            self.ax.add_collection3d(artist)
            artists.append(artist)
        return artists

    def _draw_robot_meshes(self, joints: list[float]) -> None:
        for spec, artist in zip(self._visual_specs, self._mesh_artists):
            try:
                link_tf = self.kin.link_transform(joints, spec.link_name)
            except Exception:
                artist.set_verts([])
                continue

            vis_tf = np.eye(4, dtype=float)
            vis_tf[:3, :3] = self._rot_axis_angle(np.array([1.0, 0.0, 0.0], dtype=float), 0.0)
            roll, pitch, yaw = [float(v) for v in spec.origin_rpy]
            cr = math.cos(roll)
            sr = math.sin(roll)
            cp = math.cos(pitch)
            sp = math.sin(pitch)
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
            ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
            rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            vis_tf[:3, :3] = rz @ ry @ rx
            vis_tf[:3, 3] = spec.origin_xyz

            tf = link_tf @ vis_tf
            verts_world = (tf[:3, :3] @ spec.vertices.T).T + tf[:3, 3]
            faces = [verts_world[idxs, :] for idxs in spec.faces]
            artist.set_verts(faces)

    def _draw_feasible_cloud(self) -> None:
        return

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            return np.zeros_like(v)
        return v / n

    @staticmethod
    def _rot_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        n = float(np.linalg.norm(axis))
        if n <= 1e-12 or abs(float(angle_rad)) <= 1e-12:
            return np.eye(3, dtype=float)
        u = axis / n
        ux, uy, uz = float(u[0]), float(u[1]), float(u[2])
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        omc = 1.0 - c
        return np.array(
            [
                [c + ux * ux * omc, ux * uy * omc - uz * s, ux * uz * omc + uy * s],
                [uy * ux * omc + uz * s, c + uy * uy * omc, uy * uz * omc - ux * s],
                [uz * ux * omc - uy * s, uz * uy * omc + ux * s, c + uz * uz * omc],
            ],
            dtype=float,
        )

    @staticmethod
    def _rot_zyx_deg(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
        y = math.radians(float(yaw_deg))
        p = math.radians(float(pitch_deg))
        r = math.radians(float(roll_deg))
        cz, sz = math.cos(y), math.sin(y)
        cy, sy = math.cos(p), math.sin(p)
        cx, sx = math.cos(r), math.sin(r)
        rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
        return rz @ ry @ rx

    @staticmethod
    def _orthonormalize_rot(rot: np.ndarray) -> np.ndarray:
        try:
            u, _, vt = np.linalg.svd(np.asarray(rot, dtype=float).reshape(3, 3))
            out = u @ vt
            if np.linalg.det(out) < 0.0:
                u[:, -1] *= -1.0
                out = u @ vt
            return out
        except Exception:
            return np.eye(3, dtype=float)

    @staticmethod
    def _rotmat_to_axis_angle(rot: np.ndarray) -> tuple[np.ndarray, float]:
        r = np.asarray(rot, dtype=float).reshape(3, 3)
        trace = float(np.trace(r))
        cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
        angle = math.acos(cos_angle)
        if angle <= 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=float), 0.0
        axis = np.array(
            [
                r[2, 1] - r[1, 2],
                r[0, 2] - r[2, 0],
                r[1, 0] - r[0, 1],
            ],
            dtype=float,
        )
        n = float(np.linalg.norm(axis))
        if n <= 1e-9:
            return np.array([1.0, 0.0, 0.0], dtype=float), angle
        return axis / n, angle

    def _imu_correct_wrist_rotation(self, wrist_rot_fk: np.ndarray, rec: dict | None) -> np.ndarray:
        if rec is None or not self._imu_calibrated or self._imu_rot_wrist_from_imu is None:
            return wrist_rot_fk

        imu = rec.get("imu", {})
        alignment = rec.get("imu_alignment", {})
        calibrated = bool(alignment.get("calibrated", False))
        anchor_valid = bool(alignment.get("anchor_valid", False))
        if not calibrated:
            return wrist_rot_fk

        rot_ref_from_imu = np.asarray(imu.get("rot_ref_from_imu", []), dtype=float)
        rot_base_from_ref = np.asarray(alignment.get("rot_base_from_ref", []), dtype=float)
        if rot_ref_from_imu.shape == (3, 3) and rot_base_from_ref.shape == (3, 3) and anchor_valid:
            rot_ref_from_wrist = rot_ref_from_imu @ self._imu_rot_wrist_from_imu.T
            wrist_rot_est = self._orthonormalize_rot(rot_base_from_ref @ rot_ref_from_wrist)
            rot_err = wrist_rot_est @ np.asarray(wrist_rot_fk, dtype=float).reshape(3, 3).T
            axis_ref, angle = self._rotmat_to_axis_angle(rot_err)
            angle = min(math.radians(self._imu_max_tilt_deg), max(0.0, angle * float(self._imu_fuse_weight)))
            if angle > 1e-6:
                return self._orthonormalize_rot(self._rot_axis_angle(axis_ref, angle) @ wrist_rot_fk)
            return wrist_rot_fk

        if bool(imu.get("online", False)):
            a = np.array(
                [
                    float(imu.get("ax_g", 0.0)),
                    float(imu.get("ay_g", 0.0)),
                    float(imu.get("az_g", 0.0)),
                ],
                dtype=float,
            )
            a = self._normalize(a)
            if float(np.linalg.norm(a)) > 1e-6:
                g_meas_wrist = self._normalize(self._imu_rot_wrist_from_imu @ a)
                g_fk_wrist = self._normalize(wrist_rot_fk.T @ np.array([0.0, 0.0, -1.0], dtype=float))
                dot = float(np.dot(g_meas_wrist, g_fk_wrist))
                dot = max(-1.0, min(1.0, dot))
                axis = np.cross(g_meas_wrist, g_fk_wrist)
                axis_n = float(np.linalg.norm(axis))
                if axis_n > 1e-9:
                    angle = math.atan2(axis_n, dot)
                    angle = min(math.radians(self._imu_max_tilt_deg), angle * float(self._imu_fuse_weight))
                    if angle > 1e-6:
                        return wrist_rot_fk @ self._rot_axis_angle(axis, angle)
        return wrist_rot_fk

    def _channel_color(self, ch: int) -> str:
        return self._channel_colors.get(int(ch), "#94A3B8")

    def _init_tof_coverage_artists(self) -> None:
        self._tof_cones: dict[int, Poly3DCollection] = {}
        self._tof_markers = {}

        for ch in self._tof_channels:
            color = self._channel_color(ch)
            cone = Poly3DCollection(
                [],
                facecolors=color,
                edgecolors=color,
                linewidths=0.4,
                alpha=0.08,
            )
            self.ax.add_collection3d(cone)
            marker, = self.ax.plot([], [], [], "o", color=color, markersize=4)
            self._tof_cones[ch] = cone
            self._tof_markers[ch] = marker

    def _sensor_basis(self, axis_dir: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        axis = self._normalize(axis_dir)
        uph = self._normalize(up_hint)
        right = np.cross(uph, axis)
        if float(np.linalg.norm(right)) <= 1e-9:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(axis, fallback))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            right = np.cross(fallback, axis)
        right = self._normalize(right)
        up = self._normalize(np.cross(axis, right))
        return axis, right, up

    def _legacy_sensor_basis(
        self,
        cfg: dict,
        wrist_origin: np.ndarray,
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        wrist_offset_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base_rot = self._rot_zyx_deg(theta_deg, 0.0, 0.0)
        arm_pitch_deg = math.degrees(math.atan2(float(z_mm), max(abs(float(r_mm)), 1e-3)))
        arm_rot = self._rot_zyx_deg(0.0, arm_pitch_deg + float(wrist_offset_deg), 0.0)
        mount_rot = self._rot_zyx_deg(
            float(cfg.get("yaw_deg", 0.0)),
            float(cfg.get("pitch_deg", 0.0)),
            float(cfg.get("roll_deg", 0.0)),
        )
        local_origin = np.asarray(cfg.get("origin_m", [0.0, 0.0, 0.0]), dtype=float)
        origin = np.asarray(wrist_origin, dtype=float).reshape(3) + ((base_rot @ arm_rot) @ local_origin)
        world_rot = base_rot @ arm_rot @ mount_rot
        axis = self._normalize(world_rot @ np.array([1.0, 0.0, 0.0], dtype=float))
        right = self._normalize(world_rot @ np.array([0.0, 1.0, 0.0], dtype=float))
        up = self._normalize(world_rot @ np.array([0.0, 0.0, 1.0], dtype=float))
        return origin, axis, right, up

    def _cone_faces(
        self,
        origin: np.ndarray,
        axis: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        fov_x_deg: float,
        fov_y_deg: float,
        cone_range_m: float,
        segments: int = 22,
    ) -> list[list[np.ndarray]]:
        fx = math.radians(float(fov_x_deg)) * 0.5
        fy = math.radians(float(fov_y_deg)) * 0.5

        ring = []
        for k in range(segments):
            phi = (2.0 * math.pi * k) / float(segments)
            ax = math.cos(phi) * fx
            ay = math.sin(phi) * fy
            d = self._normalize(axis + math.tan(ax) * right + math.tan(ay) * up)
            ring.append(origin + cone_range_m * d)

        faces: list[list[np.ndarray]] = []
        for k in range(segments):
            p0 = ring[k]
            p1 = ring[(k + 1) % segments]
            faces.append([origin, p0, p1])
        faces.append(ring)
        return faces

    def _update_tof_coverage(self, joints: list[float], rec: dict | None = None) -> None:
        try:
            tf_wp = self.kin.link_transform(joints, "link4")
            wp_origin = tf_wp[:3, 3].copy()
            wp_rot = tf_wp[:3, :3].copy()
        except Exception:
            for ch in self._tof_channels:
                self._tof_cones[ch].set_verts([])
                self._tof_markers[ch].set_data([], [])
                self._tof_markers[ch].set_3d_properties([])
            return

        for ch in self._tof_channels:
            cfg = SENSOR_CONFIG.get("channels", {}).get(ch)
            if cfg is None:
                self._tof_cones[ch].set_verts([])
                self._tof_markers[ch].set_data([], [])
                self._tof_markers[ch].set_3d_properties([])
                continue

            if "axis_dir" in cfg:
                wp_rot_used = self._imu_correct_wrist_rotation(wp_rot, rec)
                local_origin = np.asarray(cfg.get("origin_m", [0.0, 0.0, 0.0]), dtype=float)
                axis_local = np.asarray(cfg.get("axis_dir", [1.0, 0.0, 0.0]), dtype=float)
                up_local = np.asarray(cfg.get("up_hint", [0.0, 0.0, 1.0]), dtype=float)
                origin = wp_origin + (wp_rot_used @ local_origin)
                axis = wp_rot_used @ self._normalize(axis_local)
                up_hint = wp_rot_used @ self._normalize(up_local)
                axis, right, up = self._sensor_basis(axis, up_hint)
            else:
                theta_deg = float((rec or {}).get("theta", 90.0))
                r_mm = float((rec or {}).get("r_mm", 0.0))
                z_mm = float((rec or {}).get("z_mm", 0.0))
                wrist_offset_deg = float((rec or {}).get("wrist_offset_deg", 0.0))
                origin, axis, right, up = self._legacy_sensor_basis(
                    cfg,
                    wp_origin,
                    theta_deg=theta_deg,
                    r_mm=r_mm,
                    z_mm=z_mm,
                    wrist_offset_deg=wrist_offset_deg,
                )

            faces = self._cone_faces(
                origin=origin,
                axis=axis,
                right=right,
                up=up,
                fov_x_deg=float(cfg.get("fov_x_deg", 45.0)),
                fov_y_deg=float(cfg.get("fov_y_deg", 45.0)),
                cone_range_m=self._tof_cone_range_m,
            )
            self._tof_cones[ch].set_verts(faces)
            self._tof_markers[ch].set_data([origin[0]], [origin[1]])
            self._tof_markers[ch].set_3d_properties([origin[2]])

    def _update(self, _frame) -> None:
        if not self._paused:
            self._idx += 1
            if self._idx >= len(self.records):
                self._idx = len(self.records) - 1
                self._paused = True

        self._draw_frame(self._idx)

    def _draw_frame(self, idx: int) -> None:
        rec = self.records[idx]
        joints = [float(x) for x in rec.get("joints_deg", [90, 90, 90, 90, 90, 73])]
        chain = self.kin.chain_positions_m(joints)

        self.robot_line.set_data(chain[:, 0], chain[:, 1])
        self.robot_line.set_3d_properties(chain[:, 2])
        self._draw_robot_meshes(joints)

        tool = self.tool_points[idx]
        self.tool_marker.set_data([tool[0]], [tool[1]])
        self.tool_marker.set_3d_properties([tool[2]])

        self._update_tof_coverage(joints, rec)
        self._draw_obstacles(rec)
        self._draw_breadcrumb(idx)
        self._draw_replan_history(idx)
        self._draw_future_plan(rec)
        self._update_side_plots(idx)

        wall_t = rec.get("wall_time", 0.0)
        mode = rec.get("mode", "")
        planner_active = "ON" if rec.get("planner_active", False) else "off"
        self.ax.set_title(
            f"Session Playback  |  layout={self._sensor_layout_label}  |  "
            f"t={wall_t:.2f}  |  mode={mode}  |  planner={planner_active}"
        )

    def _update_status_text(self, rec: dict) -> None:
        obs = rec.get("obstacle", {})
        planner_root = rec.get("planner_debug", {})
        planner = rec.get("planner_debug", {}).get("debug", {})
        cmd_vel = rec.get("cmd_velocity", {})
        imu = rec.get("imu", {})
        vertical = planner_root.get("vertical_probe", {})
        tof_mins = rec.get("tof_min_by_ch_mm", [])
        tof_text = " ".join(
            f"CH{idx}={float(value):.0f}" for idx, value in enumerate(tof_mins) if value is not None
        )
        accel_norm = math.sqrt(
            float(imu.get("ax_g", 0.0)) ** 2 + float(imu.get("ay_g", 0.0)) ** 2 + float(imu.get("az_g", 0.0)) ** 2
        )
        gyro_norm = math.sqrt(
            float(imu.get("gx_dps", 0.0)) ** 2 + float(imu.get("gy_dps", 0.0)) ** 2 + float(imu.get("gz_dps", 0.0)) ** 2
        )
        selected = ""
        for cand in planner_root.get("candidate_plans", []):
            if bool(cand.get("accepted", False)):
                selected = str(cand.get("name", ""))
                break
        lines = [
            f"layout    : {self._sensor_layout_label}",
            f"view      : elev={self._view_elev:.1f}  azim={self._view_azim:.1f}",
            f"mount src : repository ToF geometry",
            f"labels    : CH0/CH1 detect  CH2/CH3 support",
            f"mode      : {rec.get('mode', '')}",
            f"planner   : {rec.get('planner_model', self.data.meta.get('planner_model', 'sector_waypoints_v1'))}",
            f"selected  : {selected}",
            f"obstacle  : {obs.get('response', '')}  src={obs.get('source', '')}",
            f"distance  : {obs.get('distance_mm', -1.0):.1f} mm",
            f"plan mode : {planner_root.get('mode', planner_root.get('planner_mode', rec.get('planner_mode', '')))}",
            f"opening   : {vertical.get('opening_pose', None)}",
            f"severity  : {planner.get('severity', 0.0):.2f}",
            f"clearance : {planner.get('min_clearance_mm', -1.0):.1f} mm",
            f"speed     : {planner.get('speed_scale', 1.0):.2f}",
            f"tof act   : {rec.get('tof_active_count', 0)}  {tof_text}",
            f"imu       : online={bool(imu.get('online', False))} cal={bool(imu.get('calibrated', False))} src={self._imu_calibration_source}",
            f"imu norm  : accel={accel_norm:.2f}g gyro={gyro_norm:.1f}dps rms={float(imu.get('rms_deg', 0.0)):.1f}",
            f"theta vel : {cmd_vel.get('theta_deg_s', 0.0):.2f} deg/s",
            f"r vel     : {cmd_vel.get('r_mm_s', 0.0):.2f} mm/s",
            f"z vel     : {cmd_vel.get('z_mm_s', 0.0):.2f} mm/s",
        ]
        self._status_text.set_text("\n".join(lines))

    def _draw_obstacles(self, rec: dict) -> None:
        if self._obs_scatter is not None:
            self._obs_scatter.remove()
            self._obs_scatter = None
        if self._blocked_pocket_scatter is not None:
            self._blocked_pocket_scatter.remove()
            self._blocked_pocket_scatter = None
        for art in self._obs_geom_scatters:
            try:
                art.remove()
            except Exception:
                pass
        for art in self._obs_geom_hulls:
            try:
                art.remove()
            except Exception:
                pass
        self._obs_geom_scatters = []
        self._obs_geom_hulls = []

        proj = rec.get("projection_debug", {})
        for ch_debug in proj.get("channels", []):
            pts = [
                np.asarray(cell.get("point_base_m", [0.0, 0.0, 0.0]), dtype=float)
                for cell in ch_debug.get("cells", [])
                if bool(cell.get("in_threshold", False)) and bool(cell.get("feasible", False))
            ]
            if not pts:
                continue
            arr = np.asarray(pts, dtype=float)
            color = self._channel_color(int(ch_debug.get("channel", -1)))
            sc = self.ax.scatter(
                arr[:, 0],
                arr[:, 1],
                arr[:, 2],
                s=30,
                c=color,
                alpha=0.22,
                marker="o",
                depthshade=False,
            )
            self._obs_geom_scatters.append(sc)
            if ConvexHull is not None and arr.shape[0] >= 4:
                try:
                    hull = ConvexHull(arr)
                    faces = [arr[simplex, :] for simplex in hull.simplices]
                    poly = Poly3DCollection(
                        faces,
                        facecolors=color,
                        edgecolors=color,
                        linewidths=0.2,
                        alpha=0.06,
                    )
                    self.ax.add_collection3d(poly)
                    self._obs_geom_hulls.append(poly)
                except Exception:
                    pass

        blocked_pockets = rec.get("blocked_pockets", [])
        if blocked_pockets:
            pts = []
            for pocket in blocked_pockets:
                try:
                    theta_deg = float(pocket.get("theta", 0.0))
                    r_mm = float(pocket.get("r_mm", 0.0))
                    z_mm = float(pocket.get("z_mm", 0.0))
                    theta = math.radians(theta_deg)
                    pts.append(
                        [
                            (r_mm / 1000.0) * math.cos(theta),
                            (r_mm / 1000.0) * math.sin(theta),
                            z_mm / 1000.0,
                        ]
                    )
                except Exception:
                    continue
            if pts:
                arr = np.asarray(pts, dtype=float)
                self._blocked_pocket_scatter = self.ax.scatter(
                    arr[:, 0],
                    arr[:, 1],
                    arr[:, 2],
                    s=18,
                    c="#FDE68A",
                    alpha=0.28,
                    marker="s",
                    depthshade=False,
                )

        obs = rec.get("obstacles", [])
        if not obs:
            return

        pts = np.array([[o.get("x_m", 0.0), o.get("y_m", 0.0), o.get("z_m", 0.0)] for o in obs], dtype=float)
        alphas = np.array([float(o.get("opacity", 0.0)) for o in obs], dtype=float)
        alphas = np.clip(alphas, 0.0, 1.0)

        colors = np.zeros((pts.shape[0], 4), dtype=float)
        colors[:, 0] = 0.53
        colors[:, 1] = 0.20
        colors[:, 2] = 0.82
        colors[:, 3] = alphas

        self._obs_scatter = self.ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=85,
            c=colors,
            marker="o",
            depthshade=False,
        )

    def _draw_breadcrumb(self, idx: int) -> None:
        t_now = float(self.times[idx])
        lo = t_now - 5.0
        mask = (self.times >= lo) & (self.times <= t_now)
        if not np.any(mask):
            self.breadcrumb_line.set_data([], [])
            self.breadcrumb_line.set_3d_properties([])
            return

        pts = self.tool_points[np.where(mask)[0], :]
        self.breadcrumb_line.set_data(pts[:, 0], pts[:, 1])
        self.breadcrumb_line.set_3d_properties(pts[:, 2])

    @classmethod
    def _record_in_replan_trajectory(cls, rec: dict) -> bool:
        mode_names = {
            str(rec.get("mode", "")),
            str(rec.get("planner_mode", "")),
            str(rec.get("planner_debug", {}).get("mode", "")),
        }
        if any(name in cls._REPLAN_TRAJECTORY_MODES for name in mode_names):
            return True
        if bool(rec.get("future_plan", [])):
            return True
        return str(rec.get("obstacle", {}).get("response", "")) == "replan"

    @classmethod
    def _build_replan_history_points(cls, records: list[dict], tool_points: np.ndarray) -> np.ndarray:
        pts = np.asarray(tool_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(records) <= 0:
            return np.zeros((0, 3), dtype=float)

        n = min(len(records), int(pts.shape[0]))
        out = np.full((n, 3), np.nan, dtype=float)
        for idx in range(n):
            if cls._record_in_replan_trajectory(records[idx]):
                out[idx, :] = pts[idx, :]
        return out

    def _draw_replan_history(self, idx: int) -> None:
        if self._replan_history_points.size <= 0 or idx < 0:
            self.replanned_line.set_data([], [])
            self.replanned_line.set_3d_properties([])
            return

        arr = self._replan_history_points[: min(idx + 1, self._replan_history_points.shape[0]), :]
        finite = np.all(np.isfinite(arr), axis=1)
        if not np.any(finite):
            self.replanned_line.set_data([], [])
            self.replanned_line.set_3d_properties([])
            return

        pts = arr[finite, :]
        self.replanned_line.set_data(pts[:, 0], pts[:, 1])
        self.replanned_line.set_3d_properties(pts[:, 2])

    def _draw_future_plan(self, rec: dict) -> None:
        if self._planned_scatter is not None:
            self._planned_scatter.remove()
            self._planned_scatter = None

        plan = rec.get("future_plan", [])
        if not plan:
            self.planned_line.set_data([], [])
            self.planned_line.set_3d_properties([])
            return

        pts = []
        for p in plan:
            try:
                xyz = self._future_plan_point_m(p)
                if xyz is None:
                    continue
                xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
                pts.append(xyz)
            except Exception:
                continue

        if not pts:
            self.planned_line.set_data([], [])
            self.planned_line.set_3d_properties([])
            return

        arr = np.asarray(pts, dtype=float)
        self.planned_line.set_data(arr[:, 0], arr[:, 1])
        self.planned_line.set_3d_properties(arr[:, 2])

    def _future_plan_point_m(self, point: dict) -> np.ndarray | None:
        joints = point.get("joints_deg", None)
        if joints is not None and len(joints) >= 5:
            try:
                tf = self.kin.joint_transform([float(x) for x in joints], "wrist_roll")
                xyz = np.asarray(tf[:3, 3], dtype=float).reshape(3)
                if np.all(np.isfinite(xyz)):
                    return xyz
            except Exception:
                pass
        e = point.get("eef_m", None)
        if e is not None and len(e) >= 3:
            xyz = np.asarray([float(e[0]), float(e[1]), float(e[2])], dtype=float)
            if np.all(np.isfinite(xyz)):
                return xyz
        return None
    def _on_key(self, event) -> None:
        if event.key == " ":
            self._paused = not self._paused
        elif event.key == "right":
            self._paused = True
            self._idx = min(len(self.records) - 1, self._idx + 1)
            self._draw_frame(self._idx)
            self.fig.canvas.draw_idle()
        elif event.key == "left":
            self._paused = True
            self._idx = max(0, self._idx - 1)
            self._draw_frame(self._idx)
            self.fig.canvas.draw_idle()
        elif event.key and event.key.lower() == "s":
            self.save_mp4()

    def save_mp4(self, out_path: Path | None = None) -> Path | None:
        if out_path is None:
            w_px, h_px = self._resolution_px
            out_path = self.data.session_dir / f"session_playback_{w_px}x{h_px}.mp4"
        print(f"[INFO] Saving MP4 -> {out_path}")

        try:
            import imageio
            import imageio_ffmpeg  # noqa: F401
        except Exception as exc:
            if animation.writers.is_available("ffmpeg"):
                writer = animation.FFMpegWriter(fps=self._fps, bitrate=3000)
                try:
                    self.anim.save(str(out_path), writer=writer, dpi=100)
                    print("[INFO] MP4 export complete")
                    try:
                        plt.close(self.fig)
                    except Exception:
                        pass
                    return out_path
                except Exception as exc2:
                    print(f"[ERR] Matplotlib ffmpeg export failed: {exc2}")
            print(
                "[ERR] MP4 export requires ffmpeg support. Install with: "
                "python -m pip install imageio-ffmpeg"
            )
            print(f"[ERR] Missing backend detail: {exc}")
            return None

        old_idx = int(self._idx)
        old_paused = bool(self._paused)
        saved_ok = False
        try:
            writer = imageio.get_writer(str(out_path), fps=self._fps, codec="libx264", macro_block_size=None)
            total = max(1, len(self.records))
            last_pct = -1
            for idx in range(total):
                self._draw_frame(idx)
                self.fig.canvas.draw()
                w, h = self.fig.canvas.get_width_height()
                frame = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
                writer.append_data(frame)
                pct = int(round(100.0 * float(idx + 1) / float(total)))
                if pct != last_pct and (pct % 2 == 0 or pct == 100):
                    filled = max(0, min(30, int(round((pct / 100.0) * 30.0))))
                    bar = "#" * filled + "-" * (30 - filled)
                    print(f"\r[INFO] Saving MP4 [{bar}] {pct:3d}%", end="", flush=True)
                    last_pct = pct
            writer.close()
            print("\r[INFO] Saving MP4 [##############################] 100%")
            print("[INFO] MP4 export complete")
            saved_ok = True
            try:
                plt.close(self.fig)
            except Exception:
                pass
            return out_path
        except Exception as exc:
            print(f"[ERR] MP4 export failed: {exc}")
            return None
        finally:
            self._idx = old_idx
            self._paused = old_paused
            if not saved_ok:
                self._draw_frame(self._idx)
                self.fig.canvas.draw_idle()

    def save_mp4_720p(self) -> Path | None:
        return self.save_mp4(self.data.session_dir / "session_playback_720p.mp4")


def _load_session(session_path: Path) -> SessionData:
    if session_path.is_dir():
        session_dir = session_path
        data_path = session_dir / "session.jsonl"
    else:
        data_path = session_path
        session_dir = session_path.parent

    if not data_path.exists():
        raise FileNotFoundError(f"session file not found: {data_path}")

    records: list[dict] = []
    with data_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise RuntimeError("session is empty")

    t0 = float(records[0].get("host_monotonic", 0.0))
    times = np.array([float(r.get("host_monotonic", t0)) - t0 for r in records], dtype=float)

    meta_path = session_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    return SessionData(session_dir=session_dir, records=records, times_s=times, meta=meta)


def _resolution_presets() -> dict[str, tuple[int, int]]:
    return {
        "1280x720 (720p)": (1280, 720),
        "1600x900": (1600, 900),
        "1920x1080 (1080p)": (1920, 1080),
    }


def _view_presets() -> dict[str, tuple[float, float]]:
    return {
        "Isometric": (24.0, -55.0),
        "Front": (15.0, -90.0),
        "Side": (15.0, 0.0),
        "Top": (90.0, -90.0),
        "Interactive": (24.0, -55.0),
    }


def _infer_session_channel_count(data: SessionData) -> int:
    supported = list(SENSOR_CONFIG.get("supported_channels", [0, 1, 2, 3]))
    default_count = 2 if len(supported) >= 2 else max(1, len(supported))

    meta_count = int(data.meta.get("tof_channels_enabled", 0) or 0)
    if meta_count > 0:
        return meta_count

    max_seen = -1
    for rec in data.records:
        for frame in rec.get("tof_frames", []):
            try:
                max_seen = max(max_seen, int(frame.get("channel", -1)))
            except Exception:
                pass
        for ch_debug in rec.get("projection_debug", {}).get("channels", []):
            try:
                max_seen = max(max_seen, int(ch_debug.get("channel", -1)))
            except Exception:
                pass
    if max_seen >= 0:
        return max_seen + 1
    return default_count


def _session_layout_label(session_dir: Path) -> str:
    meta_path = session_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        meta = {}
    n_cfg = int(meta.get("tof_channels_enabled", 0) or 0)
    if n_cfg <= 0:
        return "Unknown layout"
    return "Hyperion 4-ToF" if n_cfg > 2 else "Dual 2-ToF"


def _launch_selector(base_dir: Path) -> LaunchConfig:
    sessions = sorted([p for p in base_dir.glob("session_*") if p.is_dir()])
    if not sessions:
        raise RuntimeError(f"no sessions found in {base_dir}")

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise RuntimeError(f"tkinter unavailable: {exc}")

    res_presets = _resolution_presets()
    view_presets = _view_presets()
    chosen = {
        "path": sessions[-1],
        "action": "MATPLOT View",
        "resolution": next(iter(res_presets.keys())),
        "view": "Isometric",
    }

    root = tk.Tk()
    root.title("Session Plotter Launcher")
    root.geometry("620x380")
    root.minsize(620, 380)

    tk.Label(root, text="Select a recording session:").pack(pady=(10, 6))

    names = [p.name for p in sessions]
    combo = ttk.Combobox(root, values=names, state="readonly", width=52)
    combo.current(len(names) - 1)
    combo.pack()

    info_var = tk.StringVar(value=f"Detected layout: {_session_layout_label(sessions[-1])}")
    info_label = tk.Label(root, textvariable=info_var, anchor="w", justify="left")
    info_label.pack(pady=(8, 4))

    tk.Label(root, text="What would you like to do?").pack(pady=(12, 4))
    action_combo = ttk.Combobox(root, values=["MATPLOT View", "MP4"], state="readonly", width=24)
    action_combo.current(0)
    action_combo.pack()

    tk.Label(root, text="MP4 resolution:").pack(pady=(12, 4))
    res_combo = ttk.Combobox(root, values=list(res_presets.keys()), state="readonly", width=24)
    res_combo.current(0)
    res_combo.pack()

    tk.Label(root, text="View angle:").pack(pady=(12, 4))
    view_combo = ttk.Combobox(root, values=list(view_presets.keys()), state="readonly", width=24)
    view_combo.current(0)
    view_combo.pack()

    def _on_session_change(_event=None) -> None:
        idx = combo.current()
        if idx < 0:
            idx = len(sessions) - 1
        info_var.set(f"Detected layout: {_session_layout_label(sessions[idx])}")

    combo.bind("<<ComboboxSelected>>", _on_session_change)

    def _accept() -> None:
        idx = combo.current()
        if idx < 0:
            idx = len(sessions) - 1
        chosen["path"] = sessions[idx]
        chosen["action"] = action_combo.get() or "MATPLOT View"
        chosen["resolution"] = res_combo.get() or next(iter(res_presets.keys()))
        chosen["view"] = view_combo.get() or "Isometric"
        root.destroy()

    ttk.Button(root, text="Continue", command=_accept).pack(pady=(18, 18))
    root.mainloop()

    resolution = res_presets.get(chosen["resolution"], (1280, 720))
    elev, azim = view_presets.get(chosen["view"], (24.0, -55.0))
    return LaunchConfig(
        session_path=Path(chosen["path"]),
        action=str(chosen["action"]),
        resolution_px=resolution,
        view_name=str(chosen["view"]),
        view_elev=float(elev),
        view_azim=float(azim),
    )


def _capture_interactive_view(data: SessionData, resolution_px: tuple[int, int]) -> tuple[float, float]:
    print("[INFO] Interactive view selection: rotate the dummy figure, then close it to continue.")
    picker = SessionPlotter(data, resolution_px=resolution_px, view_elev=24.0, view_azim=-55.0)
    picker._draw_frame(0)
    plt.show()
    elev = float(picker.ax.elev)
    azim = float(picker.ax.azim)
    try:
        plt.close(picker.fig)
    except Exception:
        pass
    return elev, azim


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playback Braccio recording sessions")
    parser.add_argument(
        "--session",
        default=None,
        help="Session directory or session.jsonl path. If omitted, a dropdown selector is shown.",
    )
    parser.add_argument(
        "--sessions-dir",
        default="logs/sessions",
        help="Directory that stores session_* folders (default: logs/sessions)",
    )
    parser.add_argument(
        "--export-mp4",
        action="store_true",
        help="Export MP4 immediately and exit (no interactive window).",
    )
    parser.add_argument(
        "--resolution",
        default="1280x720",
        help="Export/view resolution, e.g. 1280x720 or 1920x1080.",
    )
    parser.add_argument(
        "--view-preset",
        default="Isometric",
        help="View preset: Isometric, Front, Side, Top, or Interactive.",
    )
    parser.add_argument("--view-elev", type=float, default=None, help="Override view elevation angle.")
    parser.add_argument("--view-azim", type=float, default=None, help="Override view azimuth angle.")
    return parser.parse_args()


def _parse_resolution(text: str) -> tuple[int, int]:
    raw = str(text).strip().lower().replace(" ", "")
    if "x" not in raw:
        raise ValueError(f"invalid resolution: {text}")
    w_txt, h_txt = raw.split("x", 1)
    return max(640, int(w_txt)), max(360, int(h_txt))


def main() -> None:
    args = _parse_args()
    sessions_dir = Path(args.sessions_dir)

    if args.session:
        session_path = Path(args.session)
        resolution_px = _parse_resolution(args.resolution)
        if args.view_elev is not None and args.view_azim is not None:
            view_elev = float(args.view_elev)
            view_azim = float(args.view_azim)
        else:
            preset = _view_presets().get(str(args.view_preset), (24.0, -55.0))
            view_elev, view_azim = preset
    else:
        launch = _launch_selector(sessions_dir)
        session_path = launch.session_path
        resolution_px = launch.resolution_px
        view_elev = launch.view_elev
        view_azim = launch.view_azim
        args.export_mp4 = args.export_mp4 or (launch.action == "MP4")

    data = _load_session(session_path)

    if str(args.view_preset) == "Interactive" and args.session:
        view_elev, view_azim = _capture_interactive_view(data, resolution_px)
    elif not args.session:
        if 'launch' in locals() and launch.view_name == "Interactive":
            view_elev, view_azim = _capture_interactive_view(data, resolution_px)

    player = SessionPlotter(data, resolution_px=resolution_px, view_elev=view_elev, view_azim=view_azim)

    if args.export_mp4:
        player.save_mp4()
        return

    print("Controls: Space=pause/resume, Left/Right=step, S=save MP4")
    plt.show()


if __name__ == "__main__":
    main()









