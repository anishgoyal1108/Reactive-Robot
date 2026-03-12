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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planning.sensor_config import SENSOR_CONFIG
from planning.urdf_kinematics import BraccioURDFKinematics


@dataclass
class SessionData:
    session_dir: Path
    records: list[dict]
    times_s: np.ndarray
    meta: dict


class SessionPlotter:
    def __init__(self, data: SessionData):
        self.data = data
        self.records = data.records
        self.times = data.times_s

        urdf_path = Path(data.meta.get("urdf_path", ""))
        if not urdf_path.exists():
            urdf_path = Path(__file__).resolve().parents[1] / "Tinkerkit_model" / "tinkerkit.urdf"
        self.kin = BraccioURDFKinematics(urdf_path)

        supported = list(SENSOR_CONFIG.get("supported_channels", [0, 1]))
        n_cfg = int(data.meta.get("tof_channels_enabled", len(supported)))
        n_cfg = max(1, min(n_cfg, len(supported)))
        self._tof_channels = supported[:n_cfg]

        threshold_mm = float(data.meta.get("threshold_mm", 300.0))
        self._tof_cone_range_m = max(0.12, min(1.0, 1.15 * (threshold_mm / 1000.0)))

        self._channel_colors = {
            0: "#3B82F6",  # west / +Y at home
            1: "#F97316",  # east / -Y at home
            2: "#22C55E",  # reserved
            3: "#A855F7",  # reserved
        }

        imu_rot = np.asarray(data.meta.get("imu_rot_wrist_from_imu", []), dtype=float)
        if imu_rot.shape == (3, 3):
            self._imu_rot_wrist_from_imu = imu_rot
        else:
            self._imu_rot_wrist_from_imu = None
        self._imu_calibrated = bool(data.meta.get("imu_calibrated", False)) and (self._imu_rot_wrist_from_imu is not None)
        self._imu_fuse_weight = 0.8
        self._imu_max_tilt_deg = 35.0

        self._idx = 0
        self._paused = False
        self._obs_scatter = None
        self.tool_points = self._precompute_tool_points()

        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        self._setup_axes()

        self.robot_line, = self.ax.plot([], [], [], "-o", color="#ff8f2a", linewidth=2.0, markersize=4)
        self.tool_marker, = self.ax.plot([], [], [], "o", color="#00c36b", markersize=6)
        self.breadcrumb_line, = self.ax.plot([], [], [], color="#00c36b", linewidth=2.0)
        self.planned_line, = self.ax.plot([], [], [], "--", color="#7DD3FC", linewidth=1.8)
        self._planned_scatter = None

        self._draw_feasible_cloud()
        self._init_tof_coverage_artists()

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
        cloud = self._load_feasible_cloud()
        if cloud is not None and cloud.size > 0:
            lo = np.min(cloud, axis=0)
            hi = np.max(cloud, axis=0)
        else:
            pts = self.tool_points
            lo = np.min(pts, axis=0) if pts.size > 0 else np.array([-0.2, -0.2, -0.2])
            hi = np.max(pts, axis=0) if pts.size > 0 else np.array([0.2, 0.2, 0.2])

        pad = np.array([0.08, 0.08, 0.08], dtype=float)
        lo = lo - pad
        hi = hi + pad

        self.ax.set_xlim(float(lo[0]), float(hi[0]))
        self.ax.set_ylim(float(lo[1]), float(hi[1]))
        self.ax.set_zlim(float(lo[2]), float(hi[2]))
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title("Braccio Session Playback")

    def _load_feasible_cloud(self) -> np.ndarray | None:
        cloud_path = Path(self.data.meta.get("feasible_cloud_path", ""))
        if not cloud_path.exists():
            cloud_path = Path(__file__).resolve().parent / "sensing" / "feasible_regions" / "workspace_cloud.npy"
        try:
            if cloud_path.exists():
                return np.load(cloud_path)
        except Exception:
            return None
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

    def _draw_feasible_cloud(self) -> None:
        cloud = self._load_feasible_cloud()
        if cloud is None or cloud.size == 0:
            return

        n = cloud.shape[0]
        keep = min(n, 14000)
        if n > keep:
            idx = np.linspace(0, n - 1, keep, dtype=int)
            cloud = cloud[idx]

        self.ax.scatter(
            cloud[:, 0],
            cloud[:, 1],
            cloud[:, 2],
            s=2,
            c="#9ad5ff",
            alpha=0.05,
            linewidths=0,
        )

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
            tf_wp = self.kin.joint_transform(joints, "wrist_pitch")
            wp_origin = tf_wp[:3, 3].copy()
            wp_rot = tf_wp[:3, :3].copy()
        except Exception:
            for ch in self._tof_channels:
                self._tof_cones[ch].set_verts([])
                self._tof_markers[ch].set_data([], [])
                self._tof_markers[ch].set_3d_properties([])
            return

        if self._imu_calibrated and rec is not None and self._imu_rot_wrist_from_imu is not None:
            imu = rec.get("imu", {})
            if bool(imu.get("online", False)):
                a = np.array([
                    float(imu.get("ax_g", 0.0)),
                    float(imu.get("ay_g", 0.0)),
                    float(imu.get("az_g", 0.0)),
                ], dtype=float)
                a = self._normalize(a)
                if float(np.linalg.norm(a)) > 1e-6:
                    g_meas_wrist = self._normalize(self._imu_rot_wrist_from_imu @ a)
                    g_fk_wrist = self._normalize(wp_rot.T @ np.array([0.0, 0.0, -1.0], dtype=float))
                    dot = float(np.dot(g_meas_wrist, g_fk_wrist))
                    dot = max(-1.0, min(1.0, dot))
                    axis = np.cross(g_meas_wrist, g_fk_wrist)
                    axis_n = float(np.linalg.norm(axis))
                    if axis_n > 1e-9:
                        angle = math.atan2(axis_n, dot)
                        angle = min(math.radians(self._imu_max_tilt_deg), angle * float(self._imu_fuse_weight))
                        if angle > 1e-6:
                            wp_rot = wp_rot @ self._rot_axis_angle(axis, angle)

        for ch in self._tof_channels:
            cfg = SENSOR_CONFIG.get("channels", {}).get(ch)
            if cfg is None:
                self._tof_cones[ch].set_verts([])
                self._tof_markers[ch].set_data([], [])
                self._tof_markers[ch].set_3d_properties([])
                continue

            local_origin = np.asarray(cfg.get("origin_m", [0.0, 0.0, 0.0]), dtype=float)
            axis_local = np.asarray(cfg.get("axis_dir", [1.0, 0.0, 0.0]), dtype=float)
            up_local = np.asarray(cfg.get("up_hint", [0.0, 0.0, 1.0]), dtype=float)

            origin = wp_origin + (wp_rot @ local_origin)
            axis = wp_rot @ self._normalize(axis_local)
            up_hint = wp_rot @ self._normalize(up_local)
            axis, right, up = self._sensor_basis(axis, up_hint)

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

        tool = self.tool_points[idx]
        self.tool_marker.set_data([tool[0]], [tool[1]])
        self.tool_marker.set_3d_properties([tool[2]])

        self._update_tof_coverage(joints, rec)
        self._draw_obstacles(rec)
        self._draw_breadcrumb(idx)
        self._draw_future_plan(rec)

        wall_t = rec.get("wall_time", 0.0)
        mode = rec.get("mode", "")
        opt = "ON" if rec.get("optimizer_active") else "off"
        self.ax.set_title(f"Session Playback  |  t={wall_t:.2f}  |  mode={mode}  |  opt={opt}")

    def _draw_obstacles(self, rec: dict) -> None:
        if self._obs_scatter is not None:
            self._obs_scatter.remove()
            self._obs_scatter = None

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
            s=120,
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
        way = []
        for idx, p in enumerate(plan):
            try:
                e = p.get("eef_m", None)
                if e is None or len(e) < 3:
                    continue
                xyz = [float(e[0]), float(e[1]), float(e[2])]
                pts.append(xyz)
                if bool(p.get("waypoint", False)) or (idx % 3 == 0):
                    way.append(xyz)
            except Exception:
                continue

        if not pts:
            self.planned_line.set_data([], [])
            self.planned_line.set_3d_properties([])
            return

        arr = np.asarray(pts, dtype=float)
        self.planned_line.set_data(arr[:, 0], arr[:, 1])
        self.planned_line.set_3d_properties(arr[:, 2])

        if way:
            warr = np.asarray(way, dtype=float)
            self._planned_scatter = self.ax.scatter(
                warr[:, 0],
                warr[:, 1],
                warr[:, 2],
                s=26,
                c="#7DD3FC",
                alpha=0.65,
                marker="o",
                depthshade=False,
            )
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
            self.save_mp4_720p()

    def save_mp4_720p(self) -> Path | None:
        out_path = self.data.session_dir / "session_playback_720p.mp4"
        writer = animation.FFMpegWriter(fps=self._fps, bitrate=3000)
        try:
            print(f"[INFO] Saving 720p MP4 -> {out_path}")
            self.anim.save(str(out_path), writer=writer, dpi=100)
            print("[INFO] MP4 export complete")
            return out_path
        except Exception as exc:
            print(f"[ERR] MP4 export failed: {exc}")
            return None


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


def _choose_session_via_dropdown(base_dir: Path) -> Path:
    sessions = sorted([p for p in base_dir.glob("session_*") if p.is_dir()])
    if not sessions:
        raise RuntimeError(f"no sessions found in {base_dir}")

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise RuntimeError(f"tkinter unavailable: {exc}")

    chosen = {"path": sessions[-1]}

    root = tk.Tk()
    root.title("Session Selector")
    root.geometry("520x120")

    tk.Label(root, text="Select a recording session:").pack(pady=8)

    names = [p.name for p in sessions]
    combo = ttk.Combobox(root, values=names, state="readonly", width=52)
    combo.current(len(names) - 1)
    combo.pack()

    def _open_selected() -> None:
        idx = combo.current()
        if idx < 0:
            idx = len(sessions) - 1
        chosen["path"] = sessions[idx]
        root.destroy()

    ttk.Button(root, text="Open Session", command=_open_selected).pack(pady=10)
    root.mainloop()
    return chosen["path"]


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
        help="Export 720p MP4 immediately and exit (no interactive window).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sessions_dir = Path(args.sessions_dir)

    if args.session:
        session_path = Path(args.session)
    else:
        session_path = _choose_session_via_dropdown(sessions_dir)

    data = _load_session(session_path)
    player = SessionPlotter(data)

    if args.export_mp4:
        player.save_mp4_720p()
        return

    print("Controls: Space=pause/resume, Left/Right=step, S=save 720p MP4")
    plt.show()


if __name__ == "__main__":
    main()









