#!/usr/bin/env python3
"""Visualize the physical ToF mount layout on the Braccio wrist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from braccio_ctrl.constants import (
    HOME_POS,
    PROFILE_SWEEP_GRIPPER_DEG,
    PROFILE_SWEEP_R_MM,
    PROFILE_SWEEP_WRIST_OFFSET_DEG,
    PROFILE_SWEEP_WRIST_ROT_DEG,
    PROFILE_SWEEP_Z_MM,
)
from braccio_ctrl.ik_solver import apply_wrist_offset, solve_ik
from planning.sensor_config import SENSOR_CONFIG, sensor_urdf_path
from session_plotter import SessionData, SessionPlotter

HOME_JOINTS = [float(v) for v in HOME_POS]


def _sweep_joints(theta_deg: float) -> list[float]:
    ik = solve_ik(float(theta_deg), float(PROFILE_SWEEP_R_MM), float(PROFILE_SWEEP_Z_MM))
    return [
        float(ik.base),
        float(ik.shoulder),
        float(ik.elbow),
        float(apply_wrist_offset(ik.wrist_vert, float(PROFILE_SWEEP_WRIST_OFFSET_DEG))),
        float(PROFILE_SWEEP_WRIST_ROT_DEG),
        float(PROFILE_SWEEP_GRIPPER_DEG),
    ]


def _dummy_record(channels: int, joints_deg: list[float], mode: str) -> dict:
    return {
        "host_monotonic": 0.0,
        "wall_time": 0.0,
        "mode": mode,
        "planner_active": False,
        "planner_model": "simple_polar_pockets_committed",
        "obstacle": {"response": "clear", "source": "", "distance_mm": -1.0},
        "planner_debug": {"debug": {"corridor": "layout", "severity": 0.0, "min_clearance_mm": 0.0, "speed_scale": 1.0}},
        "cmd_velocity": {"theta_deg_s": 0.0, "r_mm_s": 0.0, "z_mm_s": 0.0},
        "joints_deg": list(joints_deg),
        "future_plan": [],
        "obstacles": [],
        "projection_debug": {"channels": []},
        "imu": {"online": False, "calibrated": False},
        "tof_frames": [{"channel": ch} for ch in range(channels)],
    }


def _build_data(channels: int, joints_deg: list[float], pose_name: str) -> SessionData:
    repo_root = ROOT.parent
    meta = {
        "tof_channels_enabled": int(channels),
        "hyperion_mode": bool(channels > 2),
        "sensor_layout_mode": "hyperion_4tof" if channels > 2 else "dual_2tof",
        "threshold_mm": 200.0,
        "planner_model": "simple_polar_pockets_committed",
        "planner_clearance_mm": 10.0,
        "urdf_path": str(sensor_urdf_path()),
    }
    return SessionData(
        session_dir=repo_root / "logs" / "sessions",
        records=[_dummy_record(channels, joints_deg, f"layout_preview_{pose_name}")],
        times_s=np.array([0.0], dtype=float),
        meta=meta,
    )


def _layout_note(channels: int, pose_name: str, theta_deg: float) -> str:
    if channels > 2:
        return (
            f"Pose: {pose_name}"
            + (f" (theta={theta_deg:.0f})\n" if pose_name == "sweep" else "\n")
            + "Reference: URDF-backed controller/plotter geometry only\n"
            "CH2 = north/top, CH0 = west/left, CH1 = east/right, CH3 = south/bottom\n"
            "HOME top-down reference: CH0 -> -Y, CH1 -> +Y, CH2 -> -X, CH3 -> +X\n"
            "No viewer-only remap is applied"
        )
    return (
        f"Pose: {pose_name}"
        + (f" (theta={theta_deg:.0f})\n" if pose_name == "sweep" else "\n")
        + "Reference: URDF-backed controller/plotter geometry only\n"
        "CH0 = west/left, CH1 = east/right\n"
        "Boresight mapping: CH0 -> -Y, CH1 -> +Y"
    )


def _decorate_layout(plotter: SessionPlotter, channels: int, joints: list[float], pose_name: str, theta_deg: float) -> None:
    plotter._paused = True
    try:
        plotter.anim.event_source.stop()
        plotter.anim._draw_was_started = True
    except Exception:
        pass
    plotter._draw_frame(0)

    plotter.metrics_ax.set_visible(False)
    plotter.joints_ax.set_visible(False)
    plotter.ax.set_position([0.04, 0.08, 0.92, 0.86])
    plotter._status_text.set_visible(False)

    tf_wp = plotter.kin.link_transform(joints, "link4")
    wp_origin = tf_wp[:3, 3].copy()
    wp_rot = tf_wp[:3, :3].copy()

    for ch in range(channels):
        cfg = SENSOR_CONFIG["channels"][ch]
        origin = wp_origin + (wp_rot @ np.asarray(cfg["origin_m"], dtype=float))
        color = plotter._channel_color(ch)
        plotter.ax.text(
            float(origin[0]),
            float(origin[1]),
            float(origin[2]) + 0.015,
            f"CH{ch}\n{cfg.get('label', '')}",
            color=color,
            fontsize=9,
            ha="center",
            va="bottom",
            bbox={"facecolor": (0.05, 0.08, 0.12, 0.68), "edgecolor": color, "boxstyle": "round,pad=0.25"},
        )

    title_prefix = "Hyperion 4-ToF Wrist Layout" if channels > 2 else "Dual-ToF Wrist Layout"
    title = title_prefix + (f" | {pose_name} theta={theta_deg:.0f}" if pose_name == "sweep" else " | home")
    plotter.ax.set_title(title)
    plotter.fig.text(
        0.04,
        0.98,
        _layout_note(channels, pose_name, theta_deg),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
        color="#E5E7EB",
        bbox={"facecolor": (0.05, 0.08, 0.12, 0.72), "edgecolor": "#1F2937", "boxstyle": "round,pad=0.35"},
    )
    handles = []
    for ch in range(channels):
        color = plotter._channel_color(ch)
        label = str(SENSOR_CONFIG["channels"][ch].get("label", f"ch{ch}"))
        handles.append(Line2D([0], [0], color=color, lw=3.0, label=f"CH{ch}  {label}"))
    legend = plotter.fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=True)
    legend.get_frame().set_facecolor((0.05, 0.08, 0.12, 0.72))
    legend.get_frame().set_edgecolor("#1F2937")
    for txt in legend.get_texts():
        txt.set_color("#E5E7EB")


def _show_layout(channels: int, save_dir: Path | None, pose_name: str, theta_deg: float, top_down: bool) -> None:
    joints = list(HOME_JOINTS) if pose_name == "home" else _sweep_joints(theta_deg)
    data = _build_data(channels, joints, pose_name)
    elev = 90.0 if top_down else 26.0
    azim = -90.0 if top_down else -55.0
    plotter = SessionPlotter(data, resolution_px=(1440, 900), view_elev=elev, view_azim=azim)
    _decorate_layout(plotter, channels, joints, pose_name, theta_deg)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "hyperion_4tof" if channels > 2 else "dual_2tof"
        pose_suffix = f"{pose_name}_theta{int(round(theta_deg))}" if pose_name == "sweep" else pose_name
        view_suffix = "topdown" if top_down else "iso"
        out_path = save_dir / f"{suffix}_{pose_suffix}_{view_suffix}.png"
        plotter.fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"[INFO] Saved layout -> {out_path}")
        plt.close(plotter.fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Braccio ToF mount layouts using the Tinkerkit model")
    parser.add_argument("--mode", choices=("2", "4", "both"), default="both", help="Layout to display")
    parser.add_argument("--save-dir", default=None, help="Optional directory for saving PNG renders instead of showing them")
    parser.add_argument("--pose", choices=("home", "sweep"), default="home", help="Robot pose to render")
    parser.add_argument("--theta", type=float, default=90.0, help="Sweep theta to use when --pose sweep")
    parser.add_argument("--top-down", action="store_true", help="Render with a top-down verification view")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    save_dir = Path(args.save_dir).resolve() if args.save_dir else None

    modes = [2, 4] if args.mode == "both" else [int(args.mode)]
    for channels in modes:
        _show_layout(channels, save_dir, str(args.pose), float(args.theta), bool(args.top_down))

    if save_dir is None:
        plt.show()


if __name__ == "__main__":
    main()
