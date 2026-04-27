#!/usr/bin/env python3
"""Validate ToF boresights in world frame at home or sweep posture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "braccio_main_runner"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from braccio_ctrl.constants import (  # type: ignore
    HOME_POS,
    PROFILE_SWEEP_GRIPPER_DEG,
    PROFILE_SWEEP_R_MM,
    PROFILE_SWEEP_WRIST_OFFSET_DEG,
    PROFILE_SWEEP_WRIST_ROT_DEG,
    PROFILE_SWEEP_Z_MM,
)
from braccio_ctrl.ik_solver import apply_wrist_offset, solve_ik  # type: ignore
from planning.sensor_config import SENSOR_CONFIG, sensor_urdf_path  # type: ignore
from planning.urdf_kinematics import BraccioURDFKinematics  # type: ignore


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return vec / n


def _pose_joints(theta_deg: float) -> list[float]:
    ik = solve_ik(float(theta_deg), float(PROFILE_SWEEP_R_MM), float(PROFILE_SWEEP_Z_MM))
    return [
        float(ik.base),
        float(ik.shoulder),
        float(ik.elbow),
        float(apply_wrist_offset(ik.wrist_vert, float(PROFILE_SWEEP_WRIST_OFFSET_DEG))),
        float(PROFILE_SWEEP_WRIST_ROT_DEG),
        float(PROFILE_SWEEP_GRIPPER_DEG),
    ]


def _home_joints() -> list[float]:
    return [float(v) for v in HOME_POS]


def _channel_world_vectors(kin: BraccioURDFKinematics, joints: list[float]) -> list[dict]:
    tf = kin.link_transform(joints, "link4")
    origin_link4 = tf[:3, 3].copy()
    rot_link4 = tf[:3, :3].copy()
    channels: list[dict] = []
    for ch in SENSOR_CONFIG.get("supported_channels", [0, 1, 2, 3]):
        cfg = SENSOR_CONFIG["channels"][int(ch)]
        local_origin = np.asarray(cfg.get("origin_m", [0.0, 0.0, 0.0]), dtype=float)
        local_axis = _normalize(np.asarray(cfg.get("axis_dir", [0.0, 0.0, 1.0]), dtype=float))
        world_origin = origin_link4 + (rot_link4 @ local_origin)
        world_axis = _normalize(rot_link4 @ local_axis)
        channels.append(
            {
                "channel": int(ch),
                "label": str(cfg.get("label", f"ch{ch}")),
                "origin_m": world_origin,
                "axis_world": world_axis,
                "axis_link4": local_axis,
            }
        )
    return channels


def _print_report(pose_name: str, theta_deg: float, joints: list[float], channels: list[dict]) -> None:
    print(f"ToF geometry validation at pose={pose_name}")
    if pose_name == "sweep":
        print(f"theta={theta_deg:.1f} deg  r={PROFILE_SWEEP_R_MM:.1f} mm  z={PROFILE_SWEEP_Z_MM:.1f} mm")
    else:
        print("HOME pose with theta=90 and arm vertical")
    print("joints_deg =", [round(v, 3) for v in joints])
    print()
    print("Expected HOME world boresights:")
    print("  CH0 -> [ 0, -1,  0]")
    print("  CH1 -> [ 0, +1,  0]")
    print("  CH2 -> [-1,  0,  0]")
    print("  CH3 -> [+1,  0,  0]")
    print()
    for item in channels:
        origin = item["origin_m"]
        axis = item["axis_world"]
        axis_local = item["axis_link4"]
        print(
            f"CH{item['channel']} ({item['label']}): "
            f"link4_axis={np.round(axis_local, 4).tolist()}  "
            f"world_axis={np.round(axis, 4).tolist()}  "
            f"origin_m={np.round(origin, 4).tolist()}"
        )


def _plot(pose_name: str, theta_deg: float, channels: list[dict], save_path: Path | None = None) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(
        f"ToF boresights at {pose_name} pose"
        + (f" (theta={theta_deg:.1f} deg)" if pose_name == "sweep" else "")
    )
    colors = {0: "#38BDF8", 1: "#F97316", 2: "#22C55E", 3: "#A855F7"}
    for item in channels:
        ch = int(item["channel"])
        origin = np.asarray(item["origin_m"], dtype=float)
        axis = np.asarray(item["axis_world"], dtype=float)
        end = origin + (0.06 * axis)
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=colors.get(ch, "#94A3B8"),
            linewidth=3.0,
        )
        ax.scatter([origin[0]], [origin[1]], [origin[2]], color=colors.get(ch, "#94A3B8"), s=48)
        ax.text(end[0], end[1], end[2], f"CH{ch}", color=colors.get(ch, "#94A3B8"))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect((1.0, 1.0, 0.8))
    ax.view_init(elev=90.0, azim=-90.0)
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"[INFO] Saved boresight capture -> {save_path}")
        plt.close(fig)
        return
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ToF boresights in world frame at home or sweep posture")
    parser.add_argument("--pose", choices=("home", "sweep"), default="home", help="Pose to validate")
    parser.add_argument("--theta", type=float, default=90.0, help="Sweep theta in degrees (default: 90)")
    parser.add_argument("--plot", action="store_true", help="Show a simple 3D boresight plot")
    parser.add_argument("--save", default=None, help="Optional PNG path for a saved top-down boresight capture")
    args = parser.parse_args()

    urdf_path = sensor_urdf_path()
    kin = BraccioURDFKinematics(urdf_path)
    joints = _home_joints() if args.pose == "home" else _pose_joints(float(args.theta))
    channels = _channel_world_vectors(kin, joints)
    _print_report(str(args.pose), float(args.theta), joints, channels)
    if args.plot:
        _plot(str(args.pose), float(args.theta), channels, Path(args.save).resolve() if args.save else None)


if __name__ == "__main__":
    main()
