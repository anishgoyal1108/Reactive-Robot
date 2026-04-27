"""Project 8x8 ToF depth grids into base-frame 3D points."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from planning.sensor_config import SENSOR_CONFIG
from sensing.tof_frame import ToFFrameV1


def _rot_zyx(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    cz, sz = math.cos(y), math.sin(y)
    cy, sy = math.cos(p), math.sin(p)
    cx, sx = math.cos(r), math.sin(r)

    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    return rz @ ry @ rx


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def _sensor_basis(axis_dir: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = _normalize(axis_dir)
    uph = _normalize(up_hint)
    right = np.cross(uph, axis)
    if np.linalg.norm(right) <= 1e-9:
        # Degenerate hint; pick a fallback not collinear with axis.
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(axis, fallback))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(fallback, axis)
    right = _normalize(right)
    up = _normalize(np.cross(axis, right))
    return axis, right, up


def _ray_direction(axis: np.ndarray, right: np.ndarray, up: np.ndarray, ax: float, ay: float) -> np.ndarray:
    vec = axis + math.tan(ax) * right + math.tan(ay) * up
    return _normalize(vec)


def frame_cells_to_points_base(
    frame: ToFFrameV1,
    wrist_origin_base_m: np.ndarray,
    wrist_rot_base: np.ndarray | None = None,
    base_theta_deg: float = 0.0,
    arm_pitch_deg: float = 0.0,
    wrist_offset_deg: float = 0.0,
) -> np.ndarray:
    """Convert one frame to per-cell projected points in base frame."""
    cfg = SENSOR_CONFIG["channels"].get(frame.mux_channel)
    if cfg is None:
        return []

    fov_x = math.radians(cfg["fov_x_deg"])
    fov_y = math.radians(cfg["fov_y_deg"])

    rows = max(1, int(getattr(frame, "rows", 8)))
    cols = max(1, int(getattr(frame, "cols", 8)))
    if (rows * cols) != len(frame.distances_mm) or (rows * cols) != len(frame.validity):
        return []

    d = np.asarray(frame.distances_mm, dtype=float).reshape(rows, cols) / 1000.0
    v = np.asarray(frame.validity, dtype=int).reshape(rows, cols)

    r0, r1 = SENSOR_CONFIG["valid_range_mm"]
    lo = r0 / 1000.0
    hi = r1 / 1000.0

    origin_base = np.asarray(wrist_origin_base_m, dtype=float).reshape(3)
    rot_base = np.eye(3, dtype=float) if wrist_rot_base is None else np.asarray(wrist_rot_base, dtype=float)

    # Preferred wrist-frame compass model.
    if "axis_dir" in cfg:
        local_origin = np.asarray(cfg.get("origin_m", [0.0, 0.0, 0.0]), dtype=float)
        axis, right, up = _sensor_basis(
            np.asarray(cfg["axis_dir"], dtype=float),
            np.asarray(cfg.get("up_hint", [0.0, 0.0, 1.0]), dtype=float),
        )

        pts: list[dict] = []
        row_center = 0.5 * float(rows - 1)
        col_center = 0.5 * float(cols - 1)
        row_scale = max(0.5, row_center)
        col_scale = max(0.5, col_center)
        for i in range(rows):
            for j in range(cols):
                if v[i, j] == 0:
                    continue
                dist = d[i, j]
                if not np.isfinite(dist) or dist < lo or dist > hi:
                    continue

                nx = (float(j) - col_center) / col_scale
                ny = (float(i) - row_center) / row_scale
                ax = nx * (fov_x * 0.5)
                ay = ny * (fov_y * 0.5)

                dir_local = _ray_direction(axis, right, up, ax, ay)
                p_local = local_origin + dir_local * dist
                p_base = origin_base + (rot_base @ p_local)
                pts.append(
                    {
                        "row": int(i),
                        "col": int(j),
                        "distance_mm": float(dist * 1000.0),
                        "point_base": np.asarray(p_base, dtype=float),
                        "dir_local": np.asarray(dir_local, dtype=float),
                    }
                )

        return pts

    # Legacy orientation fallback.
    mount_rot = _rot_zyx(
        cfg.get("yaw_deg", 0.0),
        cfg.get("pitch_deg", 0.0),
        cfg.get("roll_deg", 0.0),
    )
    base_rot = _rot_zyx(base_theta_deg, 0.0, 0.0)
    arm_rot = _rot_zyx(0.0, arm_pitch_deg + wrist_offset_deg, 0.0)
    world_rot = base_rot @ arm_rot @ mount_rot

    mount_offset = np.asarray(cfg["origin_m"], dtype=float)
    origin = origin_base + (base_rot @ arm_rot @ mount_offset)

    pts: list[dict] = []
    row_center = 0.5 * float(rows - 1)
    col_center = 0.5 * float(cols - 1)
    row_scale = max(0.5, row_center)
    col_scale = max(0.5, col_center)
    for i in range(rows):
        for j in range(cols):
            if v[i, j] == 0:
                continue
            dist = d[i, j]
            if not np.isfinite(dist) or dist < lo or dist > hi:
                continue

            nx = (float(j) - col_center) / col_scale
            ny = (float(i) - row_center) / row_scale
            ax = nx * (fov_x * 0.5)
            ay = ny * (fov_y * 0.5)

            dir_s = np.array(
                [
                    math.cos(ay) * math.cos(ax),
                    math.cos(ay) * math.sin(ax),
                    math.sin(ay),
                ],
                dtype=float,
            )
            p_sensor = dir_s * dist
            p_base = origin + (world_rot @ p_sensor)
            pts.append(
                {
                    "row": int(i),
                    "col": int(j),
                    "distance_mm": float(dist * 1000.0),
                    "point_base": np.asarray(p_base, dtype=float),
                    "dir_local": np.asarray(dir_s, dtype=float),
                }
            )

    return pts


def frame_to_points_base(
    frame: ToFFrameV1,
    wrist_origin_base_m: np.ndarray,
    wrist_rot_base: np.ndarray | None = None,
    base_theta_deg: float = 0.0,
    arm_pitch_deg: float = 0.0,
    wrist_offset_deg: float = 0.0,
) -> np.ndarray:
    """
    Convert one frame to Nx3 points in base frame.

    Preferred mode:
    - `wrist_rot_base` is provided from FK for the wrist_pitch frame.
    - Sensor compass layout (`axis_dir`) is interpreted in that local frame.

    Compatibility mode:
    - Falls back to legacy yaw/pitch/roll config if `axis_dir` is absent.
    """
    cells = frame_cells_to_points_base(
        frame,
        wrist_origin_base_m,
        wrist_rot_base=wrist_rot_base,
        base_theta_deg=base_theta_deg,
        arm_pitch_deg=arm_pitch_deg,
        wrist_offset_deg=wrist_offset_deg,
    )
    if not cells:
        return np.zeros((0, 3), dtype=float)
    return np.vstack([np.asarray(cell["point_base"], dtype=float).reshape(3) for cell in cells])
