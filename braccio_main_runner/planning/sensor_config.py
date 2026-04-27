"""Configuration defaults for ToF mounts and projection assumptions."""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

SUPPORTED_CHANNELS = [0, 1, 2, 3]
LAYOUT_NOTES_PATH = Path(__file__).resolve().parents[2] / "tof_sensor_layout_notes.json"
DEFAULT_TOF_URDF_PATH = Path(__file__).resolve().parents[2] / "Tinkerkit_model" / "tinkerkit_with_tof.urdf"
FALLBACK_TOF_URDF_PATH = Path(__file__).resolve().parents[2] / "Tinkerkit_model" / "tinkerkit4Dof_with_tof.urdf"


def _as_vec(values, default):
    seq = list(values) if values is not None else list(default)
    if len(seq) != 3:
        seq = list(default)
    return [float(seq[0]), float(seq[1]), float(seq[2])]


def _parse_text_vec3(text, default):
    if text is None:
        return list(default)
    parts = str(text).strip().split()
    if len(parts) != 3:
        return list(default)
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except Exception:
        return list(default)


def _normalize(v):
    x, y, z = _as_vec(v, (0.0, 0.0, 1.0))
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [x / n, y / n, z / n]


def _cross(a, b):
    ax, ay, az = _as_vec(a, (0.0, 0.0, 0.0))
    bx, by, bz = _as_vec(b, (0.0, 0.0, 0.0))
    return [
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]


def _dot(a, b) -> float:
    ax, ay, az = _as_vec(a, (0.0, 0.0, 0.0))
    bx, by, bz = _as_vec(b, (0.0, 0.0, 0.0))
    return float(ax * bx + ay * by + az * bz)


def _rot_about_axis(vec, axis, angle_deg: float):
    angle = math.radians(float(angle_deg))
    ux, uy, uz = _normalize(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = _as_vec(vec, (0.0, 0.0, 1.0))
    return [
        (c + ux * ux * (1.0 - c)) * x + (ux * uy * (1.0 - c) - uz * s) * y + (ux * uz * (1.0 - c) + uy * s) * z,
        (uy * ux * (1.0 - c) + uz * s) * x + (c + uy * uy * (1.0 - c)) * y + (uy * uz * (1.0 - c) - ux * s) * z,
        (uz * ux * (1.0 - c) - uy * s) * x + (uz * uy * (1.0 - c) + ux * s) * y + (c + uz * uz * (1.0 - c)) * z,
    ]


def _apply_mount_delta(nominal: dict, delta: dict) -> dict:
    origin = _as_vec(nominal.get("origin_m", [0.0, 0.0, 0.0]), (0.0, 0.0, 0.0))
    axis = _normalize(nominal.get("axis_dir", [0.0, 0.0, 1.0]))
    up = _normalize(nominal.get("up_hint", [1.0, 0.0, 0.0]))

    delta_origin = _as_vec(delta.get("delta_origin_m", [0.0, 0.0, 0.0]), (0.0, 0.0, 0.0))
    if any(abs(v) > 0.0 for v in delta_origin):
        origin = [origin[i] + delta_origin[i] for i in range(3)]

    boresight_axis = _as_vec(delta.get("boresight_rotation_axis", [1.0, 0.0, 0.0]), (1.0, 0.0, 0.0))
    boresight_deg = float(delta.get("boresight_rotation_deg", 0.0))
    if abs(boresight_deg) > 1e-9:
        axis = _normalize(_rot_about_axis(axis, boresight_axis, boresight_deg))
        up = _normalize(_rot_about_axis(up, boresight_axis, boresight_deg))

    face_roll_deg = float(delta.get("face_roll_deg", 0.0))
    if abs(face_roll_deg) > 1e-9:
        up = _normalize(_rot_about_axis(up, axis, face_roll_deg))

    out = copy.deepcopy(nominal)
    out["origin_m"] = [float(v) for v in origin]
    out["axis_dir"] = [float(v) for v in axis]
    out["up_hint"] = [float(v) for v in up]
    out["delta_origin_m"] = [float(v) for v in delta_origin]
    out["boresight_rotation_axis"] = [float(v) for v in _normalize(boresight_axis)]
    out["boresight_rotation_deg"] = float(boresight_deg)
    out["face_roll_deg"] = float(face_roll_deg)
    return out


DEFAULT_SENSOR_CONFIG = {
    "active_channels": [0, 1],
    "supported_channels": list(SUPPORTED_CHANNELS),
    "channels": {
        0: {
            "sensor_id": "tof_ch0",
            "label": "west",
            "parent_link": "link4",
            "joint_id": "wrist_vert",
            "origin_m": [0.055, 0.0, 0.0],
            "axis_dir": [1.0, 0.0, 0.0],
            "up_hint": [0.0, 0.0, 1.0],
            "yaw_deg": 90.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "fov_x_deg": 63.0,
            "fov_y_deg": 63.0,
        },
        1: {
            "sensor_id": "tof_ch1",
            "label": "east",
            "parent_link": "link4",
            "joint_id": "wrist_vert",
            "origin_m": [-0.055, 0.0, 0.0],
            "axis_dir": [-1.0, 0.0, 0.0],
            "up_hint": [0.0, 0.0, 1.0],
            "yaw_deg": -90.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "fov_x_deg": 63.0,
            "fov_y_deg": 63.0,
        },
        2: {
            "sensor_id": "tof_ch2",
            "label": "north",
            "parent_link": "link4",
            "joint_id": "wrist_vert",
            "origin_m": [0.0, -0.055, 0.0],
            "axis_dir": [0.0, -1.0, 0.0],
            "up_hint": [0.0, 0.0, 1.0],
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "fov_x_deg": 63.0,
            "fov_y_deg": 63.0,
        },
        3: {
            "sensor_id": "tof_ch3",
            "label": "south",
            "parent_link": "link4",
            "joint_id": "wrist_vert",
            "origin_m": [0.0, 0.055, 0.0],
            "axis_dir": [0.0, 1.0, 0.0],
            "up_hint": [0.0, 0.0, 1.0],
            "yaw_deg": 180.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "fov_x_deg": 63.0,
            "fov_y_deg": 63.0,
        },
    },
    "imu": {
        "joint_id": "link4",
        "parent_link": "link4",
        "origin_m": [0.000, 0.000, 0.036],
        "axis_hint": [1.0, 0.0, 0.0],
    },
    "valid_range_mm": [40.0, 3000.0],
    "inflation_m": 0.08,
    "smoothing_alpha": 0.3,
}

def _rot_rpy(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    rx = [
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ]
    ry = [
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ]
    rz = [
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ]
    import numpy as _np
    return _np.asarray(rz, dtype=float) @ _np.asarray(ry, dtype=float) @ _np.asarray(rx, dtype=float)


def sensor_urdf_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    if DEFAULT_TOF_URDF_PATH.exists():
        return DEFAULT_TOF_URDF_PATH.resolve()
    if FALLBACK_TOF_URDF_PATH.exists():
        return FALLBACK_TOF_URDF_PATH.resolve()
    return (Path(__file__).resolve().parents[2] / "Tinkerkit_model" / "tinkerkit.urdf").resolve()


def _default_channels_from_urdf() -> dict:
    path = sensor_urdf_path()
    if not path.exists():
        return copy.deepcopy(DEFAULT_SENSOR_CONFIG["channels"])
    try:
        root = ElementTree.parse(path).getroot()
    except Exception:
        return copy.deepcopy(DEFAULT_SENSOR_CONFIG["channels"])

    label_map = {0: "west", 1: "east", 2: "north", 3: "south"}
    out = {}
    for ch in SUPPORTED_CHANNELS:
        joint = root.find(f"./joint[@name='tof_ch{ch}_joint']")
        if joint is None:
            continue
        origin_elem = joint.find("origin")
        xyz = _parse_text_vec3(origin_elem.attrib.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_text_vec3(origin_elem.attrib.get("rpy") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        rot = _rot_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        import numpy as _np
        axis = _normalize((rot @ _np.asarray([1.0, 0.0, 0.0], dtype=float)).tolist())
        up = _normalize((rot @ _np.asarray([0.0, 0.0, 1.0], dtype=float)).tolist())
        parent = joint.find("parent")
        child = joint.find("child")
        out[ch] = {
            "sensor_id": f"tof_ch{ch}",
            "label": label_map.get(ch, f"ch{ch}"),
            "parent_link": str(parent.attrib.get("link", "link4")) if parent is not None else "link4",
            "joint_id": f"tof_ch{ch}_joint",
            "link_name": str(child.attrib.get("link", f"tof_ch{ch}_link")) if child is not None else f"tof_ch{ch}_link",
            "origin_m": [float(v) for v in xyz],
            "axis_dir": [float(v) for v in axis],
            "up_hint": [float(v) for v in up],
            "mount_rpy": [float(v) for v in rpy],
            "fov_x_deg": 63.0,
            "fov_y_deg": 63.0,
        }
    if len(out) == len(SUPPORTED_CHANNELS):
        return out
    return copy.deepcopy(DEFAULT_SENSOR_CONFIG["channels"])


SENSOR_CONFIG = copy.deepcopy(DEFAULT_SENSOR_CONFIG)
SENSOR_CONFIG["channels"] = _default_channels_from_urdf()


def load_sensor_layout_notes(path: Path | None = None) -> dict:
    return {}


def _legacy_absolute_to_delta(ch: int, override: dict) -> dict:
    nominal = DEFAULT_SENSOR_CONFIG["channels"][int(ch)]
    if "mount_angle_deg" in override:
        absolute = channel_fields_from_layout(
            mount_angle_deg=float(override.get("mount_angle_deg", nominal.get("mount_angle_deg", 0.0))),
            roll_deg=float(override.get("roll_deg", 0.0)),
            radial_offset_m=float(override.get("radial_offset_m", nominal.get("radial_offset_m", 0.018))),
            axial_offset_m=float(override.get("axial_offset_m", nominal.get("axial_offset_m", 0.0))),
        )
        legacy_origin = _as_vec(absolute.get("origin_m", nominal.get("origin_m", [0.0, 0.0, 0.0])), (0.0, 0.0, 0.0))
        face_roll = float(absolute.get("roll_deg", override.get("roll_deg", 0.0)))
    else:
        legacy_origin = _as_vec(override.get("origin_m", nominal.get("origin_m", [0.0, 0.0, 0.0])), (0.0, 0.0, 0.0))
        face_roll = float(override.get("roll_deg", 0.0))
    nominal_origin = _as_vec(nominal.get("origin_m", [0.0, 0.0, 0.0]), (0.0, 0.0, 0.0))
    delta_origin = [legacy_origin[i] - nominal_origin[i] for i in range(3)]
    return {
        "delta_origin_m": [float(v) for v in delta_origin],
        "boresight_rotation_axis": [1.0, 0.0, 0.0],
        "boresight_rotation_deg": 0.0,
        "face_roll_deg": float(face_roll),
    }


def apply_sensor_layout_notes(notes: dict | None = None) -> dict:
    SENSOR_CONFIG["channels"] = _default_channels_from_urdf()
    return {"schema": "tof-layout-urdf", "reference": str(sensor_urdf_path())}


def export_sensor_layout_notes() -> dict:
    return {
        "schema": "tof-layout-urdf",
        "saved_at": "",
        "reference": str(sensor_urdf_path()),
        "channels": {},
    }


def save_sensor_layout_notes(notes: dict, path: Path | None = None) -> Path:
    note_path = Path(path) if path is not None else LAYOUT_NOTES_PATH
    return note_path


def channel_fields_from_layout(
    mount_angle_deg: float,
    roll_deg: float,
    radial_offset_m: float,
    axial_offset_m: float,
) -> dict:
    ang = math.radians(float(mount_angle_deg))
    axis = _normalize([-math.cos(ang), 0.0, math.sin(ang)])
    origin = [
        float(radial_offset_m) * float(axis[0]) + float(axial_offset_m),
        0.0,
        float(radial_offset_m) * float(axis[2]),
    ]
    ref_up = [0.0, 0.0, 1.0]
    if abs(_dot(axis, ref_up)) > 0.95:
        ref_up = [0.0, 1.0, 0.0]
    up = _normalize(_rot_about_axis(ref_up, axis, float(roll_deg)))
    return {
        "origin_m": [float(v) for v in origin],
        "axis_dir": [float(v) for v in axis],
        "up_hint": [float(v) for v in up],
        "mount_angle_deg": float(mount_angle_deg),
        "roll_deg": float(roll_deg),
        "radial_offset_m": float(radial_offset_m),
        "axial_offset_m": float(axial_offset_m),
    }


def set_channel_layout(
    ch: int,
    mount_angle_deg: float,
    roll_deg: float,
    radial_offset_m: float,
    axial_offset_m: float,
) -> dict:
    fields = channel_fields_from_layout(mount_angle_deg, roll_deg, radial_offset_m, axial_offset_m)
    SENSOR_CONFIG["channels"][int(ch)].update(fields)
    return copy.deepcopy(SENSOR_CONFIG["channels"][int(ch)])


apply_sensor_layout_notes()


def set_active_channel_count(count: int) -> list[int]:
    """Update active ToF channel list to first `count` supported channels."""
    supported = SENSOR_CONFIG["supported_channels"]
    n = max(1, min(int(count), len(supported)))
    SENSOR_CONFIG["active_channels"] = list(supported[:n])
    return list(SENSOR_CONFIG["active_channels"])


def get_active_channels() -> list[int]:
    return list(SENSOR_CONFIG.get("active_channels", []))
