from pathlib import Path

import numpy as np

from planning.sensor_config import SENSOR_CONFIG, sensor_urdf_path
from planning.tof_projection import frame_to_points_base
from planning.urdf_kinematics import BraccioURDFKinematics
from sensing.tof_frame import ToFFrameV1


def _frame_for_ch(ch: int) -> ToFFrameV1:
    return ToFFrameV1(
        seq=1,
        mcu_ms=10,
        sensor_id=f"S{ch}",
        mux_channel=ch,
        joint_id="wrist_pitch",
        status=0,
        distances_mm=[300.0] * 64,
        validity=[1] * 64,
    )


def _home_wrist_pose() -> tuple[np.ndarray, np.ndarray]:
    urdf = sensor_urdf_path()
    kin = BraccioURDFKinematics(urdf)
    joints = [90.0, 90.0, 90.0, 90.0, 90.0, 73.0]
    tf = kin.link_transform(joints, "link4")
    return tf[:3, 3].copy(), tf[:3, :3].copy()


def test_projection_outputs_points():
    origin, rot = _home_wrist_pose()
    fr = _frame_for_ch(0)
    pts = frame_to_points_base(fr, origin, wrist_rot_base=rot)
    assert pts.shape[1] == 3
    assert pts.shape[0] > 0


def test_projection_home_compass_axes_match_requested_mapping():
    origin, rot = _home_wrist_pose()
    p0 = frame_to_points_base(_frame_for_ch(0), origin, wrist_rot_base=rot)
    p1 = frame_to_points_base(_frame_for_ch(1), origin, wrist_rot_base=rot)

    m0 = np.mean(p0, axis=0)
    m1 = np.mean(p1, axis=0)

    # User-defined home mapping in XY plane.
    assert float(m0[1]) < float(origin[1])  # CH0 -> -Y
    assert float(m1[1]) > float(origin[1])  # CH1 -> +Y


def test_projection_ignores_non_tof_channel():
    origin, rot = _home_wrist_pose()
    p2 = frame_to_points_base(_frame_for_ch(2), origin, wrist_rot_base=rot)
    assert p2.shape[1] == 3
    assert p2.shape[0] > 0


def test_projection_home_compass_axes_include_north_and_south():
    origin, rot = _home_wrist_pose()
    p2 = frame_to_points_base(_frame_for_ch(2), origin, wrist_rot_base=rot)
    p3 = frame_to_points_base(_frame_for_ch(3), origin, wrist_rot_base=rot)

    m2 = np.mean(p2, axis=0)
    m3 = np.mean(p3, axis=0)

    assert float(m2[0]) < float(origin[0])  # CH2 -> -X
    assert float(m3[0]) > float(origin[0])  # CH3 -> +X


def test_projection_mid_mount_offsets_present():
    origin, rot = _home_wrist_pose()

    dists = []
    for ch in [0, 1]:
        local = np.asarray(SENSOR_CONFIG["channels"][ch]["origin_m"], dtype=float)
        world = origin + (rot @ local)
        dists.append(float(np.linalg.norm(world - origin)))

    # Sensors are mounted away from wrist_pitch joint origin (mid-body mount model).
    assert min(dists) > 0.01
