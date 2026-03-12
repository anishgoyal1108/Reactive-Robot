"""URDF-based forward kinematics helpers for the Braccio arm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree

import numpy as np


def _parse_vec3(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    parts = text.strip().split()
    if len(parts) != 3:
        return np.asarray(default, dtype=float)
    return np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def _rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _rot_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n <= 1e-12:
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


def _homogeneous(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = rot
    t[:3, 3] = trans
    return t


@dataclass(frozen=True)
class URDFJoint:
    name: str
    parent_link: str
    child_link: str
    joint_type: str
    axis_xyz: np.ndarray
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    limit_lower_rad: float | None
    limit_upper_rad: float | None


class BraccioURDFKinematics:
    """Forward kinematics from the provided Tinkerkit URDF chain."""

    SERVO_JOINT_MAP: Dict[str, int] = {
        "base": 0,
        "shoulder": 1,
        "elbow": 2,
        "wrist_pitch": 3,
        "wrist_roll": 4,
        "gripper_movable": 5,
    }

    def __init__(
        self,
        urdf_path: str | Path,
        root_link: str = "base_link",
        tip_link: str = "gripper_base",
    ):
        self.urdf_path = Path(urdf_path).resolve()
        self.root_link = root_link
        self.tip_link = tip_link
        self._joints_by_child: Dict[str, URDFJoint] = {}
        self._joints_by_name: Dict[str, URDFJoint] = {}
        self._chain_to_tip: List[URDFJoint] = []
        self._load()

    def _load(self) -> None:
        root = ElementTree.parse(self.urdf_path).getroot()
        by_child: Dict[str, URDFJoint] = {}
        by_name: Dict[str, URDFJoint] = {}

        for j in root.findall("joint"):
            name = j.attrib.get("name", "")
            j_type = j.attrib.get("type", "fixed")
            parent = (j.find("parent").attrib.get("link", "") if j.find("parent") is not None else "")
            child = (j.find("child").attrib.get("link", "") if j.find("child") is not None else "")

            origin_elem = j.find("origin")
            xyz = _parse_vec3(origin_elem.attrib.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
            rpy = _parse_vec3(origin_elem.attrib.get("rpy") if origin_elem is not None else None, (0.0, 0.0, 0.0))

            axis_elem = j.find("axis")
            axis = _parse_vec3(axis_elem.attrib.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0))

            lim = j.find("limit")
            lower = float(lim.attrib["lower"]) if lim is not None and "lower" in lim.attrib else None
            upper = float(lim.attrib["upper"]) if lim is not None and "upper" in lim.attrib else None

            joint = URDFJoint(
                name=name,
                parent_link=parent,
                child_link=child,
                joint_type=j_type,
                axis_xyz=axis,
                origin_xyz=xyz,
                origin_rpy=rpy,
                limit_lower_rad=lower,
                limit_upper_rad=upper,
            )
            by_child[child] = joint
            by_name[name] = joint

        self._joints_by_child = by_child
        self._joints_by_name = by_name
        self._chain_to_tip = self._build_chain_to_link(self.tip_link)

    def _build_chain_to_link(self, tip_link: str) -> List[URDFJoint]:
        chain: List[URDFJoint] = []
        link = tip_link
        while link != self.root_link:
            joint = self._joints_by_child.get(link)
            if joint is None:
                break
            chain.append(joint)
            link = joint.parent_link
        chain.reverse()
        return chain

    def _joint_angle_rad(self, joints_deg: List[float], joint_name: str) -> float:
        idx = self.SERVO_JOINT_MAP.get(joint_name, None)
        if idx is None or idx >= len(joints_deg):
            return 0.0
        return math.radians(float(joints_deg[idx]))

    def _transform_for_chain(self, joints_deg: List[float], chain: List[URDFJoint]) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        t = np.eye(4, dtype=float)
        by_joint: Dict[str, np.ndarray] = {}

        for j in chain:
            rpy = j.origin_rpy
            t = t @ _homogeneous(_rot_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])), j.origin_xyz)

            q = 0.0
            if j.joint_type != "fixed":
                q = self._joint_angle_rad(joints_deg, j.name)
            t = t @ _homogeneous(_rot_axis(j.axis_xyz, q), np.zeros(3, dtype=float))
            by_joint[j.name] = t.copy()

        return t, by_joint

    def joint_transform(self, joints_deg: List[float], joint_name: str) -> np.ndarray:
        joint = self._joints_by_name.get(joint_name)
        if joint is None:
            raise KeyError(f"joint not found: {joint_name}")
        chain = self._build_chain_to_link(joint.child_link)
        _, by_joint = self._transform_for_chain(joints_deg, chain)
        if joint_name not in by_joint:
            raise KeyError(f"joint transform missing: {joint_name}")
        return by_joint[joint_name]

    def link_transform(self, joints_deg: List[float], link_name: str) -> np.ndarray:
        chain = self._build_chain_to_link(link_name)
        t, _ = self._transform_for_chain(joints_deg, chain)
        return t

    def chain_positions_m(self, joints_deg: List[float], tip_link: str | None = None) -> np.ndarray:
        chain = self._chain_to_tip if tip_link in (None, self.tip_link) else self._build_chain_to_link(str(tip_link))
        t = np.eye(4, dtype=float)
        points = [t[:3, 3].copy()]

        for j in chain:
            rpy = j.origin_rpy
            t = t @ _homogeneous(_rot_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2])), j.origin_xyz)
            q = 0.0
            if j.joint_type != "fixed":
                q = self._joint_angle_rad(joints_deg, j.name)
            t = t @ _homogeneous(_rot_axis(j.axis_xyz, q), np.zeros(3, dtype=float))
            points.append(t[:3, 3].copy())

        return np.vstack(points)

    def end_effector_position_m(self, joints_deg: List[float], tip_link: str | None = None) -> np.ndarray:
        pts = self.chain_positions_m(joints_deg, tip_link=tip_link)
        return pts[-1]
