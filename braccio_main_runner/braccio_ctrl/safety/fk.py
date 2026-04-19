"""
fk.py — Forward kinematics + link-capsule geometry for the Braccio arm.

Loads ``braccio_twin/urdf/braccio.urdf`` into a private headless PyBullet
DIRECT client. Exposes ``link_endpoints(joints_deg) -> list[Capsule]`` which
returns the world-frame capsule for each arm segment, ready to hand to
``CollisionChecker``. The PyBullet client is this module's concern only —
nothing else in the safety stack talks to PyBullet.

Braccio↔URDF joint conversions match the braccio_twin SimArm exactly (reused
verbatim — see ``braccio_twin/sim_arm.py:400`` for the rationale). The two
PyBullet clients (safety + twin) coexist: PyBullet permits multiple DIRECT
clients in the same process, each with its own bodies.

Units
-----
The rest of the codebase works in millimetres; PyBullet is metric. Conversion
is centralised here: every public API on ``ForwardKinematics`` takes and returns
millimetres, and the caller never sees a metre.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Optional


# Deferred pybullet import — same pattern as braccio_twin.sim_arm so the safety
# module is importable in environments without PyBullet (tests that don't touch
# the collision stack, for instance).
def _import_pybullet():
    try:
        import pybullet  # pyright: ignore[reportMissingImports]
        import pybullet_data  # pyright: ignore[reportMissingImports]
        return pybullet, pybullet_data
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyBullet is not installed. Install it with:\n"
            "    pip install pybullet"
        ) from exc


# ── Capsule-spec table ─────────────────────────────────────────────────────
#
# Each arm segment is modelled as a capsule: a line segment between two named
# URDF link origins plus a radius. Radii are taken from the URDF's collision
# cylinder radii with a small inflation margin (research §4 recommends a
# generous radius to swallow FK noise and commanded-vs-actual servo drift).
#
# Link-origin convention in braccio.urdf (verified against urdf file):
#   upper_arm_link   origin = shoulder pivot
#   forearm_link     origin = elbow pivot
#   wrist_pitch_link origin = wrist pivot
#   wrist_roll_link  origin = wrist-roll center
#   gripper_link     origin = gripper tip
# So the capsules just chain along adjacent link origins.

@dataclass(frozen=True)
class CapsuleSpec:
    """Static capsule description: two link-origin names + radius (mm)."""
    name: str
    start_link: str
    end_link: str
    radius_mm: float


LINK_CAPSULES: list[CapsuleSpec] = [
    # Base + shoulder pillar — essentially a stubby vertical cylinder. We
    # still include it so a swung-low wrist can't collide with the base box.
    CapsuleSpec(name="base", start_link="base_link",
                end_link="shoulder_pillar_link", radius_mm=55.0),
    # Upper arm: shoulder pivot → elbow pivot. L1 = 125 mm.
    CapsuleSpec(name="upper_arm", start_link="upper_arm_link",
                end_link="forearm_link", radius_mm=25.0),
    # Forearm: elbow pivot → wrist pivot. L2 = 125 mm.
    CapsuleSpec(name="forearm", start_link="forearm_link",
                end_link="wrist_pitch_link", radius_mm=22.0),
    # Wrist pitch block → wrist roll center. ~30 mm.
    CapsuleSpec(name="wrist_pitch", start_link="wrist_pitch_link",
                end_link="wrist_roll_link", radius_mm=28.0),
    # Wrist roll → gripper tip. L3 = 60 mm.
    CapsuleSpec(name="gripper", start_link="wrist_roll_link",
                end_link="gripper_link", radius_mm=22.0),
]


# Self-collision exemption set.
#
# For a 5-link serial chain (base → upper_arm → forearm → wrist_pitch →
# gripper), any two links are physically connected through rigid intermedi-
# ates; their capsules can only touch near the shared joints by construction.
# The Braccio's joint limits make extreme self-fold geometry (e.g. wrist
# bending 180° back into the forearm) unreachable, so we exempt ALL pairs
# from self-collision and rely purely on obstacle-point checks.
#
# If a user later adds a branched tool (e.g., a parallel gripper with two
# fingers), they should swap this to a more conservative set.
_ADJACENT_PAIRS = frozenset({
    frozenset({a.name, b.name})
    for a in LINK_CAPSULES
    for b in LINK_CAPSULES
    if a.name != b.name
})


def capsule_pair_is_adjacent(a: str, b: str) -> bool:
    return frozenset({a, b}) in _ADJACENT_PAIRS


# ── Capsule instance (post-FK) ─────────────────────────────────────────────

@dataclass(frozen=True)
class Capsule:
    """A capsule in world frame, evaluated at a specific joint configuration.

    Coordinates in millimetres to match the rest of the codebase.
    """
    name: str
    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    radius_mm: float


# ── Braccio ↔ URDF joint conversions ───────────────────────────────────────
#
# Lifted verbatim from braccio_twin/sim_arm.py:400 so the safety stack's URDF
# pose matches the twin's exactly. The joint-0-indexed sign / offset pattern
# is the same; only the function names differ (``_braccio_deg_to_rad`` →
# ``_deg_to_urdf_rad``) to avoid confusion at call sites.

def _deg_to_urdf_rad(joint_idx: int, deg: float) -> float:
    if joint_idx == 0:
        return math.radians(deg - 90.0)
    if joint_idx == 1:
        return -math.radians(deg - 90.0) - math.pi / 2.0
    if joint_idx == 2:
        return math.radians(deg - 90.0) + math.pi / 2.0
    if joint_idx == 3:
        return math.radians(deg - 90.0)
    if joint_idx == 4:
        return math.radians(deg - 90.0)
    return 0.0  # gripper is fixed in the URDF


# Joint order must match the SET ALL serial protocol.
BRACCIO_JOINT_NAMES = [
    "joint_base",        # 0
    "joint_shoulder",    # 1
    "joint_elbow",       # 2
    "joint_wrist_vert",  # 3
    "joint_wrist_rot",   # 4
    "joint_gripper",     # 5 (fixed in URDF)
]


# ── ForwardKinematics class ────────────────────────────────────────────────

class ForwardKinematics:
    """Headless PyBullet-backed FK. One instance per process.

    The class is thread-safe: ``link_endpoints()`` acquires an internal RLock.
    Calls cost ~0.2 ms on a modern laptop CPU (the planner invokes it
    thousands of times during RRT expansion; profiling shows it's not the
    bottleneck — the capsule distance check is).
    """

    _DEFAULT_URDF = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "braccio_twin", "urdf", "braccio.urdf",
    )

    def __init__(self, urdf_path: Optional[str] = None):
        self._pb, _pybullet_data = _import_pybullet()
        self._lock = threading.RLock()
        self._urdf_path = urdf_path or os.path.abspath(self._DEFAULT_URDF)
        if not os.path.exists(self._urdf_path):
            raise FileNotFoundError(f"URDF not found: {self._urdf_path}")

        self._client_id: int = -1
        self._robot_id: int = -1
        # joint name → pybullet joint index
        self._joint_idx: dict[str, int] = {}
        # link name → pybullet link index (-1 for the base link)
        self._link_idx: dict[str, int] = {}

        self._connect()

    def _connect(self) -> None:
        pb = self._pb
        self._client_id = pb.connect(pb.DIRECT)
        pb.setGravity(0, 0, 0, physicsClientId=self._client_id)
        self._robot_id = pb.loadURDF(
            self._urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client_id,
        )

        self._link_idx["base_link"] = -1
        n_joints = pb.getNumJoints(self._robot_id, physicsClientId=self._client_id)
        for i in range(n_joints):
            info = pb.getJointInfo(self._robot_id, i, physicsClientId=self._client_id)
            jname = info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1]
            lname = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
            self._joint_idx[jname] = i
            self._link_idx[lname] = i

        # Sanity-check: every link the CapsuleSpec table references must exist.
        for cap in LINK_CAPSULES:
            for side in (cap.start_link, cap.end_link):
                if side not in self._link_idx:
                    raise RuntimeError(
                        f"URDF is missing link '{side}' required by capsule "
                        f"'{cap.name}'. Check {self._urdf_path}."
                    )

    # ── public API ─────────────────────────────────────────────────────────

    def link_endpoints(self, joints_deg: list[int]) -> list[Capsule]:
        """Forward kinematics → capsule list in world-frame millimetres.

        Parameters
        ----------
        joints_deg : list[int]
            Six Braccio degrees in protocol order [base, shoulder, elbow,
            wrist_v, wrist_r, gripper]. Gripper is ignored (fixed joint).
        """
        pb = self._pb
        with self._lock:
            # Reset every controllable joint to the requested angle. We use
            # resetJointState (not setJointMotorControl2) so FK is available
            # immediately without calling stepSimulation().
            for idx, name in enumerate(BRACCIO_JOINT_NAMES):
                pb_idx = self._joint_idx.get(name)
                if pb_idx is None:
                    continue
                jtype = pb.getJointInfo(
                    self._robot_id, pb_idx, physicsClientId=self._client_id
                )[2]
                if jtype == pb.JOINT_FIXED:
                    continue
                rad = _deg_to_urdf_rad(idx, float(joints_deg[idx]))
                pb.resetJointState(
                    self._robot_id, pb_idx, rad,
                    physicsClientId=self._client_id,
                )

            # Collect world positions of every link origin we need.
            link_pos_m: dict[str, tuple[float, float, float]] = {}
            needed = set()
            for cap in LINK_CAPSULES:
                needed.add(cap.start_link)
                needed.add(cap.end_link)
            for lname in needed:
                link_pos_m[lname] = self._link_world_pos(lname)

        # Build capsules (metres → millimetres).
        out: list[Capsule] = []
        for cap in LINK_CAPSULES:
            p0 = link_pos_m[cap.start_link]
            p1 = link_pos_m[cap.end_link]
            out.append(Capsule(
                name=cap.name,
                p0=(p0[0] * 1000.0, p0[1] * 1000.0, p0[2] * 1000.0),
                p1=(p1[0] * 1000.0, p1[1] * 1000.0, p1[2] * 1000.0),
                radius_mm=cap.radius_mm,
            ))
        return out

    def _link_world_pos(self, link_name: str) -> tuple[float, float, float]:
        pb = self._pb
        if link_name == "base_link":
            pos, _ = pb.getBasePositionAndOrientation(
                self._robot_id, physicsClientId=self._client_id
            )
            return tuple(pos)
        idx = self._link_idx[link_name]
        state = pb.getLinkState(
            self._robot_id, idx,
            computeForwardKinematics=True,
            physicsClientId=self._client_id,
        )
        return tuple(state[4])  # worldLinkFramePosition

    def tof_mount_pose(
        self, joints_deg: list[int], mount_name: str,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Return (position_mm, orientation_quat) of a ToF mount link.

        Used by ``WorldModel.ingest_tof_frame`` to project sensor hits into
        world frame without the arm-frame polar approximation the legacy
        ObstacleMap used. Orientation is PyBullet (xyzw).
        """
        pb = self._pb
        with self._lock:
            for idx, name in enumerate(BRACCIO_JOINT_NAMES):
                pb_idx = self._joint_idx.get(name)
                if pb_idx is None:
                    continue
                jtype = pb.getJointInfo(
                    self._robot_id, pb_idx, physicsClientId=self._client_id
                )[2]
                if jtype == pb.JOINT_FIXED:
                    continue
                rad = _deg_to_urdf_rad(idx, float(joints_deg[idx]))
                pb.resetJointState(
                    self._robot_id, pb_idx, rad,
                    physicsClientId=self._client_id,
                )
            idx = self._link_idx.get(mount_name)
            if idx is None:
                raise KeyError(f"unknown mount link: {mount_name}")
            state = pb.getLinkState(
                self._robot_id, idx,
                computeForwardKinematics=True,
                physicsClientId=self._client_id,
            )
            pos = state[4]
            orn = state[5]
        return (
            (pos[0] * 1000.0, pos[1] * 1000.0, pos[2] * 1000.0),
            (orn[0], orn[1], orn[2], orn[3]),
        )

    def disconnect(self) -> None:
        pb = self._pb
        with self._lock:
            if self._client_id >= 0:
                try:
                    pb.disconnect(physicsClientId=self._client_id)
                except Exception:  # pragma: no cover
                    pass
                self._client_id = -1

    def __del__(self):  # pragma: no cover
        try:
            self.disconnect()
        except Exception:
            pass
