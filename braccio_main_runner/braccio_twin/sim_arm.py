"""
sim_arm.py — PyBullet wrapper around the Braccio URDF.

SimArm loads the URDF, maps the 6 Braccio joints to PyBullet joint
indices, and provides the small API the rest of the twin needs:

    set_joint_targets(degrees_list)   # 6-tuple in Braccio order
    step()                            # one PyBullet physics tick
    get_joint_positions()             # 6 ints (degrees)
    get_link_state(name)              # world pose of a named link
    raycast(from_xyz, to_xyz)         # single-ray wrapper for sensors

All unit conversions happen here: the controller talks in degrees,
PyBullet talks in radians; the controller thinks in millimetres, the
sim runs in metres. Callers never see those conversions.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Optional


# The pybullet import is deferred to __init__ so that importing this
# module is cheap and doesn't explode in environments where pybullet
# isn't installed (e.g. CI running the Phase 0 refactor tests).
def _import_pybullet():
    try:
        import pybullet  # pyright: ignore[reportMissingImports]
        import pybullet_data  # pyright: ignore[reportMissingImports]
        return pybullet, pybullet_data
    except ImportError as exc:
        raise ImportError(
            "PyBullet is not installed. Install it with:\n"
            "    pip install pybullet\n"
            "(on the dev venv at braccio_main_runner/.venv)"
        ) from exc


# Braccio joint order, matches the SET ALL protocol string
BRACCIO_JOINT_NAMES = [
    "joint_base",        # 0 — base rotation
    "joint_shoulder",    # 1
    "joint_elbow",       # 2
    "joint_wrist_vert",  # 3
    "joint_wrist_rot",   # 4
    "joint_gripper",     # 5 — fixed in URDF, kept as placeholder
]


@dataclass
class LinkPose:
    """World pose of a URDF link — position in metres, orientation quat (xyzw)."""
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


class SimArm:
    """
    Wraps a single PyBullet client and a single Braccio arm instance.

    Thread safety
    -------------
    All mutating methods take an RLock. The SimBackend runs a single
    ticker thread; the sensor bridge and the command-parser thread also
    poke this class, so the lock keeps the step / read flows ordered.
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        gui: bool = False,
        time_step: float = 1.0 / 240.0,
        max_joint_rate_rad_s: float = 3.0,
    ):
        self._pybullet, self._pybullet_data = _import_pybullet()
        self._gui = gui
        self._time_step = time_step
        self._lock = threading.RLock()
        # Software slew rate — how many radians a joint can move per
        # step(). We interpolate toward the target instead of using
        # PyBullet's POSITION_CONTROL (which struggles with the tiny
        # inertias in the URDF).
        self._max_step_rad = max_joint_rate_rad_s * time_step

        if urdf_path is None:
            urdf_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "urdf",
                "braccio.urdf",
            )
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"Braccio URDF not found: {urdf_path}")
        self._urdf_path = urdf_path

        # PyBullet handles, populated by _connect()
        self._client_id: int = -1
        self._robot_id: int = -1
        self._ground_id: int = -1

        # Map Braccio joint name → PyBullet joint index
        self._joint_index: dict[str, int] = {}
        # Map link name → PyBullet link index (-1 for base)
        self._link_index: dict[str, int] = {}
        # Per-joint target radians (kept in URDF convention, not Braccio
        # degrees). Populated by set_joint_targets().
        self._target_rad: dict[int, float] = {}

        self._connect()
        self._load_world()

    # ── setup ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        pb = self._pybullet
        mode = pb.GUI if self._gui else pb.DIRECT
        self._client_id = pb.connect(mode)
        pb.setAdditionalSearchPath(self._pybullet_data.getDataPath())
        pb.setTimeStep(self._time_step, physicsClientId=self._client_id)
        # Gravity is intentionally disabled: real Braccio servos hold
        # position under gravity, but PyBullet's POSITION_CONTROL with
        # the tiny torque limits in our URDF can't match that. Obstacles
        # in ObstacleWorld are all baseMass=0 (fixed), so turning gravity
        # off only affects the arm — and makes commanded poses reach
        # their targets cleanly.
        pb.setGravity(0, 0, 0, physicsClientId=self._client_id)
        if self._gui:
            # Sensible camera — look at the arm from the front-right.
            pb.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=[0, 0, 0.15],
                physicsClientId=self._client_id,
            )
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_GUI, 0, physicsClientId=self._client_id
            )

    def _load_world(self) -> None:
        pb = self._pybullet
        # Ground plane
        try:
            self._ground_id = pb.loadURDF(
                "plane.urdf", physicsClientId=self._client_id
            )
        except Exception:
            # Fallback: build a thin box at z=0 if plane.urdf is missing
            col = pb.createCollisionShape(
                pb.GEOM_BOX,
                halfExtents=[2.0, 2.0, 0.001],
                physicsClientId=self._client_id,
            )
            vis = pb.createVisualShape(
                pb.GEOM_BOX,
                halfExtents=[2.0, 2.0, 0.001],
                rgbaColor=[0.85, 0.85, 0.85, 1.0],
                physicsClientId=self._client_id,
            )
            self._ground_id = pb.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[0, 0, 0],
                physicsClientId=self._client_id,
            )

        # Braccio arm
        self._robot_id = pb.loadURDF(
            self._urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client_id,
        )

        # Enumerate joints and links by name
        n_joints = pb.getNumJoints(self._robot_id, physicsClientId=self._client_id)
        self._joint_index.clear()
        self._link_index.clear()
        self._link_index["base_link"] = -1
        for i in range(n_joints):
            info = pb.getJointInfo(
                self._robot_id, i, physicsClientId=self._client_id
            )
            jname = info[1].decode("utf-8") if isinstance(info[1], bytes) else info[1]
            lname = info[12].decode("utf-8") if isinstance(info[12], bytes) else info[12]
            self._joint_index[jname] = i
            self._link_index[lname] = i
            # Disable PyBullet's default velocity-control motor on every
            # joint. Otherwise stepSimulation() introduces tiny damping
            # drift on stationary joints, which then round-trips as ±1°
            # in the Braccio-degree readback.
            if info[2] != pb.JOINT_FIXED:
                pb.setJointMotorControl2(
                    bodyUniqueId=self._robot_id,
                    jointIndex=i,
                    controlMode=pb.VELOCITY_CONTROL,
                    force=0.0,
                    physicsClientId=self._client_id,
                )

        # Move each controllable joint to Braccio's HOME_POS (90° for all
        # except gripper) so the arm starts in a sensible pose.
        self.set_joint_targets([90, 90, 90, 90, 90, 73], teleport=True)

    # ── control ──────────────────────────────────────────────────────────

    def set_joint_targets(
        self, degrees: list[int], teleport: bool = False
    ) -> None:
        """
        Update position targets for all 6 joints.

        Parameters
        ----------
        degrees  : [base, shoulder, elbow, wrist_v, wrist_r, gripper] in
                   Braccio's 0–180° convention.
        teleport : if True, snap each joint to its new target instantly.
                   If False, the target is stored and step() interpolates
                   toward it at ``max_joint_rate_rad_s``.
        """
        pb = self._pybullet
        with self._lock:
            for braccio_idx, braccio_deg in enumerate(degrees):
                jname = BRACCIO_JOINT_NAMES[braccio_idx]
                if jname not in self._joint_index:
                    continue
                pb_idx = self._joint_index[jname]
                # Gripper is fixed in the current URDF; skip.
                jtype = pb.getJointInfo(
                    self._robot_id, pb_idx, physicsClientId=self._client_id
                )[2]
                if jtype == pb.JOINT_FIXED:
                    continue
                rad = _braccio_deg_to_rad(braccio_idx, braccio_deg)
                self._target_rad[pb_idx] = rad
                if teleport:
                    pb.resetJointState(
                        self._robot_id,
                        pb_idx,
                        rad,
                        physicsClientId=self._client_id,
                    )

    def get_joint_positions(self) -> list[int]:
        """Return current joint positions as Braccio degrees (0–180)."""
        pb = self._pybullet
        out: list[int] = []
        with self._lock:
            for braccio_idx, jname in enumerate(BRACCIO_JOINT_NAMES):
                if jname not in self._joint_index:
                    out.append(90)
                    continue
                pb_idx = self._joint_index[jname]
                jtype = pb.getJointInfo(
                    self._robot_id, pb_idx, physicsClientId=self._client_id
                )[2]
                if jtype == pb.JOINT_FIXED:
                    out.append(73)  # gripper placeholder
                    continue
                rad, *_ = pb.getJointState(
                    self._robot_id, pb_idx, physicsClientId=self._client_id
                )
                out.append(_rad_to_braccio_deg(braccio_idx, rad))
        return out

    def step(self) -> None:
        """
        Advance the simulation by one tick.

        Moves every joint toward its stored target by at most
        ``max_joint_rate_rad_s * time_step`` radians (clamping to the
        target when closer than one step). This is a purely kinematic
        step — we deliberately do not call ``stepSimulation()``:

        * resetJointState already propagates forward kinematics, so
          raycasts immediately see the new pose.
        * Gravity is off and obstacles are fixed-base, so there are no
          dynamic forces we need the constraint solver to resolve.
        * Skipping physics prevents tiny contact/damping perturbations
          that would otherwise drift stationary joints off target.
        """
        pb = self._pybullet
        with self._lock:
            for pb_idx, target in self._target_rad.items():
                current, *_ = pb.getJointState(
                    self._robot_id, pb_idx, physicsClientId=self._client_id
                )
                diff = target - current
                if abs(diff) <= self._max_step_rad:
                    new_pos = target
                else:
                    new_pos = current + math.copysign(self._max_step_rad, diff)
                if new_pos != current:
                    pb.resetJointState(
                        self._robot_id,
                        pb_idx,
                        new_pos,
                        physicsClientId=self._client_id,
                    )

    # ── link / ray helpers ───────────────────────────────────────────────

    def get_link_pose(self, link_name: str) -> LinkPose:
        """World-frame pose of a named URDF link."""
        pb = self._pybullet
        with self._lock:
            if link_name == "base_link":
                pos, orn = pb.getBasePositionAndOrientation(
                    self._robot_id, physicsClientId=self._client_id
                )
                return LinkPose(tuple(pos), tuple(orn))
            idx = self._link_index.get(link_name)
            if idx is None:
                raise KeyError(f"unknown link: {link_name}")
            state = pb.getLinkState(
                self._robot_id,
                idx,
                computeForwardKinematics=True,
                physicsClientId=self._client_id,
            )
            pos = tuple(state[4])       # worldLinkFramePosition
            orn = tuple(state[5])       # worldLinkFrameOrientation
            return LinkPose(pos, orn)

    def raycast(
        self, origin: tuple[float, float, float], direction: tuple[float, float, float],
        max_dist: float,
    ) -> float:
        """
        Shoot a single ray and return the distance to the first hit
        (metres). Returns `max_dist` if nothing is hit inside range or
        if the hit body is the arm itself (self-intersection guard).
        """
        pb = self._pybullet
        length = math.sqrt(sum(d * d for d in direction))
        if length < 1e-9:
            return max_dist
        dx, dy, dz = (d / length for d in direction)
        end = (
            origin[0] + dx * max_dist,
            origin[1] + dy * max_dist,
            origin[2] + dz * max_dist,
        )
        with self._lock:
            result = pb.rayTest(
                origin, end, physicsClientId=self._client_id
            )
        if not result:
            return max_dist
        hit_body_id, _hit_link, hit_frac, _hit_pos, _hit_norm = result[0]
        if hit_body_id < 0 or hit_frac >= 1.0:
            return max_dist
        if hit_body_id == self._robot_id:
            return max_dist  # ignore self-hits
        return max_dist * hit_frac

    def get_client_id(self) -> int:
        return self._client_id

    def get_robot_id(self) -> int:
        return self._robot_id

    def disconnect(self) -> None:
        pb = self._pybullet
        if self._client_id >= 0:
            try:
                pb.disconnect(physicsClientId=self._client_id)
            except Exception:
                pass
            self._client_id = -1


# ── joint conversion helpers ────────────────────────────────────────────
#
# The Braccio servo convention is 0–180° for every joint. At Braccio
# "home" (all 90°) the real arm sits in an L-shape: upper arm vertical,
# forearm horizontal. These helpers translate between that convention
# and the URDF's natural zero pose ("all links along +X").
#
#   Joint 0  base rotation   Braccio 0..180° → URDF  -π/2..+π/2  (linear)
#   Joint 1  shoulder        Braccio 90°     → URDF  -π/2  (upper arm up)
#                            increasing Braccio deg → URDF decreases
#                            (servo rotates backward when you tell it more)
#   Joint 2  elbow           Braccio 90°     → URDF  +π/2  (forearm fwd)
#                            increasing Braccio deg → URDF increases
#   Joint 3  wrist vertical  Braccio 0..180° → URDF  -π/2..+π/2  (linear)
#   Joint 4  wrist rotation  Braccio 0..180° → URDF  -π/2..+π/2  (linear)
#   Joint 5  gripper         fixed in URDF — only the target dict echoes
#
# The net effect at HOME = [90,90,90,90,90,73]:
#   upper_arm  → pointing straight up   (+Z)
#   forearm    → pointing straight fwd  (+X)
#   wrist      → aligned with forearm   (+X)
#

def _braccio_deg_to_rad(joint_idx: int, deg: float) -> float:
    if joint_idx == 0:
        return math.radians(deg - 90.0)
    if joint_idx == 1:
        # shoulder: 90° → -π/2, 15° → -π/12, 165° → -11π/12
        return -math.radians(deg - 90.0) - math.pi / 2.0
    if joint_idx == 2:
        # elbow: 90° → +π/2, 0° → 0, 180° → π
        return math.radians(deg - 90.0) + math.pi / 2.0
    if joint_idx == 3:
        return math.radians(deg - 90.0)
    if joint_idx == 4:
        return math.radians(deg - 90.0)
    return 0.0  # gripper


def _rad_to_braccio_deg(joint_idx: int, rad: float) -> int:
    if joint_idx == 0:
        return int(round(math.degrees(rad) + 90.0))
    if joint_idx == 1:
        return int(round(90.0 - math.degrees(rad + math.pi / 2.0)))
    if joint_idx == 2:
        return int(round(90.0 + math.degrees(rad - math.pi / 2.0)))
    if joint_idx == 3:
        return int(round(math.degrees(rad) + 90.0))
    if joint_idx == 4:
        return int(round(math.degrees(rad) + 90.0))
    return 73
