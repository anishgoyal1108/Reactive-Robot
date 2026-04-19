"""
braccio_ctrl.safety — Obstacle-aware motion-planning stack.

Replaces the legacy motion_guard / auto_sweep / obstacle_map / obstacle_memory
modules with a research-backed architecture (see ~/.claude/plans/calm-sprouting-shell.md
and ~/Downloads/research.md):

    WorldModel        — timestamped point cloud + KDTree + coarse log-odds grid
    ForwardKinematics — headless PyBullet FK over braccio.urdf, returns link capsules
    CollisionChecker  — analytical full-arm capsule-to-point-cloud distance
    SafetyPlanner     — BiRRT wrapper over pybullet_planning.birrt + smooth_path
    CascadingReplanner — 6-strategy recovery ladder for sequence mode
    PolarObstacleMap  — 1D angular map for sweep skip-ahead
    PreScanRoutine    — wrist-sweep workspace observation
    hysteresis        — Schmitt trigger, EMA, rate clamp primitives
    behavior_tree     — py_trees root governing manual / sweep / sequence modes
    RefusalDialog     — curses overlay for manual-mode collision refusal

Every motion request — manual key, sweep tick, sequence step, web DSL command —
routes through SafetyAPI.plan_and_validate(current_q, goal_q) and is validated
by the CollisionChecker along the joint-space interpolated path.
"""

from .hysteresis import EMA, RateClamp, SchmittTrigger
from .world_model import WorldModel
from .fk import ForwardKinematics, LINK_CAPSULES, CapsuleSpec
from .collision import (
    CollisionChecker,
    capsule_point_cloud_distance,
    capsule_capsule_distance,
)
from .planner import PlanResult, SafetyPlanner
from .polar_map import PolarObstacleMap
from .pre_scan import PreScanRoutine
from .replanner import (
    CascadingReplanner,
    SegmentResult,
    Waypoint,
    WaypointType,
    default_alternate_ik,
)
from .behavior import (
    BehaviorTreeRunner,
    RefusalInfo,
    SafetyBlackboard,
    build_emergency_branch,
    build_manual_branch,
    build_root,
    build_sequence_branch,
    build_sweep_branch,
)
from .dialog import DialogChoice, RefusalDialog, render_to_curses
from .api import SafetyAPI, SafetyAPIConfig, TofProjector

__all__ = [
    "EMA",
    "RateClamp",
    "SchmittTrigger",
    "WorldModel",
    "ForwardKinematics",
    "LINK_CAPSULES",
    "CapsuleSpec",
    "CollisionChecker",
    "capsule_point_cloud_distance",
    "capsule_capsule_distance",
    "PlanResult",
    "SafetyPlanner",
    "PolarObstacleMap",
    "PreScanRoutine",
    "CascadingReplanner",
    "SegmentResult",
    "Waypoint",
    "WaypointType",
    "default_alternate_ik",
    "BehaviorTreeRunner",
    "RefusalInfo",
    "SafetyBlackboard",
    "build_emergency_branch",
    "build_manual_branch",
    "build_root",
    "build_sequence_branch",
    "build_sweep_branch",
    "DialogChoice",
    "RefusalDialog",
    "render_to_curses",
    "SafetyAPI",
    "SafetyAPIConfig",
    "TofProjector",
]
