"""
behavior.py — py_trees behavior tree that orchestrates the whole safety stack.

Research §7 root layout:

    Root (Selector, memory=False)
    ├── EmergencyBranch      IR severity ≥ CLOSE → halt + retreat to HOME
    ├── ManualBranch         when bb.mode == 'manual' → validate keyboard intent
    ├── SweepBranch          when bb.mode == 'sweep' → BiRRT sweep + polar skip
    └── SequenceBranch       when bb.mode == 'sequence' → cascading replanner

The tree ticks at 10 Hz from ``BehaviorTreeRunner``'s daemon thread. The
controller's curses loop never touches a planner or collision checker
directly — it posts intents to the shared ``SafetyBlackboard`` (which wraps
``py_trees.blackboard.Client``) and reads back the BT's status / refusal
reason.

The branches are deliberately flat and self-contained — each branch owns
one small state machine rather than nesting deeply. This makes them easy to
unit-test (see ``tests/safety/test_behavior_tree.py``) and easy to swap out
later if a branch grows.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import py_trees

from ..constants import SWEEP_BIRRT_FALLBACK_TIMEOUT_S
from ..ik_solver import solve_ik
from .hysteresis import EMA, SchmittTrigger
from .planner import SafetyPlanner
from .polar_map import PolarObstacleMap
from .pre_scan import PreScanRoutine
from .replanner import CascadingReplanner, SegmentResult, Waypoint
from .world_model import WorldModel


# ── Blackboard wrapper ─────────────────────────────────────────────────────

_BB_NAMESPACE = "rr_safety"

# py_trees requires every key to be registered; we enumerate them here so
# registration happens exactly once (see ``SafetyBlackboard.__init__``).
_BB_KEYS = [
    # Inputs populated by the controller thread --------------------------
    "mode",              # 'idle' | 'manual' | 'sweep' | 'sequence'
    "current_q",         # current joint list (6 ints)
    "manual_intent",     # {'axis': 'theta'|'r'|'z'|..., 'delta': int} or None
    "ir_severity",       # 0..3 from tof_sensor
    "tof_filtered",      # [ch0..ch3] post-hysteresis distances mm
    "sequence_queue",    # list[Waypoint]
    "strict_mode",       # bool: open refusal dialog on manual failure?
    "sweep_z_index",     # index into the Z-ladder used by the sweep branch
    "sweep_z_origin",    # ladder index to return to after an overtake
    "sweep_z_tries",     # consecutive z-ladder steps without a sweep_direct success
    # Outputs published by the BT for the controller thread --------------
    "pending_command",   # {'joints': [...], 'delta': int} or None
    "last_strategy",     # e.g. 'direct' | 'via_home' | 'bailout_home'
    "last_failure",      # last PlanResult.failure_reason or None
    "refusal",           # RefusalInfo dataclass or None
    "bt_state",          # 'manual_validate' | 'sweep_running' | 'emergency_halt' | ...
    "sweep_direction",   # +1 | -1
    "sweep_target_deg",  # desired theta
    "emergency",         # bool latched by EmergencyBranch
    "polar_snapshot",    # list[(lo,hi)] for telemetry
    "replan_event",      # bumped once per fresh replan (for sound hook)
]


@dataclass
class RefusalInfo:
    """Context for the manual-mode refusal dialog."""
    attempted_q: list[int]
    current_q: list[int]
    reason: str
    nearest_link: Optional[str] = None
    nearest_gap_mm: float = 0.0


class SafetyBlackboard:
    """Typed wrapper around a py_trees Blackboard client.

    py_trees Blackboards require explicit read/write permission registration;
    this wrapper keeps all of that in one place and hides the string keys
    from call sites (which would otherwise typo and silently fail).
    """

    def __init__(self, name: str = "safety_bb"):
        self._bb = py_trees.blackboard.Client(name=name, namespace=_BB_NAMESPACE)
        for key in _BB_KEYS:
            self._bb.register_key(key=key,
                                  access=py_trees.common.Access.READ)
            self._bb.register_key(key=key,
                                  access=py_trees.common.Access.WRITE)
        # Defaults
        self._bb.mode = "idle"
        self._bb.current_q = [90, 90, 90, 90, 90, 73]
        self._bb.manual_intent = None
        self._bb.ir_severity = 0
        self._bb.tof_filtered = [float("inf"), float("inf"),
                                 float("inf"), float("inf")]
        self._bb.sequence_queue = []
        self._bb.pending_command = None
        self._bb.last_strategy = ""
        self._bb.last_failure = None
        self._bb.refusal = None
        self._bb.bt_state = "idle"
        self._bb.sweep_direction = +1
        self._bb.sweep_target_deg = 90.0
        self._bb.emergency = False
        self._bb.polar_snapshot = []
        self._bb.strict_mode = False
        self._bb.sweep_z_index = 0
        self._bb.sweep_z_origin = 0
        self._bb.sweep_z_tries = 0
        self._bb.replan_event = 0

    # ── attribute-style access ─────────────────────────────────────────────
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._bb, name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_bb":
            object.__setattr__(self, name, value)
            return
        if name in _BB_KEYS:
            setattr(self._bb, name, value)
        else:
            object.__setattr__(self, name, value)


# ── Branch builders ────────────────────────────────────────────────────────

class _BranchBehaviour(py_trees.behaviour.Behaviour):
    """Base class that stores a reference to ``SafetyBlackboard``. All our
    leaves share this — py_trees requires every behaviour to subclass
    ``Behaviour`` and override ``update()``.
    """

    def __init__(self, name: str, bb: SafetyBlackboard):
        super().__init__(name=name)
        self._bb = bb


def _send_command(
    bb: SafetyBlackboard, joints: list[int], delta: int = 3,
) -> None:
    bb.pending_command = {"joints": list(joints), "delta": int(delta)}


# ── Emergency branch ───────────────────────────────────────────────────────

class _EmergencyCheck(_BranchBehaviour):
    """If IR severity has hit CLOSE (=2) or DANGER (=3), halt and retreat."""

    def __init__(self, bb: SafetyBlackboard, home_q: list[int]):
        super().__init__("emergency_check", bb)
        self._home = list(home_q)

    def update(self) -> py_trees.common.Status:
        severity = int(getattr(self._bb, "ir_severity", 0))
        if severity >= 2:
            if not self._bb.emergency:
                self._bb.emergency = True
                _send_command(self._bb, self._home, delta=5)
            self._bb.bt_state = "emergency_halt"
            return py_trees.common.Status.SUCCESS
        if self._bb.emergency:
            # IR cleared — release the latch so other branches can run.
            self._bb.emergency = False
        return py_trees.common.Status.FAILURE


def build_emergency_branch(
    bb: SafetyBlackboard, home_q: list[int] = (90, 90, 90, 90, 90, 73),
) -> py_trees.behaviour.Behaviour:
    return _EmergencyCheck(bb, list(home_q))


# ── Manual branch ──────────────────────────────────────────────────────────

class _ManualValidate(_BranchBehaviour):
    """Consume ``bb.manual_intent`` and advance the arm along a planned,
    validated path. Default behaviour is "silently auto-replan":

      1. Try the direct plan. If it succeeds, send the next step.
      2. If the direct plan fails, try a short ladder of nearby-joint
         perturbations on elbow / wrist-vertical / wrist-rotation.
         These cheap nudges let the arm slide around small obstacles
         without opening a dialog.
      3. If everything fails, clear the intent and record the failure.
         When ``bb.strict_mode`` is true, also populate ``bb.refusal`` so
         the controller can open the curses refusal dialog. Otherwise the
         refusal is silent — the user can just press the key again.

    Every fresh refusal bumps ``bb.replan_event`` so the controller can
    play the completion-fail.oga sound once per event.
    """

    def __init__(
        self,
        bb: SafetyBlackboard,
        planner: SafetyPlanner,
        nearest_obstacle_fn: Optional[Callable[[list[int]],
                                                Optional[tuple[str, float, np.ndarray]]]] = None,
    ):
        super().__init__("manual_validate", bb)
        self._planner = planner
        self._nearest = nearest_obstacle_fn

    def update(self) -> py_trees.common.Status:
        if self._bb.mode != "manual":
            return py_trees.common.Status.FAILURE
        intent = self._bb.manual_intent
        if intent is None:
            self._bb.bt_state = "manual_idle"
            return py_trees.common.Status.SUCCESS

        current = list(self._bb.current_q)
        target = list(intent)
        if len(target) != 6:
            self._bb.manual_intent = None
            return py_trees.common.Status.FAILURE

        self._bb.bt_state = "manual_validate"

        # 0. If the arm is currently in collision (a just-ingested ToF point
        # landed inside a capsule), every plan from here would fail with
        # start_in_collision — which is what used to deadlock the BT.
        # Move to a nearby safe config instead, keeping the user's intent
        # live so the next tick can continue toward it once we're clear.
        if self._planner.collision.config_in_collision(current):
            escape = self._planner.escape_from_collision(current)
            if escape is not None:
                _send_command(self._bb, escape, delta=3)
                self._bb.last_strategy = "escape_collision"
                self._bb.last_failure = "start_in_collision"
                self._bb.refusal = None
                self._bb.replan_event = int(self._bb.replan_event) + 1
                return py_trees.common.Status.SUCCESS
            # No escape found — fall through to the refusal path so the
            # user can back off manually.

        # 1. Direct plan.
        result = self._planner.plan(current, target)
        if result.success:
            next_q = result.path[1] if len(result.path) > 1 else result.path[0]
            _send_command(self._bb, next_q, delta=3)
            self._bb.last_strategy = "direct"
            self._bb.last_failure = None
            self._bb.refusal = None
            if next_q == target:
                self._bb.manual_intent = None
            return py_trees.common.Status.SUCCESS

        # 2. Joint-nudge ladder — try a handful of nearby alternates that
        # keep the commanded axis moving in the user's chosen direction
        # but lift the elbow / rotate the wrist to slip past an obstacle.
        for nudge_q, tag in _manual_nudge_candidates(current, target):
            nudge_result = self._planner.plan(current, nudge_q)
            if nudge_result.success:
                next_q = (nudge_result.path[1]
                          if len(nudge_result.path) > 1
                          else nudge_result.path[0])
                _send_command(self._bb, next_q, delta=3)
                self._bb.last_strategy = f"manual_nudge:{tag}"
                self._bb.last_failure = None
                self._bb.refusal = None
                # Leave the intent in place so the user's held key keeps
                # advancing toward the original target. The nudge only
                # supplies a single step of detour.
                return py_trees.common.Status.SUCCESS

        # 3. Nothing worked — record the failure, bump the replan event,
        # and optionally populate the refusal dialog payload.
        self._bb.replan_event = int(self._bb.replan_event) + 1
        self._bb.last_strategy = "refused"
        self._bb.last_failure = result.failure_reason
        self._bb.manual_intent = None
        if self._bb.strict_mode:
            info = RefusalInfo(
                attempted_q=target,
                current_q=current,
                reason=result.failure_reason or "unknown",
            )
            if self._nearest is not None:
                try:
                    near = self._nearest(target)
                    if near is not None:
                        info.nearest_link = near[0]
                        info.nearest_gap_mm = float(near[1])
                except Exception:
                    pass
            self._bb.refusal = info
        else:
            self._bb.refusal = None
        return py_trees.common.Status.SUCCESS


def _manual_nudge_candidates(
    current: list[int], target: list[int],
) -> list[tuple[list[int], str]]:
    """Return a short list of ``(candidate_q, tag)`` tuples that preserve
    the manual axis intent but vary elbow / wrist to go around an
    obstacle. Capped small so the per-tick planner budget stays tight.
    """
    cands: list[tuple[list[int], str]] = []
    for delta_elbow in (+15, -15, +25, -25):
        q = list(target)
        q[2] = max(0, min(180, q[2] + delta_elbow))
        cands.append((q, f"elbow{delta_elbow:+d}"))
    for delta_wv in (+20, -20):
        q = list(target)
        q[3] = max(0, min(180, q[3] + delta_wv))
        cands.append((q, f"wv{delta_wv:+d}"))
    return cands


def build_manual_branch(
    bb: SafetyBlackboard, planner: SafetyPlanner,
    nearest_obstacle_fn=None,
) -> py_trees.behaviour.Behaviour:
    return _ManualValidate(bb, planner, nearest_obstacle_fn)


# ── Sweep branch ───────────────────────────────────────────────────────────

SWEEP_STEP_DEG = 5
SWEEP_R_DEFAULT_MM = 152.0
SWEEP_Z_DEFAULT_MM = 35.0
SWEEP_THETA_MIN = 0.0
SWEEP_THETA_MAX = 180.0
SWEEP_SKIP_MARGIN_DEG = 10.0
SWEEP_FOV_HALF_DEG = 22.5
# Alternate Z levels the branch visits when the current slice is blocked.
SWEEP_Z_LADDER_MM = (35.0, 60.0, 90.0, 10.0, -20.0)


class _SweepTick(_BranchBehaviour):
    """One sweep tick: advance the base angle by ``step_deg``, plan the
    short transit, and send the first step.

    When the direction of travel is blocked:
      * try to skip past the block using the polar map;
      * if the skip fails in both directions, climb the Z ladder (the
        sweep moves up / over the obstacle);
      * if every Z level is blocked, halt and record ``sweep_blocked``.

    The polar map is filled from the world-frame point cloud only — we do
    NOT mark the *current* theta just because a ToF Schmitt trigger fired.
    That false-positive (side-facing CH1/CH2 firing on the arm's own
    structure or on an obstacle behind the sweep direction) was one of
    the reasons the legacy sweep stopped and jittered.
    """

    def __init__(
        self,
        bb: SafetyBlackboard,
        planner: SafetyPlanner,
        world: WorldModel,
        polar: PolarObstacleMap,
        hysteresis: dict[int, SchmittTrigger],
        step_deg: float = SWEEP_STEP_DEG,
        r_mm: float = SWEEP_R_DEFAULT_MM,
        z_mm: float = SWEEP_Z_DEFAULT_MM,
        z_ladder: tuple[float, ...] = SWEEP_Z_LADDER_MM,
    ):
        super().__init__("sweep_tick", bb)
        self._planner = planner
        self._world = world
        self._polar = polar
        self._hysteresis = hysteresis
        self._step = float(step_deg)
        self._r = float(r_mm)
        self._z_default = float(z_mm)
        self._z_ladder = tuple(float(z) for z in z_ladder)

    def _current_z(self) -> float:
        """Z level the sweep is currently running at — index into ladder."""
        idx = int(getattr(self._bb, "sweep_z_index", 0) or 0)
        if 0 <= idx < len(self._z_ladder):
            return self._z_ladder[idx]
        return self._z_default

    def update(self) -> py_trees.common.Status:
        if self._bb.mode != "sweep":
            return py_trees.common.Status.FAILURE

        self._polar.decay()
        self._refresh_polar_from_world()

        direction = int(self._bb.sweep_direction)
        current_q = list(self._bb.current_q)
        current_theta = float(current_q[0])

        # Escape-from-collision: if the world model has a point inside a
        # capsule (typical after a ToF reflection off the arm itself), the
        # planner will refuse forever with ``start_in_collision``. Detour
        # to a nearby safe config before touching the sweep logic.
        if self._planner.collision.config_in_collision(current_q):
            escape = self._planner.escape_from_collision(current_q)
            if escape is not None:
                _send_command(self._bb, escape, delta=3)
                self._bb.last_strategy = "escape_collision"
                self._bb.last_failure = "start_in_collision"
                self._bb.replan_event = int(self._bb.replan_event) + 1
                self._bb.bt_state = "sweep_escape"
                return py_trees.common.Status.SUCCESS

        next_theta = current_theta + direction * self._step
        if next_theta < SWEEP_THETA_MIN or next_theta > SWEEP_THETA_MAX:
            # Reached the end — reverse direction.
            direction = -direction
            self._bb.sweep_direction = direction
            next_theta = current_theta + direction * self._step
            next_theta = max(SWEEP_THETA_MIN, min(SWEEP_THETA_MAX, next_theta))

        # Escalation ladder when the next step is blocked:
        #   1. skip-via-polar in the forward direction (narrow-notch case);
        #   2. z-ladder — cheap, up to len(z_ladder)-1 attempts before escalating;
        #   3. one BiRRT in full 6-DOF to the opposite edge — lets the planner
        #      lift the elbow/wrist over or under a one-sided obstacle;
        #   4. last resort: flip direction so the arm keeps moving.
        # The key fix from the old code is that direction-flip is now the
        # fallback of last resort, not the immediate response — otherwise a
        # one-sided block just makes the arm wrap around to the opposite
        # half-plane and oscillate instead of going over the obstacle.
        if self._polar.is_blocked(next_theta):
            skip = self._polar.next_unblocked(
                next_theta, direction, margin_deg=SWEEP_SKIP_MARGIN_DEG,
            )
            if skip is not None:
                next_theta = float(skip)
            else:
                # Forward skip failed — the block extends to the edge or the
                # margin lands outside the domain. Try escalation.
                z_tries = int(
                    getattr(self._bb, "sweep_z_tries", 0) or 0
                )
                if (
                    z_tries < len(self._z_ladder) - 1
                    and self._advance_z_ladder()
                ):
                    self._bb.sweep_z_tries = z_tries + 1
                    new_z = self._current_z()
                    self._bb.bt_state = "sweep_z_ladder"
                    self._bb.last_strategy = (
                        f"sweep_z_ladder:{int(round(new_z))}"
                    )
                    self._bb.last_failure = None
                    self._bb.polar_snapshot = self._polar.blocked_ranges()
                    return py_trees.common.Status.SUCCESS

                # Z-ladder exhausted — one BiRRT to the opposite edge. The
                # budget is small (SWEEP_BIRRT_FALLBACK_TIMEOUT_S) because
                # this fires ~once per obstacle event and the resulting
                # path caches subsequent ticks.
                edge_theta = (
                    SWEEP_THETA_MAX if direction > 0 else SWEEP_THETA_MIN
                )
                edge_q = self._ik_target(
                    edge_theta, self._r, self._current_z(), current_q,
                )
                if edge_q is not None:
                    over = self._planner.plan(
                        current_q, edge_q,
                        timeout_s=SWEEP_BIRRT_FALLBACK_TIMEOUT_S,
                    )
                    if over.success and over.path:
                        next_q = (
                            over.path[1]
                            if len(over.path) > 1
                            else over.path[0]
                        )
                        _send_command(self._bb, next_q, delta=3)
                        self._bb.sweep_target_deg = edge_theta
                        self._bb.bt_state = "sweep_running"
                        self._bb.last_strategy = "sweep_birrt_over"
                        self._bb.last_failure = None
                        self._bb.polar_snapshot = (
                            self._polar.blocked_ranges()
                        )
                        self._bb.sweep_z_tries = 0
                        return py_trees.common.Status.SUCCESS

                # Last resort — flip direction so the arm keeps moving
                # rather than stalling.
                direction = -direction
                self._bb.sweep_direction = direction
                self._bb.bt_state = "sweep_edge_reverse"
                self._bb.last_strategy = "sweep_edge_reverse"
                self._bb.last_failure = "z_ladder_and_birrt_exhausted"
                self._bb.replan_event = int(self._bb.replan_event) + 1
                self._bb.sweep_z_tries = 0
                return py_trees.common.Status.SUCCESS

        self._bb.sweep_target_deg = next_theta
        target_q = [int(round(next_theta))] + current_q[1:]

        res = self._planner.plan(current_q, target_q)
        if not res.success:
            # Mark the next theta as blocked so we skip it next tick.
            self._polar.mark_blocked(next_theta)
            self._bb.replan_event = int(self._bb.replan_event) + 1
            self._bb.last_failure = res.failure_reason
            self._bb.last_strategy = f"sweep_replan:{res.failure_reason}"
            # After marking this theta blocked, try to reroute *this* tick
            # by picking a neighbour that isn't blocked — avoids a visible
            # "stuck for one tick every replan" stutter.
            reroute = self._polar.next_unblocked(
                next_theta, direction, margin_deg=SWEEP_SKIP_MARGIN_DEG,
            )
            if reroute is not None:
                alt_q = [int(round(reroute))] + current_q[1:]
                alt = self._planner.plan(current_q, alt_q)
                if alt.success:
                    next_q = alt.path[1] if len(alt.path) > 1 else alt.path[0]
                    _send_command(self._bb, next_q, delta=3)
                    self._bb.sweep_target_deg = reroute
                    self._bb.bt_state = "sweep_running"
                    self._bb.last_strategy = "sweep_reroute"
                    self._bb.polar_snapshot = self._polar.blocked_ranges()
                    return py_trees.common.Status.SUCCESS
            self._bb.bt_state = "sweep_replan"
            return py_trees.common.Status.SUCCESS

        next_q = res.path[1] if len(res.path) > 1 else res.path[0]
        _send_command(self._bb, next_q, delta=3)
        self._bb.bt_state = "sweep_running"
        self._bb.last_strategy = "sweep_direct"
        self._bb.last_failure = None
        self._bb.polar_snapshot = self._polar.blocked_ranges()
        self._bb.sweep_z_tries = 0

        # Overtake recovery — one rung per successful clear step. If the
        # lower slice turns out to still be blocked, the next tick's
        # forward_blocked check re-climbs via _advance_z_ladder, reusing
        # the same escalation ladder that handled the initial overtake.
        z_idx = int(self._bb.sweep_z_index)
        z_origin = int(getattr(self._bb, "sweep_z_origin", 0) or 0)
        if z_idx > z_origin and self._descend_z_ladder():
            new_z = self._current_z()
            self._bb.last_strategy = (
                f"sweep_descend:{int(round(new_z))}"
            )

        return py_trees.common.Status.SUCCESS

    def _refresh_polar_from_world(self) -> None:
        z = self._current_z()
        ranges = self._world.polar_blocked_ranges(
            self._r, z, fov_half_deg=SWEEP_FOV_HALF_DEG,
        )
        for lo, hi in ranges:
            self._polar.mark_range(lo, hi)

    def _shift_z_index(self, delta: int) -> bool:
        """Move the sweep Z rung by ``delta``. Returns ``False`` if the new
        index would fall outside ``[0, len(ladder))`` — caller escalates.

        Clears the polar map on a successful shift so the new z-slice
        re-populates from the world model on the next tick. Shared between
        the ascent (``_advance_z_ladder``) and descent (``_descend_z_ladder``)
        paths so the polar-clear / bounds-check logic lives in exactly one
        place.
        """
        cur = int(getattr(self._bb, "sweep_z_index", 0) or 0)
        new = cur + int(delta)
        if not (0 <= new < len(self._z_ladder)):
            return False
        if new == cur:
            return False
        self._bb.sweep_z_index = new
        self._polar.clear()
        return True

    def _advance_z_ladder(self) -> bool:
        """Climb one rung toward the top of the ladder."""
        return self._shift_z_index(+1)

    def _descend_z_ladder(self) -> bool:
        """Step one rung toward the bottom of the ladder (origin)."""
        return self._shift_z_index(-1)

    def _ik_target(
        self,
        theta_deg: float,
        r_mm: float,
        z_mm: float,
        current_q: list[int],
    ) -> Optional[list[int]]:
        """6-DOF goal config for the sweep's (θ, r, z) slice.

        Uses the existing 2-link planar IK for base/shoulder/elbow/wrist-v.
        Wrist-rot and gripper inherit from ``current_q``. Returns ``None``
        when the Cartesian target is outside the arm's dexterous workspace.
        """
        theta_rad = math.radians(theta_deg)
        x = r_mm * math.cos(theta_rad)
        y = r_mm * math.sin(theta_rad)
        sol = solve_ik(x, y, z_mm)
        if sol is None:
            return None
        return [
            int(sol.base),
            int(sol.shoulder),
            int(sol.elbow),
            int(sol.wrist_vert),
            int(current_q[4]),
            int(current_q[5]),
        ]


def build_sweep_branch(
    bb: SafetyBlackboard,
    planner: SafetyPlanner,
    world: WorldModel,
    polar: PolarObstacleMap,
    hysteresis: dict[int, SchmittTrigger],
    **kwargs,
) -> py_trees.behaviour.Behaviour:
    return _SweepTick(bb, planner, world, polar, hysteresis, **kwargs)


# ── Sequence branch ────────────────────────────────────────────────────────

class _SequenceTick(_BranchBehaviour):
    """Drain ``bb.sequence_queue`` one waypoint at a time through the
    cascading replanner.

    Per-step re-validation: before sending each step of the cached segment
    path, the checker re-runs ``config_in_collision`` against the *current*
    world model. If the previously-clear step is now blocked (an obstacle
    moved into the path), the cached path is discarded and the segment is
    re-planned from the current arm pose on the next tick.
    """

    def __init__(
        self,
        bb: SafetyBlackboard,
        replanner: CascadingReplanner,
        collision_checker=None,
    ):
        super().__init__("sequence_tick", bb)
        self._replanner = replanner
        self._collision = collision_checker
        self._active_path: Optional[list[list[int]]] = None
        self._path_idx = 0
        self._active_goal: Optional[Waypoint] = None

    def update(self) -> py_trees.common.Status:
        if self._bb.mode != "sequence":
            return py_trees.common.Status.FAILURE

        current = list(self._bb.current_q)

        # Escape-from-collision: identical reason as the manual branch —
        # a newly-ingested point may have put the arm into a "planner says
        # start_in_collision" state. Retreat before touching the queue.
        if (
            self._collision is not None
            and self._collision.config_in_collision(current)
        ):
            escape = self._replanner.planner.escape_from_collision(current)
            if escape is not None:
                _send_command(self._bb, escape, delta=3)
                self._bb.last_strategy = "escape_collision"
                self._bb.last_failure = "start_in_collision"
                self._bb.replan_event = int(self._bb.replan_event) + 1
                self._active_path = None
                self._path_idx = 0
                self._bb.bt_state = "sequence_escape"
                return py_trees.common.Status.SUCCESS

        # Continue playing back the active segment path, if any.
        if self._active_path is not None and self._path_idx < len(self._active_path):
            target = self._active_path[self._path_idx]
            if target == current:
                self._path_idx += 1
                if self._path_idx >= len(self._active_path):
                    self._active_path = None
                    self._path_idx = 0
                    return py_trees.common.Status.SUCCESS
                target = self._active_path[self._path_idx]

            # Per-step re-validation against the CURRENT world model.
            if self._collision is not None and self._collision.config_in_collision(target):
                # Cached path now leads through a freshly-detected obstacle.
                # Drop the cached path, bump the replan counter, and push
                # the same goal back to the front of the queue so the
                # next tick re-enters the "plan a new segment" branch.
                self._bb.replan_event = int(self._bb.replan_event) + 1
                self._bb.last_strategy = "sequence_revalidate"
                if self._active_goal is not None:
                    self._bb.sequence_queue = (
                        [self._active_goal] + list(self._bb.sequence_queue or [])
                    )
                self._active_path = None
                self._path_idx = 0
                self._active_goal = None
                self._bb.bt_state = "sequence_replan"
                return py_trees.common.Status.SUCCESS

            _send_command(self._bb, target, delta=3)
            self._bb.bt_state = "sequence_playback"
            return py_trees.common.Status.SUCCESS

        # Grab the next waypoint.
        queue: list[Waypoint] = list(self._bb.sequence_queue or [])
        if not queue:
            self._bb.bt_state = "sequence_done"
            return py_trees.common.Status.SUCCESS

        goal = queue[0]
        next_goal = queue[1] if len(queue) > 1 else None
        start_wp = Waypoint(q=current)
        self._bb.bt_state = "sequence_planning"
        result: SegmentResult = self._replanner.execute_segment(
            start_wp, goal, next_goal=next_goal,
        )
        self._bb.last_strategy = result.strategy
        self._bb.last_failure = result.failure_reason
        if result.strategy.startswith("via_") or result.strategy == "wait_retry":
            self._bb.replan_event = int(self._bb.replan_event) + 1

        if not result.success:
            self._bb.replan_event = int(self._bb.replan_event) + 1
            self._bb.sequence_queue = []
            self._active_path = result.path
            self._path_idx = 0
            self._active_goal = None
            self._bb.bt_state = "sequence_failed"
            return py_trees.common.Status.SUCCESS

        if result.strategy == "skip_via" and len(queue) >= 2:
            queue = queue[2:]
        else:
            queue = queue[1:]
        self._bb.sequence_queue = queue
        self._active_path = result.path
        self._active_goal = goal
        self._path_idx = 1 if result.path and result.path[0] == current else 0
        return py_trees.common.Status.SUCCESS


def build_sequence_branch(
    bb: SafetyBlackboard, replanner: CascadingReplanner,
    collision_checker=None,
) -> py_trees.behaviour.Behaviour:
    return _SequenceTick(bb, replanner, collision_checker=collision_checker)


# ── Root tree ──────────────────────────────────────────────────────────────

def build_root(
    bb: SafetyBlackboard,
    planner: SafetyPlanner,
    replanner: CascadingReplanner,
    world: WorldModel,
    polar: PolarObstacleMap,
    hysteresis: dict[int, SchmittTrigger],
    home_q: list[int] = (90, 90, 90, 90, 90, 73),
    nearest_obstacle_fn=None,
    collision_checker=None,
) -> py_trees.behaviour.Behaviour:
    root = py_trees.composites.Selector(name="rr_safety_root", memory=False)
    root.add_children([
        build_emergency_branch(bb, home_q),
        build_manual_branch(bb, planner, nearest_obstacle_fn),
        build_sweep_branch(bb, planner, world, polar, hysteresis),
        build_sequence_branch(bb, replanner, collision_checker=collision_checker),
    ])
    return root


# ── Runner ─────────────────────────────────────────────────────────────────

@dataclass
class BehaviorTreeRunner:
    """Wraps the root + a daemon tick thread.

    ``send_cmd`` is the only I/O boundary: called whenever ``pending_command``
    is non-None. In production it routes to ``serial_bridge.send_cmd``; tests
    hand in a simple list.append.
    """
    bb: SafetyBlackboard
    root: py_trees.behaviour.Behaviour
    send_cmd: Callable[[dict], None]
    tick_hz: float = 10.0
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bt_tick", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def tick_once(self) -> None:
        self.root.tick_once()
        cmd = self.bb.pending_command
        if cmd is not None:
            self.send_cmd(cmd)
            self.bb.pending_command = None

    def _loop(self) -> None:
        period = 1.0 / max(1e-3, self.tick_hz)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick_once()
            except Exception:
                # Don't let a branch exception kill the thread. The failing
                # branch will continue to fail on each tick until reset.
                pass
            elapsed = time.monotonic() - t0
            remaining = period - elapsed
            if remaining > 0:
                self._stop.wait(remaining)
