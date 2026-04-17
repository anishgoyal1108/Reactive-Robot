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

from .. import constants as _c
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
    "sweep_z_visited",   # list[int] of ladder indices tried since last clear
    "sweep_detour_path", # list[list[int]] of 6-DOF waypoints for the active detour; [] when none
    "sweep_detour_idx",  # index of the next detour waypoint to send
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
        self._bb.sweep_z_visited = []
        self._bb.sweep_detour_path = []
        self._bb.sweep_detour_idx = 0
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
    bypass_smoothing: bool = False,
) -> None:
    cmd: dict = {"joints": list(joints), "delta": int(delta)}
    if bypass_smoothing:
        cmd["bypass_smoothing"] = True
    bb.pending_command = cmd


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
                _send_command(self._bb, self._home, delta=5, bypass_smoothing=True)
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

def _find_block_at(
    ranges: list[tuple[float, float]],
    theta: float,
    direction: int,
    fov_half_deg: float,
) -> Optional[tuple[float, float]]:
    """Pick the blocked interval containing / just ahead of ``theta``.

    "Ahead" is in the sweep direction — for +1 we want the first range
    whose ``lo <= theta + fov_half``; for -1 the range whose ``hi >=
    theta - fov_half``. Returns ``None`` when no block is adjacent, which
    tells ``_compute_theta_clear`` to stop chaining.
    """
    if not ranges:
        return None
    if direction >= 0:
        for lo, hi in ranges:
            if hi < theta:
                continue
            if lo <= theta + fov_half_deg:
                return (float(lo), float(hi))
        return None
    for lo, hi in reversed(ranges):
        if lo > theta:
            continue
        if hi >= theta - fov_half_deg:
            return (float(lo), float(hi))
    return None


# Tuning pulled from constants.py — constants.py is the single source of
# truth. Pre-2026-04-17 this module re-declared its own values and silently
# shadowed any change made in constants.py (which is why the sweep still
# ran at 5°/tick after the user edited constants.SWEEP_STEP_DEG).
SWEEP_STEP_DEG = float(_c.SWEEP_STEP_DEG)
SWEEP_R_DEFAULT_MM = float(_c.SWEEP_R_DEFAULT_MM)
SWEEP_Z_DEFAULT_MM = float(_c.SWEEP_Z_DEFAULT_MM)
SWEEP_THETA_MIN = 0.0
SWEEP_THETA_MAX = 180.0
SWEEP_SKIP_MARGIN_DEG = float(_c.SWEEP_SKIP_MARGIN_DEG)
SWEEP_FOV_HALF_DEG = float(_c.SWEEP_FOV_HALF_DEG)
# Alternate Z levels the branch visits when the current slice is blocked.
SWEEP_Z_LADDER_MM = tuple(float(z) for z in _c.SWEEP_Z_LADDER_MM)


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
        # Simple-reactive-mode bookkeeping.
        self._last_reverse_t: float = 0.0

    def _current_z(self) -> float:
        """Z level the sweep is currently running at — index into ladder."""
        idx = int(getattr(self._bb, "sweep_z_index", 0) or 0)
        if 0 <= idx < len(self._z_ladder):
            return self._z_ladder[idx]
        return self._z_default

    # ── Simple replan sweep (demo-safe) ────────────────────────────────────

    def _tof_primary_hit(self) -> tuple[bool, int, float]:
        """Check live ToF readings against per-channel engage thresholds.

        Returns ``(hit, ch, distance_mm)``. Only primary channels 0/1/2
        count — CH3 (ground-facing) is always ignored.
        """
        tof = list(getattr(self._bb, "tof_filtered", []) or [])
        thresholds = list(_c.TOF_ENGAGE_MM)
        hit = False
        hit_ch = -1
        hit_d = float("inf")
        for ch in (0, 1, 2):
            if ch >= len(tof) or ch >= len(thresholds):
                continue
            d = float(tof[ch])
            if not math.isfinite(d):
                continue
            if d < float(thresholds[ch]) and d < hit_d:
                hit = True
                hit_d = d
                hit_ch = ch
        return hit, hit_ch, hit_d

    def _compute_theta_clear(
        self, next_theta: float, direction: int,
    ) -> Optional[float]:
        """First clear θ past the obstacle at the sweep's NOMINAL (r, z) slice.

        The simple-replan sweep always runs on the nominal line, so
        ``θ_clear`` is always queried against ``(SWEEP_R_DEFAULT_MM,
        SWEEP_Z_DEFAULT_MM)`` — never the z-ladder's current rung.
        Chain-skips adjacent blocks separated by <=
        ``2 * SWEEP_SKIP_MARGIN_DEG`` up to ``SWEEP_DETOUR_CHAIN_MAX``
        times. Returns ``None`` if the clear side falls outside [0, 180]
        (caller reverses). Falls back to a default FoV block centred on
        ``next_theta`` when the cloud is empty.
        """
        r = float(self._r)
        z = float(self._z_default)
        fov_half = float(SWEEP_FOV_HALF_DEG)
        skip_margin = float(SWEEP_SKIP_MARGIN_DEG)
        chain_cap = int(getattr(_c, "SWEEP_DETOUR_CHAIN_MAX", 3))

        ranges = self._world.polar_blocked_ranges(
            r, z, fov_half_deg=fov_half,
        )
        # Cloud stale / empty — assume a default block centred on next_theta.
        if not ranges:
            ranges = [(
                max(SWEEP_THETA_MIN, next_theta - fov_half),
                min(SWEEP_THETA_MAX, next_theta + fov_half),
            )]

        theta_probe = float(next_theta)
        for _ in range(max(1, chain_cap)):
            block = _find_block_at(ranges, theta_probe, direction, fov_half)
            if block is None:
                break
            if direction > 0:
                theta_probe = float(block[1]) + skip_margin
            else:
                theta_probe = float(block[0]) - skip_margin

        if theta_probe < SWEEP_THETA_MIN or theta_probe > SWEEP_THETA_MAX:
            return None
        return theta_probe

    def _build_detour_path(
        self,
        theta_cur: float,
        theta_clear: float,
        current_q: list[int],
    ) -> tuple[Optional[list[list[int]]], str]:
        """Parametric radial-pullback detour from θ_cur → θ_clear at fixed z.

        Returns ``(path, shape_tag)``. Tries ``pull_in`` first (``r_dip``
        subtracted at midpoint) — the arm curves INWARD around the
        obstacle. Falls back to ``push_out`` on IK/collision failure.
        Returns ``(None, "")`` if both shapes fail on any waypoint.
        """
        steps = int(getattr(_c, "SWEEP_DETOUR_STEPS", 6))
        r_nominal = float(self._r)
        # Detour ALWAYS ends on the nominal sweep z-line, regardless of
        # where the arm drifted to. Final waypoint = (θ_clear, r_nom, z_nom).
        z_fixed = float(self._z_default)
        r_dip_cfg = float(getattr(_c, "SWEEP_DETOUR_R_DIP_MM", 40.0))
        r_min = float(getattr(_c, "SWEEP_DETOUR_R_MIN_MM", 60.0))
        # Adaptive dip: never pull r below r_min + 30 mm safety margin.
        r_dip = max(10.0, min(r_dip_cfg, r_nominal - r_min - 30.0))

        def _sample(sign: int) -> Optional[list[list[int]]]:
            waypoints: list[list[int]] = []
            for i in range(1, steps + 1):
                t = float(i) / float(steps)  # (0, 1]
                s = 0.5 - 0.5 * math.cos(math.pi * t)
                theta_t = theta_cur + (theta_clear - theta_cur) * s
                r_t = r_nominal + sign * r_dip * math.sin(math.pi * t)
                q = self._ik_target(theta_t, r_t, z_fixed, current_q)
                if q is None:
                    return None
                if self._planner.collision.config_in_collision(q):
                    return None
                waypoints.append(q)
            return waypoints

        path = _sample(sign=-1)     # pull-in
        if path is not None:
            return path, "pull_in"
        path = _sample(sign=+1)     # push-out fallback
        if path is not None:
            return path, "push_out"
        return None, ""

    def _send_reverse(
        self, hold_q: list[int], hit_ch: int, reason: str,
    ) -> py_trees.common.Status:
        """DRY helper: flip direction (respecting cooldown), hold pose, bump
        replan_event. Used when the arm can't detour (edge case, IK fail,
        or mid-detour re-validation failure)."""
        cooldown = float(
            getattr(_c, "SIMPLE_SWEEP_REVERSE_COOLDOWN_S", 0.4)
        )
        now = time.monotonic()
        direction = int(self._bb.sweep_direction) or 1
        if now - self._last_reverse_t >= cooldown:
            direction = -direction
            self._bb.sweep_direction = direction
            self._last_reverse_t = now
        self._bb.sweep_detour_path = []
        self._bb.sweep_detour_idx = 0
        self._bb.replan_event = int(self._bb.replan_event) + 1
        _send_command(self._bb, list(hold_q), delta=3)
        self._bb.bt_state = "sweep_running"
        tag = f"ch{hit_ch}" if hit_ch >= 0 else reason
        self._bb.last_strategy = f"simple_reverse:{tag}"
        self._bb.last_failure = None
        self._bb.polar_snapshot = []
        return py_trees.common.Status.SUCCESS

    def _simple_replan_tick(self) -> py_trees.common.Status:
        """Reactive replan-around sweep — detour on ToF hit, reverse only
        at domain edges or when no safe detour exists.

        State machine:
          1. If a detour is active, re-validate the next waypoint against
             the live cloud and send it. Abort → reverse.
          2. Otherwise, if a primary ToF channel trips, plan a new detour.
             No safe θ_clear → reverse. IK/collision fail → reverse.
          3. Otherwise, advance θ by ``SWEEP_STEP_DEG`` (flip at domain
             edges, same as the prior reactive reverse).
        """
        current_q = list(self._bb.current_q)
        current_theta = float(current_q[0])
        direction = int(self._bb.sweep_direction) or 1

        # ── Phase 1: active detour playback ────────────────────────────
        path: list[list[int]] = list(
            getattr(self._bb, "sweep_detour_path", []) or []
        )
        if path:
            idx = int(getattr(self._bb, "sweep_detour_idx", 0) or 0)
            if 0 <= idx < len(path):
                target = list(path[idx])
                if self._planner.collision.config_in_collision(target):
                    return self._send_reverse(
                        current_q, -1, "detour_revalidate",
                    )
                _send_command(self._bb, target, delta=3)
                self._bb.sweep_detour_idx = idx + 1
                if idx + 1 >= len(path):
                    self._bb.sweep_detour_path = []
                    self._bb.sweep_detour_idx = 0
                self._bb.bt_state = "sweep_running"
                # Strategy tag persists through playback (set at entry).
                self._bb.last_failure = None
                self._bb.polar_snapshot = []
                return py_trees.common.Status.SUCCESS
            # idx out of range — clear and fall through as if no detour.
            self._bb.sweep_detour_path = []
            self._bb.sweep_detour_idx = 0

        # ── Phase 2: ToF trigger → new detour ──────────────────────────
        hit, hit_ch, _hit_d = self._tof_primary_hit()
        if hit:
            next_theta = current_theta + direction * self._step
            # Sanity: clamp next_theta for θ_clear search.
            probe = max(SWEEP_THETA_MIN, min(SWEEP_THETA_MAX, next_theta))
            theta_clear = self._compute_theta_clear(probe, direction)
            if theta_clear is None:
                return self._send_reverse(
                    current_q, hit_ch, "edge_no_detour",
                )
            if abs(theta_clear - current_theta) < 1e-3:
                # False positive — world model disagrees with ToF. Just
                # hold this tick (don't reverse, don't advance into it).
                _send_command(self._bb, current_q, delta=3)
                self._bb.bt_state = "sweep_running"
                self._bb.last_strategy = f"simple_hold:ch{hit_ch}"
                self._bb.last_failure = None
                self._bb.polar_snapshot = []
                return py_trees.common.Status.SUCCESS
            new_path, shape_tag = self._build_detour_path(
                current_theta, theta_clear, current_q,
            )
            if new_path is None:
                return self._send_reverse(
                    current_q, hit_ch, "ik_or_collision",
                )
            self._bb.sweep_detour_path = list(new_path)
            self._bb.sweep_detour_idx = 1
            first_q = list(new_path[0])
            _send_command(self._bb, first_q, delta=3)
            self._bb.sweep_target_deg = float(theta_clear)
            self._bb.bt_state = "sweep_running"
            self._bb.last_strategy = f"sweep_detour:{shape_tag}"
            self._bb.last_failure = None
            self._bb.polar_snapshot = []
            self._bb.replan_event = int(self._bb.replan_event) + 1
            return py_trees.common.Status.SUCCESS

        # ── Phase 3: normal advance on the NOMINAL sweep line ──────────
        #
        # The simple-replan sweep owns its own (r, z) line — pressing `Z`
        # snaps the arm onto (SWEEP_R_DEFAULT_MM, SWEEP_Z_DEFAULT_MM) and
        # stays there forever. Every advance tick IK's to the nominal
        # line, so after any detour (or any manual pose the arm happened
        # to be in when sweep was engaged), the arm self-corrects back to
        # the sweep plane within one or two ticks (RateClamp slews the
        # transition smoothly).
        next_theta = current_theta + direction * self._step
        if next_theta < SWEEP_THETA_MIN or next_theta > SWEEP_THETA_MAX:
            direction = -direction
            self._bb.sweep_direction = direction
            next_theta = current_theta + direction * self._step
            next_theta = max(SWEEP_THETA_MIN, min(SWEEP_THETA_MAX, next_theta))

        nominal_q = self._ik_target(
            float(next_theta), float(self._r), float(self._z_default),
            current_q,
        )
        if nominal_q is not None:
            target_q = list(nominal_q)
        else:
            # IK shouldn't fail on the nominal line, but if it does,
            # keep sweeping rather than freezing — fall back to base-only.
            target_q = [int(round(next_theta))] + current_q[1:]
        _send_command(self._bb, target_q, delta=3)
        self._bb.sweep_target_deg = float(next_theta)
        self._bb.bt_state = "sweep_running"
        self._bb.last_strategy = "simple_direct"
        self._bb.last_failure = None
        self._bb.polar_snapshot = []
        return py_trees.common.Status.SUCCESS

    def update(self) -> py_trees.common.Status:
        if self._bb.mode != "sweep":
            return py_trees.common.Status.FAILURE

        self._world.decay()

        # Demo-safe path: on a live ToF hit, parametrise a smooth radial
        # pullback detour around the obstacle, stream it one waypoint per
        # tick, and resume the sweep on the far side. Only reverses when
        # the clear-side θ falls outside [0, 180]. See SIMPLE_REPLAN_MODE
        # docstring in constants.py.
        if bool(getattr(_c, "SIMPLE_REPLAN_MODE", False)):
            return self._simple_replan_tick()

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
                # margin lands outside the domain. Try escalation via the
                # obstacle-aware z-ladder picker (see _pick_safe_z_rung).
                z_tries = int(
                    getattr(self._bb, "sweep_z_tries", 0) or 0
                )
                visited = list(
                    getattr(self._bb, "sweep_z_visited", []) or []
                )
                obs_z_extent = self._world.obstacle_z_extent(
                    self._r, float(next_theta),
                    fov_half_deg=SWEEP_FOV_HALF_DEG,
                )
                next_rung = (
                    self._pick_safe_z_rung(obs_z_extent, visited)
                    if z_tries < len(self._z_ladder) - 1
                    else None
                )
                if next_rung is not None:
                    cur_idx = int(
                        getattr(self._bb, "sweep_z_index", 0) or 0
                    )
                    self._bb.sweep_z_visited = list(
                        sorted({*visited, cur_idx})
                    )
                    self._bb.sweep_z_index = next_rung
                    self._polar.clear()
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
                        self._bb.sweep_z_visited = []
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
                self._bb.sweep_z_visited = []
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
        self._bb.sweep_z_visited = []

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
        """Populate the polar map from world-cloud points that actually sit
        near the arm's CURRENT gripper position, not the canonical
        (r=SWEEP_R_DEFAULT_MM, z=SWEEP_Z_DEFAULT_MM) slice.

        Wrist-mounted ToF sensors project points near the wrist's real
        world-frame position — if the user reconfigures the arm (wrist
        tilt, different elbow, custom z), those points land at a
        completely different (r, z) slice than the sweep default. The old
        query missed them entirely and the BT never saw a block, which is
        why putting a hand over the ToF produced a WARN but no replan.

        We query at the gripper tip (last capsule) — that's where the
        sensors are mounted and where obstacles naturally project. Falls
        back to the ladder-defined (r, z) if FK fails for any reason.
        """
        try:
            caps = self._planner.fk.link_endpoints(list(self._bb.current_q))
            tip = caps[-1].p1
            r_world = math.hypot(tip[0], tip[1])
            z_world = tip[2]
        except Exception:
            r_world = self._r
            z_world = self._current_z()
        ranges = self._world.polar_blocked_ranges(
            r_world, z_world, fov_half_deg=SWEEP_FOV_HALF_DEG,
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

    def _pick_safe_z_rung(
        self,
        obs_z_extent: Optional[tuple[float, float]],
        visited: list[int],
    ) -> Optional[int]:
        """Pick the best unvisited ladder rung given the obstacle's Z span.

        Returns ``None`` if the obstacle is too tall for the ladder to
        escape (vertical column) or every remaining rung is already
        visited. Otherwise returns the rung index with the most clearance
        from the obstacle — which is why an arm at the bottom rung with an
        obstacle overhead picks a higher rung ("go up instead of further
        down"), addressing the session_20260417_020606 bottom-of-ladder
        trap.
        """
        cur = int(getattr(self._bb, "sweep_z_index", 0) or 0)
        visited_set = set(int(i) for i in (visited or []))
        visited_set.add(cur)

        remaining = [
            i for i in range(len(self._z_ladder)) if i not in visited_set
        ]
        if not remaining:
            return None

        if obs_z_extent is None:
            # No obstacle info in the blocked slice — fall back to the
            # closest unvisited rung (same behaviour as linear advance).
            cur_z = self._z_ladder[cur]
            remaining.sort(key=lambda i: abs(self._z_ladder[i] - cur_z))
            return remaining[0]

        obs_z_min, obs_z_max = obs_z_extent
        obs_span = float(obs_z_max - obs_z_min)
        ladder_span = float(max(self._z_ladder) - min(self._z_ladder))
        # A column of points spanning ≥60% of the ladder cannot be
        # overtaken by any rung — kick up to BiRRT / direction flip.
        if ladder_span > 0 and obs_span > 0.6 * ladder_span:
            return None

        # Prefer rungs OUTSIDE the obstacle envelope (with a small margin),
        # sorted by distance from the obstacle's centre — "furthest from
        # the blocker wins", which cleanly picks "up" when the obstacle is
        # low and "down" when it's high.
        margin_mm = 20.0
        obs_mid = 0.5 * (obs_z_min + obs_z_max)
        outside = [
            i for i in remaining
            if self._z_ladder[i] < obs_z_min - margin_mm
            or self._z_ladder[i] > obs_z_max + margin_mm
        ]
        if outside:
            outside.sort(key=lambda i: abs(self._z_ladder[i] - obs_mid),
                         reverse=True)
            return outside[0]

        # Every remaining rung is inside the obstacle envelope — nothing
        # the z-ladder can do; escalate.
        return None

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
