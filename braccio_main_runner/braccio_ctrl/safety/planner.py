"""
planner.py — BiRRT in joint space + path smoothing.

Thin wrapper around ``pybullet_planning.birrt`` that hides the distance /
sample / extend function plumbing and presents a single ``plan()`` entry
point. The heavy lifting lives in pybullet-planning (research §2 top pick);
we only supply Braccio-specific callables:

  * distance_fn   weighted sum of |Δdeg| per joint (shoulder/elbow weighted 2×)
  * sample_fn     uniform random sample inside the Braccio's JOINT_LIMITS
  * extend_fn     linear interpolation in joint space at a fixed step (5°),
                  matching the Arduino's ``Braccio.move()`` interpolation
  * collision_fn  from ``CollisionChecker.collision_fn()``

On failure the planner distinguishes ``birrt_timeout`` from ``ik_unreachable``
so the CascadingReplanner can pick the right recovery strategy.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import numpy as np

from pybullet_planning import birrt, smooth_path  # type: ignore[import-not-found]

from .collision import CollisionChecker
from .fk import BRACCIO_JOINT_NAMES, ForwardKinematics


# ── Tunables (mirrored in constants.py Phase 4) ────────────────────────────
#
# Kept here as module defaults so the planner is usable in tests without
# pulling in controller/constants. constants.py will re-export the same names
# once Phase 4 wires the stack into the controller.

BIRRT_TIMEOUT_S_DEFAULT = 2.0
BIRRT_EXTEND_DEG_DEFAULT = 5.0
SMOOTH_MAX_ITERS_DEFAULT = 50
SMOOTH_TIMEOUT_S_DEFAULT = 0.5

# Per-joint distance weighting. Shoulder/elbow carry the arm's mass and sweep
# the biggest Cartesian arcs per degree; penalise those more so the planner
# prefers wrist-driven detours when both are feasible.
JOINT_DISTANCE_WEIGHTS = np.array([1.0, 2.0, 2.0, 1.0, 1.0, 0.0])  # gripper irrelevant

# Joint limits — taken verbatim from braccio_ctrl.constants.JOINT_LIMITS.
# Gripper (index 5) is fixed in the URDF and ignored during planning.
_JOINT_LIMITS = [
    (0,   180),  # base
    (15,  165),  # shoulder
    (0,   180),  # elbow
    (0,   180),  # wrist vertical
    (0,   180),  # wrist rotation
    (10,   73),  # gripper (passed through untouched)
]


# ── PlanResult ─────────────────────────────────────────────────────────────

@dataclass
class PlanResult:
    """Outcome of a single ``SafetyPlanner.plan()`` call.

    On success, ``path`` is a list of joint-degree configurations starting at
    ``start_q`` and ending at ``goal_q`` (inclusive). On failure ``path`` is
    ``None`` and ``failure_reason`` names the first stage that gave up.
    """
    success: bool
    path: Optional[list[list[int]]] = None
    failure_reason: Optional[str] = None
    planner_time_s: float = 0.0
    num_waypoints: int = 0

    def __bool__(self) -> bool:
        return self.success


# ── SafetyPlanner ──────────────────────────────────────────────────────────

class SafetyPlanner:
    """BiRRT planner in joint space with capsule-based collision checking.

    Thread-safety: the planner is stateless between calls — concurrent calls
    from the BT and curses thread are safe as long as each gets its own
    CollisionChecker / WorldModel references (which are themselves locked).
    """

    def __init__(
        self,
        collision: CollisionChecker,
        fk: ForwardKinematics,
        extend_step_deg: float = BIRRT_EXTEND_DEG_DEFAULT,
        default_timeout_s: float = BIRRT_TIMEOUT_S_DEFAULT,
        rng: Optional[random.Random] = None,
    ):
        self._collision = collision
        self._fk = fk
        self._extend_step = float(extend_step_deg)
        self._default_timeout = float(default_timeout_s)
        self._rng = rng or random.Random()

    @property
    def collision(self) -> CollisionChecker:
        """Public accessor for the checker so branches can call
        ``config_in_collision`` without reaching into a private attribute.
        """
        return self._collision

    # ── Public API ─────────────────────────────────────────────────────────

    def plan(
        self,
        start_q: list[int],
        goal_q: list[int],
        timeout_s: Optional[float] = None,
        smooth: bool = True,
    ) -> PlanResult:
        """Plan a collision-free joint-space path from start_q → goal_q.

        Returns ``PlanResult(success=True, path=[...])`` or a failure result
        with a diagnostic ``failure_reason`` one of:

            ``start_in_collision``
            ``goal_in_collision``
            ``goal_out_of_limits``
            ``birrt_timeout``
        """
        t_start = time.monotonic()
        timeout = float(self._default_timeout if timeout_s is None else timeout_s)

        # Boundary checks — cheaper to validate up front than to hand to BiRRT.
        if not self._in_joint_limits(goal_q):
            return PlanResult(False, None, "goal_out_of_limits",
                              time.monotonic() - t_start)
        if self._collision.config_in_collision(start_q):
            return PlanResult(False, None, "start_in_collision",
                              time.monotonic() - t_start)
        if self._collision.config_in_collision(goal_q):
            return PlanResult(False, None, "goal_in_collision",
                              time.monotonic() - t_start)

        # Trivial case — segment itself is clear. Skip BiRRT entirely.
        if not self._collision.path_in_collision(start_q, goal_q):
            path = [list(map(int, start_q)), list(map(int, goal_q))]
            return PlanResult(True, path, None,
                              time.monotonic() - t_start, len(path))

        dist_fn = self._distance_fn
        sample_fn = self._sample_fn
        extend_fn = partial(self._extend_fn, step_deg=self._extend_step)
        coll_fn = self._collision.collision_fn()

        raw = birrt(
            list(map(int, start_q)),
            list(map(int, goal_q)),
            dist_fn, sample_fn, extend_fn, coll_fn,
            max_time=timeout,
        )
        if raw is None:
            return PlanResult(False, None, "birrt_timeout",
                              time.monotonic() - t_start)

        path = [list(map(int, q)) for q in raw]

        if smooth and len(path) > 2:
            try:
                smoothed = smooth_path(
                    path, extend_fn, coll_fn,
                    distance_fn=dist_fn,
                    max_smooth_iterations=SMOOTH_MAX_ITERS_DEFAULT,
                    max_time=SMOOTH_TIMEOUT_S_DEFAULT,
                    coarse_waypoints=True,
                )
                if smoothed is not None and len(smoothed) >= 2:
                    path = [list(map(int, q)) for q in smoothed]
            except Exception:
                # smooth_path is best-effort; the unsmoothed path is still valid.
                pass

        return PlanResult(True, path, None,
                          time.monotonic() - t_start, len(path))

    def escape_from_collision(
        self,
        current_q: list[int],
        home_q: Optional[list[int]] = None,
        max_steps_deg: int = 40,
    ) -> Optional[list[int]]:
        """Find a nearby collision-free config when the arm is currently in
        collision (typically because a fresh ToF point was ingested inside
        the arm's capsule volume). Returns the first safe candidate or
        ``None`` if every probed direction remains blocked.

        The search tries a short ladder of per-axis retreats plus a
        "toward-HOME" retreat, in order of smallest joint-space delta. We
        deliberately keep the search cheap (~20 probes, no BiRRT) because
        it runs every tick when the world model has a freshly-ingested
        arm-self point; we just need *a* safe step, not the optimal one.
        """
        cur = [int(round(float(x))) for x in current_q]
        home = list(home_q) if home_q is not None else [90, 90, 90, 90, 90, cur[5]]
        if not self._collision.config_in_collision(cur):
            return cur

        candidates: list[list[int]] = []
        # 1. Per-axis nudges across a wide range of magnitudes. The arm-self
        # exclusion sphere can be 80–120 mm wide, so small 5° moves often
        # can't clear it — escalate up to ±40° on the big joints.
        steps = (5, 10, 15, 20, 25, 30, 35, 40)
        for idx in (1, 2, 3, 0, 4):
            for s in steps:
                for sign in (+1, -1):
                    q = list(cur)
                    q[idx] = max(
                        _JOINT_LIMITS[idx][0],
                        min(_JOINT_LIMITS[idx][1], q[idx] + sign * s),
                    )
                    candidates.append(q)

        # 2. Interpolate toward HOME at a few fractions (the retreat-to-safe
        # fallback — always reachable for the Braccio).
        for t in (0.15, 0.3, 0.5, 0.75, 1.0):
            q = [
                int(round((1.0 - t) * cur[i] + t * home[i])) for i in range(6)
            ]
            candidates.append(q)

        # 3. Pairwise shoulder+elbow retreats. Some poses need both joints to
        # move together to clear the obstacle sphere.
        for ds, de in ((+15, +15), (+15, -15), (-15, +15), (-15, -15),
                       (+25, +25), (-25, -25), (+30, -30), (-30, +30)):
            q = list(cur)
            q[1] = max(_JOINT_LIMITS[1][0], min(_JOINT_LIMITS[1][1], q[1] + ds))
            q[2] = max(_JOINT_LIMITS[2][0], min(_JOINT_LIMITS[2][1], q[2] + de))
            candidates.append(q)

        # We ACCEPT that the path from cur→q may still be in collision for the
        # first interpolated samples — by definition ``cur`` itself collides.
        # The whole point of escape is to move *out* of that state, so we only
        # require the TARGET to be clear. Pick the shortest-distance candidate
        # so the arm barely moves past the obstacle region.
        best: Optional[list[int]] = None
        best_cost = float("inf")
        for q in candidates:
            q = [
                max(_JOINT_LIMITS[i][0], min(_JOINT_LIMITS[i][1], int(q[i])))
                for i in range(6)
            ]
            if q == cur:
                continue
            if self._collision.config_in_collision(q):
                continue
            # Require the *end* of the transit to be clear for a handful of
            # consecutive interp samples — that way the arm enters free
            # space before the command ends, rather than just grazing a
            # clear pose for an instant.
            samples_ok = True
            for t_check in (0.6, 0.8, 1.0):
                q_check = [
                    int(round((1.0 - t_check) * cur[i] + t_check * q[i]))
                    for i in range(6)
                ]
                if self._collision.config_in_collision(q_check):
                    samples_ok = False
                    break
            if not samples_ok:
                continue
            cost = float(np.sum(
                JOINT_DISTANCE_WEIGHTS * np.abs(
                    np.asarray(q, dtype=np.float64)
                    - np.asarray(cur, dtype=np.float64)
                )
            ))
            if cost < best_cost:
                best = q
                best_cost = cost
        return best

    def plan_cartesian(
        self,
        start_q: list[int],
        goal_xyz_mm: tuple[float, float, float],
        current_wrist_v: int = 90,
        current_wrist_r: int = 90,
        current_gripper: int = 73,
        timeout_s: Optional[float] = None,
    ) -> PlanResult:
        """Plan to a Cartesian gripper-tip target via the existing IK solver.

        Delegates IK to ``braccio_ctrl.ik_solver.solve_ik``. Returns
        ``failure_reason='ik_unreachable'`` when the target is outside the
        arm's dexterous workspace.
        """
        # Lazy import to avoid circular dependency when running the safety
        # stack in isolation.
        from ..ik_solver import solve_ik

        x, y, z = goal_xyz_mm
        sol = solve_ik(float(x), float(y), float(z))
        if sol is None:
            return PlanResult(False, None, "ik_unreachable", 0.0)
        goal_q = [
            int(sol.base),
            int(sol.shoulder),
            int(sol.elbow),
            int(sol.wrist_vert + current_wrist_v - 90),
            int(current_wrist_r),
            int(current_gripper),
        ]
        return self.plan(start_q, goal_q, timeout_s=timeout_s)

    # ── BiRRT callables ────────────────────────────────────────────────────

    @staticmethod
    def _distance_fn(a, b) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        return float(np.sum(JOINT_DISTANCE_WEIGHTS * np.abs(aa - bb)))

    def _sample_fn(self) -> list[int]:
        return [self._rng.randint(lo, hi) for lo, hi in _JOINT_LIMITS]

    @staticmethod
    def _extend_fn(q_from, q_to, step_deg: float) -> list[list[int]]:
        a = np.asarray(q_from, dtype=np.float64)
        b = np.asarray(q_to, dtype=np.float64)
        diff = b - a
        max_step = float(np.max(np.abs(diff)))
        if max_step < 1e-9:
            return [list(map(int, q_to))]
        n_steps = max(1, int(math.ceil(max_step / float(step_deg))))
        out: list[list[int]] = []
        for k in range(1, n_steps + 1):
            t = k / float(n_steps)
            q = np.rint(a + t * diff).astype(int).tolist()
            out.append(q)
        return out

    @staticmethod
    def _in_joint_limits(q) -> bool:
        return all(
            lo <= int(round(v)) <= hi
            for (lo, hi), v in zip(_JOINT_LIMITS, q)
        )
