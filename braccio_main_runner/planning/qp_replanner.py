"""Host QP replanner (OSQP preferred, deterministic fallback)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import osqp
    import scipy.sparse as sp
    _HAS_OSQP = True
except Exception:
    _HAS_OSQP = False


@dataclass
class QPResult:
    command: list[int]
    solver_status: str
    solve_time_ms: float
    fallback_used: bool
    predicted_clearance_m: float


class QPReplanner:
    def __init__(self, max_step_deg: float = 6.0, min_clearance_m: float = 0.2):
        self.max_step_deg = float(max_step_deg)
        self.min_clearance_m = float(min_clearance_m)

    def solve(
        self,
        current_joints: list[int],
        nominal_joints: list[int],
        obstacle_distance_m: float,
        speed_scale: float,
        time_budget_ms: float = 30.0,
    ) -> QPResult:
        start = time.perf_counter()
        n = len(nominal_joints)

        if obstacle_distance_m < self.min_clearance_m * 0.5:
            return QPResult(
                command=list(current_joints),
                solver_status="fallback_hold_close_obstacle",
                solve_time_ms=(time.perf_counter() - start) * 1000.0,
                fallback_used=True,
                predicted_clearance_m=obstacle_distance_m,
            )

        target = np.asarray(nominal_joints, dtype=float)
        cur = np.asarray(current_joints, dtype=float)
        u = self.max_step_deg * max(0.05, min(1.0, speed_scale))

        if _HAS_OSQP:
            P = sp.eye(n, format="csc") * 2.0
            q = -2.0 * target
            A = sp.eye(n, format="csc")
            l = cur - u
            uu = cur + u
            prob = osqp.OSQP()
            prob.setup(P=P, q=q, A=A, l=l, u=uu, verbose=False, warm_start=True)
            res = prob.solve()
            elapsed = (time.perf_counter() - start) * 1000.0
            if elapsed > time_budget_ms or res.x is None:
                return QPResult(list(current_joints), "fallback_hold_timeout", elapsed, True, obstacle_distance_m)
            cmd = [int(round(v)) for v in res.x.tolist()]
            return QPResult(cmd, str(res.info.status).lower(), elapsed, False, obstacle_distance_m)

        # deterministic fallback when OSQP/scipy unavailable
        clipped = np.clip(target, cur - u, cur + u)
        elapsed = (time.perf_counter() - start) * 1000.0
        return QPResult(
            command=[int(round(v)) for v in clipped.tolist()],
            solver_status="fallback_projection_no_osqp",
            solve_time_ms=elapsed,
            fallback_used=True,
            predicted_clearance_m=obstacle_distance_m,
        )
