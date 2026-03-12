"""Linear-relaxed replanner in polar task space (theta, r, z).

Tuning location:
- `RelaxedWeights` controls tracking/smoothing/avoidance weighting.
- `RelaxedLimits` controls velocity caps and avoidance retreats.

The obstacle constraint is a linearized distance half-space around the current
wrist point, projected with a weighted metric (no heavy solver required).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _polar_point_m(theta_deg: float, r_mm: float, z_mm: float) -> np.ndarray:
    th = np.deg2rad(theta_deg)
    return np.array([r_mm * np.cos(th), r_mm * np.sin(th), z_mm], dtype=float) / 1000.0


def _distance_gradient(theta_deg: float, r_mm: float, z_mm: float, obstacle_point_m: np.ndarray) -> tuple[np.ndarray, float]:
    th = np.deg2rad(theta_deg)
    p = _polar_point_m(theta_deg, r_mm, z_mm)
    diff = p - obstacle_point_m

    # Derivatives wrt [theta_deg, r_mm, z_mm]
    dp_dtheta = np.array([-r_mm * np.sin(th), r_mm * np.cos(th), 0.0], dtype=float) * (np.pi / 180.0) / 1000.0
    dp_dr = np.array([np.cos(th), np.sin(th), 0.0], dtype=float) / 1000.0
    dp_dz = np.array([0.0, 0.0, 1.0], dtype=float) / 1000.0

    grad = np.array(
        [
            2.0 * float(np.dot(diff, dp_dtheta)),
            2.0 * float(np.dot(diff, dp_dr)),
            2.0 * float(np.dot(diff, dp_dz)),
        ],
        dtype=float,
    )
    g0 = float(np.dot(diff, diff))
    return grad, g0


@dataclass(frozen=True)
class RelaxedWeights:
    # Tracking/smoothing terms.
    w_track_theta: float = 3.0
    w_smooth_theta: float = 1.5

    w_track_r: float = 2.2
    w_smooth_r: float = 2.0
    w_avoid_r: float = 4.0

    w_track_z: float = 2.2
    w_smooth_z: float = 2.0
    w_avoid_z: float = 4.0


@dataclass(frozen=True)
class RelaxedLimits:
    # Velocity limits (units per second).
    theta_vel_nominal_deg_s: float = 55.0
    theta_vel_avoid_deg_s: float = 22.0

    r_vel_nominal_mm_s: float = 120.0
    r_vel_avoid_mm_s: float = 55.0

    z_vel_nominal_mm_s: float = 110.0
    z_vel_avoid_mm_s: float = 50.0

    # Avoidance geometry relaxation amplitudes.
    replan_retreat_mm: float = 28.0
    replan_lift_mm: float = 55.0
    backaway_retreat_mm: float = 38.0
    backaway_lift_mm: float = 70.0


class LinearRelaxedReplanner:
    """Lightweight replanner for smooth obstacle-aware polar commanding."""

    def __init__(self, weights: RelaxedWeights | None = None, limits: RelaxedLimits | None = None):
        self.weights = weights or RelaxedWeights()
        self.limits = limits or RelaxedLimits()

    def step(
        self,
        desired_theta: float,
        desired_r: float,
        desired_z: float,
        prev_theta: float,
        prev_r: float,
        prev_z: float,
        obstacle_response: str,
        obstacle_dist_mm: float,
        obstacle_threshold_mm: float,
        dt_s: float,
        r_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
        obstacle_point_m: np.ndarray | None = None,
        endpoint_lock: bool = False,
    ) -> tuple[float, float, float, dict]:
        dt = max(1e-3, dt_s)

        severity = 0.0
        mode = "clear"
        avoid_r = desired_r
        avoid_z = desired_z

        if obstacle_response == "back_away":
            severity = 1.0
            mode = "back_away"
            avoid_r = desired_r - self.limits.backaway_retreat_mm
            avoid_z = desired_z + self.limits.backaway_lift_mm
        elif obstacle_point_m is not None and obstacle_threshold_mm > 1e-6:
            if not np.isfinite(obstacle_dist_mm) or obstacle_dist_mm < 0.0:
                cur = _polar_point_m(prev_theta, prev_r, prev_z)
                obstacle_dist_mm = float(np.linalg.norm(cur - obstacle_point_m) * 1000.0)
            d = max(0.0, obstacle_dist_mm)
            severity = _clamp((obstacle_threshold_mm - d) / obstacle_threshold_mm, 0.0, 1.0)
            if severity > 0.0:
                mode = "replan"
                avoid_r = desired_r - self.limits.replan_retreat_mm * severity
                avoid_z = desired_z + self.limits.replan_lift_mm * severity

        avoid_r = _clamp(avoid_r, r_bounds[0], r_bounds[1])
        avoid_z = _clamp(avoid_z, z_bounds[0], z_bounds[1])

        wt = self.weights
        theta_target = (
            wt.w_track_theta * desired_theta + wt.w_smooth_theta * prev_theta
        ) / (wt.w_track_theta + wt.w_smooth_theta)

        r_target = (
            wt.w_track_r * desired_r + wt.w_smooth_r * prev_r + wt.w_avoid_r * avoid_r
        ) / (wt.w_track_r + wt.w_smooth_r + wt.w_avoid_r)

        z_target = (
            wt.w_track_z * desired_z + wt.w_smooth_z * prev_z + wt.w_avoid_z * avoid_z
        ) / (wt.w_track_z + wt.w_smooth_z + wt.w_avoid_z)

        lim = self.limits
        if severity > 0.0:
            th_v = lim.theta_vel_avoid_deg_s
            r_v = lim.r_vel_avoid_mm_s
            z_v = lim.z_vel_avoid_mm_s
            speed_scale = 1.0 - 0.75 * severity
        else:
            th_v = lim.theta_vel_nominal_deg_s
            r_v = lim.r_vel_nominal_mm_s
            z_v = lim.z_vel_nominal_mm_s
            speed_scale = 1.0

        th_step = th_v * dt
        r_step = r_v * dt
        z_step = z_v * dt

        theta_lo = max(0.0, prev_theta - th_step)
        theta_hi = min(180.0, prev_theta + th_step)
        r_lo = max(r_bounds[0], prev_r - r_step)
        r_hi = min(r_bounds[1], prev_r + r_step)
        z_lo = max(z_bounds[0], prev_z - z_step)
        z_hi = min(z_bounds[1], prev_z + z_step)

        q = np.array(
            [
                _clamp(theta_target, theta_lo, theta_hi),
                _clamp(r_target, r_lo, r_hi),
                _clamp(z_target, z_lo, z_hi),
            ],
            dtype=float,
        )

        # Linearized half-space projection for obstacle clearance.
        if obstacle_point_m is not None and severity > 0.0 and mode == "replan":
            q0 = np.array([prev_theta, prev_r, prev_z], dtype=float)
            grad, g0 = _distance_gradient(prev_theta, prev_r, prev_z, np.asarray(obstacle_point_m, dtype=float))
            safe_d_m = max(0.04, obstacle_threshold_mm / 1000.0)
            b = safe_d_m * safe_d_m - g0 + float(np.dot(grad, q0))
            lhs = float(np.dot(grad, q))

            if lhs < b and float(np.linalg.norm(grad)) > 1e-12:
                # Weighted metric projection (W^-1).
                winv = np.array(
                    [
                        1.0 / max(1e-6, wt.w_track_theta + wt.w_smooth_theta),
                        1.0 / max(1e-6, wt.w_track_r + wt.w_smooth_r + wt.w_avoid_r),
                        1.0 / max(1e-6, wt.w_track_z + wt.w_smooth_z + wt.w_avoid_z),
                    ],
                    dtype=float,
                )
                wg = winv * grad
                denom = float(np.dot(grad, wg))
                if denom > 1e-12:
                    q = q + ((b - lhs) / denom) * wg

        q[0] = _clamp(float(q[0]), theta_lo, theta_hi)
        q[1] = _clamp(float(q[1]), r_lo, r_hi)
        q[2] = _clamp(float(q[2]), z_lo, z_hi)

        if endpoint_lock:
            q[1] = desired_r
            q[2] = desired_z

        debug = {
            "mode": mode,
            "severity": float(severity),
            "speed_scale": float(speed_scale),
            "theta_target": float(theta_target),
            "r_target": float(r_target),
            "z_target": float(z_target),
            "avoid_r": float(avoid_r),
            "avoid_z": float(avoid_z),
            "obstacle_dist_mm": float(obstacle_dist_mm),
        }
        return float(q[0]), float(q[1]), float(q[2]), debug
