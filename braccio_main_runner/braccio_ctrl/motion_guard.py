"""
motion_guard.py — Shared obstacle-avoidance gate for every arm motion.

Every code path that moves the end-effector XYZ — manual IK keys, saved
states, sequence editor steps, HOME, and the autonomous sweep — routes
through MotionGuard.plan_clear_pose() before the SET ALL command is built.

The guard has a single authoritative notion of "is this pose clear":
a sphere of SWEEP_COLLISION_RADIUS_MM at the target (theta, r, z),
checked against the persistent voxel memory first (O(k)) and the
rolling cloud as a fallback (O(N)).

plan_clear_pose() walks a deterministic fallback ladder, assuming the
obstacle is stationary (per the persistent voxel memory semantics):

  1. Desired pose (clear target AND clear swept path)
  2. Z-hop      — retry at each SWEEP_Z_CANDIDATES level
  3. Theta nudge — ±GUARD_THETA_NUDGE_STEPS_DEG, z-hop retried at each
  4. Radial retreat — pull the tip closer to the base by GUARD_R_RETREAT_STEPS_MM
  5. None — caller rejects the command and records last_error

The is_clear / find_clear_z primitives are the same logic that AutoSweeper
used to carry internally; they live here now so the sweeper and the
controller share a single implementation.

Live-sensor gate
----------------
Voxel memory is built from where the on-arm ToF sensors *have been*
pointed. The arm's forearm and wrist extend well beyond the 80 mm sphere
at the end-effector, so a hand sitting in front of the arm can remain
outside the destination sphere (and therefore look "clear" to the voxel
check) while still being physically in the path of the arm structure.

To close that gap, plan_clear_pose consults the live ToF snapshot on
every call. When a primary (REPLAN-authority) channel is actively
reporting dist < threshold, the guard refuses any command that isn't an
unambiguous retreat: only radial retraction (target.r < current.r) and
meaningful z-changes (|Δz| > Z_STEP) are allowed. All other motions are
rejected with "blocked by live sensor" until the sensor clears. This
runs on every command so the obstacle reaction is synchronous with the
user's keystroke — not dependent on the voxel memory catching up.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from .ik_solver import polar_to_cartesian, reachability
from .tof_sensor import ObstacleResponse
from .constants import (
    SWEEP_COLLISION_RADIUS_MM,
    SWEEP_Z_CANDIDATES,
    GUARD_PATH_SAMPLES,
    GUARD_THETA_NUDGE_STEPS_DEG,
    GUARD_R_RETREAT_STEPS_MM,
    SWEEP_THETA_MIN, SWEEP_THETA_MAX,
    R_MIN, R_MAX, Z_MIN, Z_MAX,
    Z_STEP,
    SENSOR_REPLAN_CHANNELS,
)


Pose = Tuple[float, float, float]   # (theta_deg, r_mm, z_mm)


class MotionGuard:
    """
    Collision-aware motion gate shared by AutoSweeper and BraccioController.

    The guard holds a reference to the ObstacleMap (which owns the voxel
    memory and the rolling cloud) and exposes query primitives plus a
    deterministic replanning ladder.

    If `tof_state` is supplied, plan_clear_pose also consults the live
    ToF snapshot on every call — see "Live-sensor gate" in the module
    docstring for why this is necessary.
    """

    def __init__(self, obstacle_map, tof_state=None) -> None:
        self._map = obstacle_map
        self._tof_state = tof_state
        self._radius = float(SWEEP_COLLISION_RADIUS_MM)

    # ── Primitive: sphere-at-pose clearance ──────────────────────────────

    def is_clear(self, theta: float, r: float, z: float) -> bool:
        """
        Return True if an end-effector sphere at (theta, r, z) is clear of
        every known obstacle.

        Fast path : persistent voxel memory (O(k))
        Fallback  : rolling point cloud (O(N))
        """
        try:
            x, y = polar_to_cartesian(theta, r)
        except Exception:
            return True   # unevaluable → don't block

        pos = (float(x), float(y), float(z))
        radius = self._radius

        try:
            if self._map.memory_occupied_near(pos, radius):
                return False
        except Exception:
            pass

        try:
            import numpy as np
            pts = self._map.all_points()
            if pts is not None and len(pts) > 0:
                dx = pts[:, 0] - pos[0]
                dy = pts[:, 1] - pos[1]
                dz = pts[:, 2] - pos[2]
                if np.any(dx * dx + dy * dy + dz * dz < radius * radius):
                    return False
        except Exception:
            pass

        return True

    # ── Primitive: z-hop search ──────────────────────────────────────────

    def find_clear_z(self, theta: float, r: float,
                     current_z: float) -> Optional[float]:
        """
        Iterate SWEEP_Z_CANDIDATES (above current first, then below) and
        return the first z that is both kinematically reachable at
        (theta, r) and passes is_clear(). None if nothing clears.
        """
        above = sorted(z for z in SWEEP_Z_CANDIDATES if z > current_z + 1e-3)
        below = sorted(
            (z for z in SWEEP_Z_CANDIDATES if z < current_z - 1e-3),
            reverse=True,
        )
        for z in list(above) + list(below):
            if z < Z_MIN or z > Z_MAX:
                continue
            try:
                if reachability(theta, r, z) != 'ok':
                    continue
            except Exception:
                continue
            if self.is_clear(theta, r, z):
                return float(z)
        return None

    # ── Swept-volume path check ──────────────────────────────────────────

    def is_clear_path(self, from_pose: Pose, to_pose: Pose,
                      samples: int = GUARD_PATH_SAMPLES) -> bool:
        """
        Sample `samples` interpolated waypoints between `from_pose` and
        `to_pose` in polar space and confirm every one is clear.

        The Arduino interpolates SET ALL commands in joint space, so the
        end-effector does NOT travel a Cartesian straight line between
        poses — but a polar-space straight line is a reasonable superset
        of the set of intermediate positions the tip can occupy. Blocking
        any of those samples is enough to reject the whole move.
        """
        if samples < 2:
            samples = 2
        th0, r0, z0 = from_pose
        th1, r1, z1 = to_pose
        for i in range(1, samples + 1):
            t = i / float(samples)
            th = th0 + (th1 - th0) * t
            r  = r0  + (r1  - r0)  * t
            z  = z0  + (z1  - z0)  * t
            if not self.is_clear(th, r, z):
                return False
        return True

    # ── Full replanning ladder ───────────────────────────────────────────

    def plan_clear_pose(self, desired: Pose,
                        current: Pose) -> Optional[Pose]:
        """
        Return a clear pose near `desired` that is reachable from `current`
        without sweeping through an obstacle, or None if no replan succeeds.

        Ladder:
          0. Live ToF gate — if a primary channel is hot, only retreats
             (r↓ or meaningful z change) are considered. theta-only moves
             and no-op poses return None immediately so the caller reports
             "blocked by live sensor".
          1. `desired` itself
          2. z-hop at (desired θ, desired r), via find_clear_z
          3. θ ± nudge at (same r, desired z), z-hop retried at each nudge
          4. Radial retreat at (desired θ, desired z)
          5. None
        """
        d_theta, d_r, d_z = desired

        # 0. Live-sensor gate. When a primary ToF is actively reading an
        #    obstacle, the only commands that may proceed are explicit
        #    retreats from the user (target.r < current.r) or meaningful
        #    z-changes. Anything else returns None immediately so the
        #    caller reports "no clear route". We deliberately do NOT try
        #    retreat fallbacks here — that would still advance theta in
        #    the direction the user pressed, against the spirit of the
        #    rejection. The user must press s/q/e themselves to back away.
        if not self._live_gate_allows(current, (d_theta, d_r, d_z)):
            return None

        # 1. Desired — the happy path.
        if self._pose_ok(current, (d_theta, d_r, d_z)):
            return (d_theta, d_r, d_z)

        # 2. Z-hop at the desired (theta, r).
        z_alt = self.find_clear_z(d_theta, d_r, d_z)
        if z_alt is not None and self._pose_ok(current, (d_theta, d_r, z_alt)):
            return (d_theta, d_r, z_alt)

        # 3. Theta nudges — both signs, retry z-hop at each.
        for delta in GUARD_THETA_NUDGE_STEPS_DEG:
            for sign in (+1.0, -1.0):
                th = _clamp(d_theta + sign * delta,
                            SWEEP_THETA_MIN, SWEEP_THETA_MAX)
                if abs(th - d_theta) < 1e-3:
                    continue   # clamped to the same spot
                # Try the desired z first, then a z-hop from there.
                if self._pose_ok(current, (th, d_r, d_z)):
                    return (th, d_r, d_z)
                z_n = self.find_clear_z(th, d_r, d_z)
                if z_n is not None and self._pose_ok(current, (th, d_r, z_n)):
                    return (th, d_r, z_n)

        # 4. Radial retreat — pull the tip closer to the base at (d_theta, d_z).
        for dr in GUARD_R_RETREAT_STEPS_MM:
            r_alt = _clamp(d_r - dr, R_MIN, R_MAX)
            if abs(r_alt - d_r) < 1e-3:
                continue
            if self._pose_ok(current, (d_theta, r_alt, d_z)):
                return (d_theta, r_alt, d_z)
            z_r = self.find_clear_z(d_theta, r_alt, d_z)
            if z_r is not None and self._pose_ok(current, (d_theta, r_alt, z_r)):
                return (d_theta, r_alt, z_r)

        return None

    # ── Internals ────────────────────────────────────────────────────────

    def _live_gate_allows(self, current: Pose, target: Pose) -> bool:
        """
        Synchronous live-sensor check.

        Returns True if there is no hot live ToF reading OR the target is
        an unambiguous retreat from `current`. Returns False otherwise.

        Rules — in order:
          1. No tof_state, or live obstacle_response != REPLAN → allow.
          2. Source channel is not a REPLAN-authority channel → allow
             (advisory/ignored sensors don't gate manual input).
          3. target.r < current.r by ≥ 1 mm — unambiguous retract → allow.
          4. |target.z - current.z| > Z_STEP/2 — meaningful z change → allow.
          5. Otherwise → reject.

        Note: a theta-only change with no radial retreat is rejected even
        when the arm is rotating away from the obstacle. The guard cannot
        predict the sensor reading at the target pose, so the safest
        policy is to require an unambiguous retreat axis.
        """
        if self._tof_state is None:
            return True
        try:
            snap = self._tof_state.snapshot()
        except Exception:
            return True

        if snap.get('obstacle_response') != ObstacleResponse.REPLAN:
            return True

        source = snap.get('obstacle_source', '') or ''
        try:
            ch_idx = int(source.replace('tof_ch', ''))
        except (ValueError, AttributeError):
            return True   # unknown source (e.g. IR advisory) — don't gate
        if ch_idx not in SENSOR_REPLAN_CHANNELS:
            return True

        _c_theta, c_r, c_z = current
        _t_theta, t_r, t_z = target

        # Unambiguous radial retreat — always allowed so the user has a way out.
        if t_r < c_r - 1e-3:
            return True
        # Meaningful z-change — allowed. Z_STEP/2 keeps micro-adjustments
        # from sneaking through while real key presses (≥ Z_STEP) pass.
        if abs(t_z - c_z) > (Z_STEP / 2.0):
            return True

        return False

    def _pose_ok(self, current: Pose, target: Pose) -> bool:
        """Target is reachable, clear, and the swept path is clear."""
        th, r, z = target
        if z < Z_MIN or z > Z_MAX:
            return False
        if r < R_MIN or r > R_MAX:
            return False
        try:
            if reachability(th, r, z) != 'ok':
                return False
        except Exception:
            return False
        if not self.is_clear(th, r, z):
            return False
        if not self.is_clear_path(current, target):
            return False
        return True


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
