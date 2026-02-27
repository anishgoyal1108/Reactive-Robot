"""
ik_solver.py — Python mirror of the Arduino solveIK() function.

Coordinate system (origin = shoulder pivot M2):
  +X  radially outward (horizontal, arm forward at theta=0)
  +Y  90° left of +X
  +Z  upward

The IK solves for a 2-link planar arm (shoulder + elbow) plus a fixed
gripper offset (L3). The wrist vertical is computed to keep the gripper
horizontal by default; a caller-supplied offset allows manual tilt.
"""

import math
from .constants import L1, L2, L3, JOINT_LIMITS


class IKResult:
    """Joint angles (degrees, integers) returned by solve_ik()."""
    def __init__(self, base: int, shoulder: int, elbow: int, wrist_vert: int):
        self.base       = base
        self.shoulder   = shoulder
        self.elbow      = elbow
        self.wrist_vert = wrist_vert

    def __repr__(self):
        return (f"IKResult(B={self.base}, S={self.shoulder}, "
                f"E={self.elbow}, Wv={self.wrist_vert})")


def solve_ik(x_mm: float, y_mm: float, z_mm: float):
    """
    Solve 2-link planar IK for Braccio.

    Returns IKResult on success, or None if the target is unreachable.

    Parameters
    ----------
    x_mm, y_mm : horizontal Cartesian coordinates (mm) from shoulder pivot
    z_mm       : vertical height (mm) relative to shoulder pivot
                 (negative = below the pivot, e.g. table surface)
    """
    # Base rotation — atan2 in the horizontal plane
    base_deg = math.degrees(math.atan2(y_mm, x_mm))
    lo, hi = JOINT_LIMITS[0]
    base_deg = max(lo, min(hi, base_deg))

    # Subtract gripper length so IK targets the wrist pivot, not the gripper tip
    r = math.sqrt(x_mm ** 2 + y_mm ** 2) - L3
    h = z_mm
    if r < 0.0:
        r = 0.0   # clamp: target closer than gripper length

    dist2 = r * r + h * h
    dist  = math.sqrt(dist2)

    if dist < 1e-3:
        return None          # target at origin
    if dist > (L1 + L2):
        return None          # too far (fully extended arm can't reach)
    if dist < abs(L1 - L2):
        return None          # too close (inside the inner unreachable zone)

    # Elbow angle via law of cosines
    cos_elbow = (dist2 - L1 ** 2 - L2 ** 2) / (2.0 * L1 * L2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_deg = 180.0 - math.degrees(math.acos(cos_elbow))

    # Shoulder angle
    alpha    = math.atan2(h, r)
    cos_beta = (dist2 + L1 ** 2 - L2 ** 2) / (2.0 * dist * L1)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    shoulder_deg = math.degrees(alpha + math.acos(cos_beta))
    s_lo, s_hi = JOINT_LIMITS[1]
    shoulder_deg = max(s_lo, min(s_hi, shoulder_deg))

    # Wrist vertical: keeps gripper horizontal (auto-level formula)
    wrist_vert_deg = wrist_level_angle(shoulder_deg, elbow_deg)

    return IKResult(
        base       = int(round(base_deg)),
        shoulder   = int(round(shoulder_deg)),
        elbow      = int(round(elbow_deg)),
        wrist_vert = int(round(wrist_vert_deg)),
    )


def polar_to_cartesian(theta_deg: float, r_mm: float):
    """
    Convert polar (theta, r) → Cartesian (x, y).

    theta_deg maps directly to the base servo angle convention used in the
    Arduino sketch: theta=90 → arm pointing straight forward.

    Returns (x_mm, y_mm).
    """
    theta_rad = math.radians(theta_deg)
    x = r_mm * math.cos(theta_rad)
    y = r_mm * math.sin(theta_rad)
    return x, y


def wrist_level_angle(shoulder_deg: float, elbow_deg: float) -> float:
    """
    Compute the wrist vertical angle that keeps the gripper horizontal.

    Formula: WristV = 180 - Shoulder - Elbow
    (derived from the constraint that all three link angles sum to 180°
    for a level end-effector in the vertical plane.)
    """
    angle = 180.0 - shoulder_deg - elbow_deg
    w_lo, w_hi = JOINT_LIMITS[3]
    return max(w_lo, min(w_hi, angle))


def apply_wrist_offset(base_wrist_deg: float, offset_deg: float) -> int:
    """
    Add a manual tilt offset to the auto-level wrist angle and clamp to limits.
    """
    w_lo, w_hi = JOINT_LIMITS[3]
    return int(round(max(w_lo, min(w_hi, base_wrist_deg + offset_deg))))


def reachability(theta_deg: float, r_mm: float, z_mm: float) -> str:
    """
    Check whether a polar target is reachable without moving the arm.

    Returns 'ok', 'too_far', 'too_close', or 'at_origin'.
    Useful for clamping UI feedback before sending a command.
    """
    x, y = polar_to_cartesian(theta_deg, r_mm)
    r = math.sqrt(x ** 2 + y ** 2) - L3
    if r < 0.0:
        r = 0.0
    dist = math.sqrt(r * r + z_mm * z_mm)
    if dist < 1e-3:
        return 'at_origin'
    if dist > (L1 + L2):
        return 'too_far'
    if dist < abs(L1 - L2):
        return 'too_close'
    return 'ok'
