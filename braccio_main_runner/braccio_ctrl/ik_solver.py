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
from .constants import L1, L2, L3, JOINT_LIMITS, R_MIN


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
    Compute the wrist vertical angle that keeps the gripper level in the
    overhand orientation (gripper tip forward-and-up as
    ``wrist_vert → 180°``).

    Formula: WristV = Shoulder + Elbow - 180

    Convention flip from the original "180 - Shoulder - Elbow": on the
    physical Braccio the wrist-vertical servo mounts so that increasing
    ``wrist_vert`` rotates the gripper upward. The old convention drove
    the wrist toward 0 as shoulder+elbow grew, which produced the
    "backwards / under-hand" look the user reported. The flipped form
    saturates wrist at 180 (overhand) when shoulder and elbow are bent
    together — matches the intuition "bending the arm in → gripper
    rotates forward-and-up".
    """
    angle = shoulder_deg + elbow_deg - 180.0
    w_lo, w_hi = JOINT_LIMITS[3]
    return max(w_lo, min(w_hi, angle))


def apply_wrist_offset(base_wrist_deg: float, offset_deg: float) -> int:
    """
    Add a manual tilt offset to the auto-level wrist angle and clamp to limits.
    """
    w_lo, w_hi = JOINT_LIMITS[3]
    return int(round(max(w_lo, min(w_hi, base_wrist_deg + offset_deg))))


def fk_polar(joints: list) -> tuple:
    """
    Forward kinematics: joint servo angles → IK polar state (theta, r, z).

    This is the exact inverse of solve_ik() / polar_to_cartesian().
    Used to sync the software IK state from the arm's actual joint positions
    so the first commanded move after startup is a delta from where the arm
    really is, not from a stale software default.

    Parameters
    ----------
    joints : 6-element list [Base, Shoulder, Elbow, WristV, WristR, Gripper]
             (degrees, as reported by the Arduino)

    Returns
    -------
    (theta_deg, r_mm, z_mm) — the polar coordinates that solve_ik() would
    have needed to produce these Base/Shoulder/Elbow angles.
    """
    base_deg     = float(joints[0])
    shoulder_deg = float(joints[1])
    elbow_deg    = float(joints[2])

    theta = base_deg

    # Distance from shoulder pivot to wrist pivot (law of cosines).
    # The interior elbow angle in the kinematic triangle equals elbow_deg.
    E_rad = math.radians(elbow_deg)
    dist2 = L1 ** 2 + L2 ** 2 - 2.0 * L1 * L2 * math.cos(E_rad)
    dist  = math.sqrt(max(0.0, dist2))

    if dist < 1e-3:
        # Fully folded arm — return a safe near-origin value
        return theta, L3 + 1.0, 0.0

    # Beta: angle at the shoulder vertex of the kinematic triangle.
    # Matches the cos_beta used in solve_ik().
    cos_beta = (dist2 + L1 ** 2 - L2 ** 2) / (2.0 * dist * L1)
    cos_beta = max(-1.0, min(1.0, cos_beta))
    beta_deg = math.degrees(math.acos(cos_beta))

    # Alpha: elevation angle of the wrist-pivot target from horizontal.
    # In solve_ik(): shoulder = alpha + beta  →  alpha = shoulder - beta
    alpha_deg = shoulder_deg - beta_deg
    alpha_rad = math.radians(alpha_deg)

    # Horizontal and vertical components of the shoulder→wrist vector
    r_eff = dist * math.cos(alpha_rad)   # effective reach (to wrist pivot)
    z     = dist * math.sin(alpha_rad)   # height (signed: negative = below pivot)

    # Add gripper length back to get the tip reach
    r = r_eff + L3

    r = max(R_MIN, r)   # keep within IK reach envelope
    return theta, r, z


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


def fk_polar(joints: list) -> tuple[float, float, float]:
    """
    Approximate forward kinematics inverse of solve_ik() in polar space.

    Parameters
    ----------
    joints : [base, shoulder, elbow, wristV, wristR, gripper]

    Returns
    -------
    (theta_deg, r_mm, z_mm)
    """
    if len(joints) < 3:
        raise ValueError('joints must contain at least base/shoulder/elbow')

    theta = float(joints[0])
    shoulder = math.radians(float(joints[1]))
    elbow_internal = math.radians(180.0 - float(joints[2]))

    # Wrist-pivot position from 2-link planar FK.
    r_wrist = L1 * math.cos(shoulder) + L2 * math.cos(shoulder - elbow_internal)
    z = L1 * math.sin(shoulder) + L2 * math.sin(shoulder - elbow_internal)

    # solve_ik subtracts L3 from radial distance; add it back to recover r.
    r = max(0.0, r_wrist + L3)
    return float(theta), float(r), float(z)
