"""
ik_solver.py — Analytical IK/FK for the Braccio, based on cgxeiji/CGx-InverseK.

Reference implementation: https://github.com/cgxeiji/CGx-InverseK

This module mirrors that library's math directly in Python. The key ideas:
 * Work in an INTERNAL arm-plane frame rotated by −π/2 from the world
   (r, z) plane. In the internal frame, "up" is +x and "forward" is −y.
 * At HOME the arm is stacked vertical; all four Braccio joint commands
   equal 90°. In the internal frame every joint's "internal angle" is 0.
 * Braccio servo convention: ``braccio_deg = (internal_rad + π/2) × 180/π``
   so a servo value of 90° corresponds to internal 0° (extended / aligned).
 * Shoulder equation `_shoulder = α − β` places the upper arm on the
   elbow-down side (elbow tucked toward the arm's reach direction).
   Falls back to elbow-up (α + β) if the primary branch lands any joint
   outside its servo limits.
 * Wrist equation `_wrist = φ − _shoulder − _elbow` closes the kinematic
   chain so the end-effector points at the requested approach angle φ
   (in world radians, with φ=π/2 meaning "gripper pointing straight up").

CGx's example only uses these three joint servos; the Braccio's base is a
planar rotation we add back on top: ``B_cmd = 180° − θ_world`` so that
pressing the A/D keys to rotate the tip feels in the correct direction
given the physical arm's base-servo mounting.
"""

from __future__ import annotations

import math
from typing import Optional

from .constants import L1, L2, L3, JOINT_LIMITS


# ── Data class ────────────────────────────────────────────────────────────

class IKResult:
    """Joint angles (degrees, integers) returned by ``solve_ik``."""

    __slots__ = ("base", "shoulder", "elbow", "wrist_vert")

    def __init__(self, base: int, shoulder: int, elbow: int, wrist_vert: int):
        self.base       = base
        self.shoulder   = shoulder
        self.elbow      = elbow
        self.wrist_vert = wrist_vert

    def __repr__(self) -> str:
        return (f"IKResult(B={self.base}, S={self.shoulder}, "
                f"E={self.elbow}, Wv={self.wrist_vert})")


# ── Low-level utilities ───────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _b2a(deg: float) -> float:
    """Braccio degrees → internal radians. Braccio 90° = internal 0."""
    return math.radians(deg) - math.pi / 2.0


def _a2b(rad: float) -> float:
    """Internal radians → Braccio degrees."""
    return math.degrees(rad + math.pi / 2.0)


def _in_range(rad: float, limits_deg: tuple) -> bool:
    """Whether an internal-frame angle falls within a Braccio servo's range."""
    lo, hi = limits_deg
    return _b2a(lo) <= rad <= _b2a(hi)


def polar_to_cartesian(theta_deg: float, r_mm: float) -> tuple[float, float]:
    """Convert polar (θ, r) → Cartesian (x, y). θ = 0 → +X."""
    t = math.radians(theta_deg)
    return r_mm * math.cos(t), r_mm * math.sin(t)


# ── Core planar solver (CGx _solve) ───────────────────────────────────────

def _solve_planar(x_mm: float, y_mm: float, phi_rad: float
                   ) -> Optional[tuple[float, float, float]]:
    """2-link + wrist IK in the arm-plane.

    Inputs (x, y) are the tip target in the plane, where x is the
    horizontal reach and y is the height above the shoulder pivot.
    ``phi_rad`` is the desired gripper approach angle in world radians
    (π/2 = vertical-up, 0 = horizontal-forward, −π/2 = vertical-down).

    Returns ``(shoulder_rad, elbow_rad, wrist_rad)`` in the internal
    frame, or ``None`` if unreachable. The caller converts to Braccio
    degrees with _a2b(). This is a direct translation of the CGx
    library's ``_solve`` function (src/InverseK.cpp:81-128) except we
    compare against Braccio JOINT_LIMITS when checking ``_in_range``.
    """
    # Frame rotation: the library works in a plane rotated by −π/2 from
    # the natural (r, z) plane so "up" becomes +x in the internal frame.
    r     = math.hypot(x_mm, y_mm)
    theta = math.atan2(y_mm, x_mm)
    _x = r * math.cos(theta - math.pi / 2.0)
    _y = r * math.sin(theta - math.pi / 2.0)
    _phi = phi_rad - math.pi / 2.0

    # Wrist pivot: pull back from the tip along the requested approach.
    xw = _x - L3 * math.cos(_phi)
    yw = _y - L3 * math.sin(_phi)

    alpha = math.atan2(yw, xw)
    R     = math.hypot(xw, yw)
    if R < 1e-6 or R > (L1 + L2) or R < abs(L1 - L2):
        return None

    # Law of cosines — triangle (shoulder, elbow, wrist).
    cos_beta  = (R * R + L1 * L1 - L2 * L2) / (2.0 * R * L1)
    cos_gamma = (L1 * L1 + L2 * L2 - R * R) / (2.0 * L1 * L2)
    cos_beta  = _clamp(cos_beta,  -1.0, 1.0)
    cos_gamma = _clamp(cos_gamma, -1.0, 1.0)
    beta  = math.acos(cos_beta)
    gamma = math.acos(cos_gamma)

    # Primary (elbow-down) branch.
    shoulder = alpha - beta
    elbow    = math.pi - gamma
    wrist    = _phi - shoulder - elbow

    if (_in_range(shoulder, JOINT_LIMITS[1])
            and _in_range(elbow, JOINT_LIMITS[2])
            and _in_range(wrist, JOINT_LIMITS[3])):
        return shoulder, elbow, wrist

    # Elbow-up fallback (flip the bend and recompute the wrist).
    shoulder = alpha + beta
    elbow    = -(math.pi - gamma)
    wrist    = _phi - shoulder - elbow
    if (_in_range(shoulder, JOINT_LIMITS[1])
            and _in_range(elbow, JOINT_LIMITS[2])
            and _in_range(wrist, JOINT_LIMITS[3])):
        return shoulder, elbow, wrist

    return None


def _solve_planar_free_phi(x_mm: float, y_mm: float, phi_initial: float
                            ) -> Optional[tuple[float, float, float, float]]:
    """Find the nearest reachable approach angle to ``phi_initial``.

    Tries ``phi_initial`` first, then sweeps outward in ±1° steps until a
    reachable φ is found or the full ±180° range has been exhausted.
    Returns ``(shoulder, elbow, wrist, phi_used)`` in radians or ``None``.

    The sweep STARTS at the caller's desired φ so a user's explicit
    gripper pitch (via wrist_offset) is respected whenever possible. If
    the exact φ is unreachable the fallback lands on the closest
    reachable alternative — matching what CGx's free-phi overload
    effectively does, but anchored to the caller's intent instead of
    always defaulting to vertical.
    """
    # Try the user's preferred phi first.
    result = _solve_planar(x_mm, y_mm, phi_initial)
    if result is not None:
        s, e, w = result
        return s, e, w, phi_initial
    # Sweep outward in ±1° increments.
    step = math.radians(1.0)
    for i in range(1, 181):
        for sign in (1, -1):
            phi = phi_initial + sign * i * step
            result = _solve_planar(x_mm, y_mm, phi)
            if result is not None:
                s, e, w = result
                return s, e, w, phi
    return None


# ── Forward kinematics ────────────────────────────────────────────────────

def fk_tip_physical(joints) -> tuple[float, float, float]:
    """Physical forward kinematics — where the tip ACTUALLY lands.

    The CGx-based ``fk_polar`` uses the library's internal sign
    convention for elbow and wrist. The physical Braccio on this arm
    (per the probe_joint_conventions.py findings) bends the opposite
    way: E > 90 tips the forearm FORWARD (toward +r), not backward.
    So the joints the IK outputs, when executed by the physical servo,
    land the tip at a different location than CGx's FK predicts.

    This function uses the physical convention:
        θ_upper_world = S
        θ_fa_world    = S + 90 − E
        θ_g_world     = S + 180 − E − WV

    Returns (theta_deg, r_mm, z_mm) of the actual physical tip in the
    shoulder-pivot frame. Use this for state tracking, RL observation
    vectors, and any display that needs to match what the user sees
    physically. Use ``fk_polar`` only when you need the inverse of
    ``solve_ik`` for CGx's internal math.
    """
    if len(joints) < 4:
        raise ValueError("joints must contain at least [B, S, E, WV]")
    B  = float(joints[0])
    S  = float(joints[1])
    E  = float(joints[2])
    WV = float(joints[3])

    th_upper = math.radians(S)
    th_fa    = math.radians(S + 90.0  - E)
    th_g     = math.radians(S + 180.0 - E - WV)

    r_tip = (L1 * math.cos(th_upper)
             + L2 * math.cos(th_fa)
             + L3 * math.cos(th_g))
    z_tip = (L1 * math.sin(th_upper)
             + L2 * math.sin(th_fa)
             + L3 * math.sin(th_g))
    theta = 180.0 - B
    return theta, r_tip, z_tip


def fk_polar(joints) -> tuple[float, float, float]:
    """Servo angles → tip polar (θ, r, z).

    Inverse of ``solve_ik`` in the same conventions. Uses the internal
    frame forward kinematics: the upper arm, forearm, and hand each
    rotate the arm plane by their internal angle.
    """
    if len(joints) < 4:
        raise ValueError("joints must contain at least [B, S, E, WV]")
    B  = float(joints[0])
    S  = float(joints[1])
    E  = float(joints[2])
    WV = float(joints[3])

    # Internal-frame joint angles.
    s_int = _b2a(S)
    e_int = _b2a(E)
    w_int = _b2a(WV)

    # Forward kinematics in the internal frame (x = up, y = −forward).
    # Each link contributes a vector rotated by the cumulative angle.
    s_world = s_int
    f_world = s_int + e_int           # forearm world angle (internal)
    g_world = s_int + e_int + w_int   # gripper world angle (internal)

    _x = (L1 * math.cos(s_world)
          + L2 * math.cos(f_world)
          + L3 * math.cos(g_world))
    _y = (L1 * math.sin(s_world)
          + L2 * math.sin(f_world)
          + L3 * math.sin(g_world))

    # Un-rotate the internal frame back to world. In _solve_planar we
    # did (_x, _y) = (r·cos(θ−π/2), r·sin(θ−π/2)) with input (r_tip,
    # z_tip). That simplifies to (_x, _y) = (z_tip, −r_tip), so the
    # inverse is (r_tip, z_tip) = (−_y, _x).
    r_tip = -_y
    z_tip = _x

    theta = 180.0 - B
    return theta, r_tip, z_tip


# ── Public IK entry point ────────────────────────────────────────────────

def solve_ik(x_mm: float, y_mm: float, z_mm: float,
             gripper_world_deg: float = 0.0,
             theta_hint_deg: Optional[float] = None) -> Optional[IKResult]:
    """Braccio IK entry point.

    Parameters
    ----------
    x_mm, y_mm, z_mm
        Tip target in the shoulder-pivot frame.
    gripper_world_deg
        Desired gripper world angle in the (r, z) plane. 0 = horizontal
        forward, +90 = vertical up, −90 = vertical down.
    theta_hint_deg
        Used for the base command when the tip is on the base axis
        (r ≲ 1 mm). Without this hint the base snaps to 180° at the
        singular radius.
    """
    r_tip = math.hypot(x_mm, y_mm)

    # Base rotation — invert the servo flip so pressing A/D rotates the
    # tip intuitively. At r ≈ 0 atan2 is meaningless; use the hint.
    if r_tip < 1.0 and theta_hint_deg is not None:
        theta_world = theta_hint_deg
    else:
        theta_world = math.degrees(math.atan2(y_mm, x_mm))
    b_lo, b_hi = JOINT_LIMITS[0]
    base_cmd = _clamp(180.0 - theta_world, b_lo, b_hi)

    # In-plane target: use the radial reach (not x) as the forward axis
    # and z as the vertical axis, since CGx's _solve assumes a planar arm.
    phi_rad = math.radians(gripper_world_deg)

    # Try the requested approach angle first; if that specific pose is
    # unreachable, find the nearest reachable phi to honour the caller's
    # intent as closely as possible.
    free = _solve_planar_free_phi(r_tip, z_mm, phi_rad)
    if free is None:
        return None
    s_rad, e_rad, w_rad, _phi_used = free

    return IKResult(
        base       = int(round(base_cmd)),
        shoulder   = int(round(_a2b(s_rad))),
        elbow      = int(round(_a2b(e_rad))),
        wrist_vert = int(round(_a2b(w_rad))),
    )


# ── Auto-level + offset helpers ───────────────────────────────────────────

def wrist_level_angle(shoulder_deg: float, elbow_deg: float) -> float:
    """Return WV that aims the gripper horizontally (φ = 0).

    Derived from the CGx chain-closure `wrist = φ − shoulder − elbow` in
    the internal frame. For φ = 0: wrist_internal = −s_int − e_int, so
    wrist_braccio = 90° − (S − 90°) − (E − 90°) = 270 − S − E. Clamped
    to the physical WV servo range.
    """
    wv = 270.0 - shoulder_deg - elbow_deg
    w_lo, w_hi = JOINT_LIMITS[3]
    return _clamp(wv, w_lo, w_hi)


def apply_wrist_offset(base_wrist_deg: float, offset_deg: float) -> int:
    """Add a manual tilt offset on top of an auto-level wrist."""
    w_lo, w_hi = JOINT_LIMITS[3]
    return int(round(_clamp(base_wrist_deg + offset_deg, w_lo, w_hi)))


# ── Reachability hint (cheap polar-space check) ───────────────────────────

def reachability(theta_deg: float, r_mm: float, z_mm: float) -> str:
    """Coarse workspace membership check. Returns 'ok', 'too_far',
    'too_close', or 'at_origin'. Does NOT run the full planar solve."""
    x, y = polar_to_cartesian(theta_deg, r_mm)
    r_tip = math.hypot(x, y)
    r_w = r_tip - L3
    z_w = z_mm
    dist = math.hypot(r_w, z_w)
    if dist < 1e-3:
        return 'at_origin'
    if dist > (L1 + L2):
        return 'too_far'
    if dist < abs(L1 - L2):
        return 'too_close'
    return 'ok'
