"""
arm_state.py — Thread-safe shared state for the arm controller.

ArmState is the single source of truth for all mutable state.  It is
read by the display thread and written by the control loop and the serial
reader thread, so every mutation goes through a Lock.

Two groups of state:
  - IK / polar state  : what the user is commanding (theta, r, z, offsets)
  - Joint state       : last commanded servo angles (mirrored from commands)
  - Equilibrium state : the pose that the 'H' key returns to

The lock is exposed as _lock so the controller can batch multiple
mutations inside a single `with state._lock:` block efficiently.

RLock (re-entrant lock) is used so that methods which acquire the lock
internally (e.g. set_equil_from_current) can be called safely from inside
an outer `with state._lock:` block in the controller without deadlocking.
"""

import threading
from .constants import (
    HOME_POS, DELTA_DEFAULT,
    DEFAULT_THETA, DEFAULT_R, DEFAULT_Z, DEFAULT_WRIST_OFFSET,
    GRIPPER_CLOSE,
)


class ArmState:
    def __init__(self) -> None:
        self._lock = threading.RLock()   # RLock: re-entrant, prevents nested deadlock

        # ── Joint angles (degrees) — shadowed from commands sent ──────────
        self.joints: list[float] = [float(x) for x in HOME_POS]

        # ── IK polar coordinates ──────────────────────────────────────────
        self.theta: float = DEFAULT_THETA   # base servo angle, 0–180°
        self.r:     float = DEFAULT_R       # horizontal reach, mm
        self.z:     float = DEFAULT_Z       # height relative to shoulder, mm

        # ── Wrist state ───────────────────────────────────────────────────
        # wrist_offset: manual tilt added on top of the auto-level angle.
        # Interpreted as the desired gripper world pitch = −wrist_offset
        # (so offset=0 → horizontal; offset=−90 → vertical up; offset=+90
        # → vertical down). Default is −90 to keep the gripper pointing
        # straight up at startup, matching HOME_POS's physical pose.
        self.wrist_offset: float = DEFAULT_WRIST_OFFSET
        self.wrist_rot:    int   = 90       # wrist rotation (independent)

        # ── Gripper ───────────────────────────────────────────────────────
        self.gripper: int = GRIPPER_CLOSE

        # ── Slew rate (deg/tick at 100 Hz) ────────────────────────────────
        self.delta: int = DELTA_DEFAULT

        # ── Equilibrium (the pose 'H' returns to; Shift+H saves it) ──────
        self.equil_theta:        float = DEFAULT_THETA
        self.equil_r:            float = DEFAULT_R
        self.equil_z:            float = DEFAULT_Z
        self.equil_wrist_offset: float = DEFAULT_WRIST_OFFSET
        self.equil_wrist_rot:    int   = 90
        self.equil_gripper:      int   = GRIPPER_CLOSE

        # ── Serial connection status ──────────────────────────────────────
        self.connected:  bool = False
        self.last_cmd:   str  = ""
        self.last_resp:  str  = ""
        self.last_error: str  = ""

        # Polar (theta/r/z) is ALWAYS user-authoritative. We keep a flag for
        # backward compatibility with the controller's _drain_responses path
        # (which gates the one-shot fk_polar sync on first POS), but we
        # default it to True so the sync never fires. Rationale: the sync
        # ran fk_polar on the first POS response, but at DELTA=1 the Arduino
        # is still transiting from HOME when that first POS arrives, so
        # fk_polar(mid_flight_joints) produces bogus polar values and
        # drives state into near-singular corners of the workspace. Every
        # subsequent keypress then hits "IK unreachable" and locks up.
        # Keeping polar at the software default (DEFAULT_THETA/R/Z) means
        # keypresses operate on a stable virtual pose; the arm transits to
        # that pose on first command.
        self.polar_synced: bool = True

        # ── ToF / IR obstacle state (updated by controller) ─────────────
        self.obstacle_response: str   = 'clear'    # 'clear', 'replan', 'back_away'
        self.obstacle_source:   str   = ''          # 'tof_chN' or 'ir'
        self.obstacle_dist_mm:  float = -1.0        # closest measured distance
        self.tof_snapshot:      dict  = {}          # latest ToFState.snapshot()

    # ── Mutations ─────────────────────────────────────────────────────────

    def update_joints_from_ik(self, ik_result,
                               wrist_rot: int, gripper: int) -> None:
        """Apply a solved IK result to the joint shadow array.

        ``ik_result.wrist_vert`` was computed by ``solve_ik`` with the
        caller-provided ``gripper_world_deg`` already baked in. We just
        copy that WV through; no additional wrist_level_angle
        recomputation is needed.
        """
        with self._lock:
            self.joints[0] = ik_result.base
            self.joints[1] = ik_result.shoulder
            self.joints[2] = ik_result.elbow
            self.joints[3] = ik_result.wrist_vert
            self.joints[4] = wrist_rot
            self.joints[5] = gripper

    def set_equil_from_current(self) -> None:
        """Save the current IK state as the new equilibrium."""
        with self._lock:
            self.equil_theta        = self.theta
            self.equil_r            = self.r
            self.equil_z            = self.z
            self.equil_wrist_offset = self.wrist_offset
            self.equil_wrist_rot    = self.wrist_rot
            self.equil_gripper      = self.gripper

    def restore_equil(self) -> None:
        """Restore IK state to the saved equilibrium values."""
        with self._lock:
            self.theta        = self.equil_theta
            self.r            = self.equil_r
            self.z            = self.equil_z
            self.wrist_offset = self.equil_wrist_offset
            self.wrist_rot    = self.equil_wrist_rot
            self.gripper      = self.equil_gripper

    # ── Thread-safe snapshot ──────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Return a copy of all display-relevant state.
        Callers must never hold the lock across rendering.
        """
        with self._lock:
            return {
                'joints':         list(self.joints),
                'theta':          self.theta,
                'r':              self.r,
                'z':              self.z,
                'wrist_offset':   self.wrist_offset,
                'wrist_rot':      self.wrist_rot,
                'gripper':        self.gripper,
                'delta':          self.delta,
                'equil_theta':    self.equil_theta,
                'equil_r':        self.equil_r,
                'equil_z':        self.equil_z,
                'connected':      self.connected,
                'last_cmd':       self.last_cmd,
                'last_resp':      self.last_resp,
                'last_error':     self.last_error,
                # ToF / IR
                'obstacle_response': self.obstacle_response,
                'obstacle_source':   self.obstacle_source,
                'obstacle_dist_mm':  self.obstacle_dist_mm,
                'tof_snapshot':      dict(self.tof_snapshot) if self.tof_snapshot else {},
            }
