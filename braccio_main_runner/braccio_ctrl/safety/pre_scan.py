"""
pre_scan.py — Pre-motion workspace observation routine.

Research §3 identifies that wrist-mounted ToF sensors create an unavoidable
blind spot: the arm literally cannot see obstacles it hasn't pointed at. The
mitigation is a short pre-scan that sweeps the wrist through a handful of
poses before any planned motion, letting the ToF bridge accumulate ~15 frames
across the hemisphere. Cells that remain unobserved stay 'unknown' in the
coarse grid, which the planner biases paths away from.

The routine is intentionally agnostic about *how* poses get sent: callers
pass a ``send_pose`` callback. This keeps pre_scan.py decoupled from
SerialBridge / BraccioBackend / simulator — the BT integration wires
whichever is appropriate at runtime.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable, Optional

from ..constants import JOINT_LIMITS


# Default scan pattern: wrist-rotation swings while the arm stays at its
# current non-wrist pose. Five poses → roughly 1.5 s of scan at the sweep
# tick rate (10 Hz) assuming the wrist servo finishes each commanded step
# within 150 ms. Tuned to match the research's "3-5 key poses" guidance.
_DEFAULT_WRIST_R_OFFSETS_DEG = (-40, -20, 0, +20, +40)
_DEFAULT_WRIST_V_OFFSETS_DEG = (-20, 0, +20)


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class PreScanRoutine:
    """Emits a short sequence of wrist poses to reveal the workspace.

    Parameters
    ----------
    send_pose : Callable[[list[int]], None]
        Callback invoked once per pose. Receives a 6-element Braccio-degree
        list ready for SET ALL. The caller is responsible for serial I/O.
    read_current_joints : Callable[[], list[int]]
        Returns the current joint state, so the routine can pivot the wrist
        without disturbing the base / shoulder / elbow.
    settle_s : float
        Seconds to wait between successive commands to let the servo settle
        and the ToF bridge ingest frames.
    wrist_r_offsets : iterable of int
        Offsets (degrees, relative to current wrist rotation) to visit.
    wrist_v_offsets : iterable of int
        Offsets (degrees) for the vertical wrist axis.
    """

    def __init__(
        self,
        send_pose: Callable[[list[int]], None],
        read_current_joints: Callable[[], list[int]],
        settle_s: float = 0.30,
        wrist_r_offsets: Iterable[int] = _DEFAULT_WRIST_R_OFFSETS_DEG,
        wrist_v_offsets: Iterable[int] = _DEFAULT_WRIST_V_OFFSETS_DEG,
    ):
        self._send = send_pose
        self._read = read_current_joints
        self._settle_s = float(settle_s)
        self._wrist_r_offsets = tuple(int(x) for x in wrist_r_offsets)
        self._wrist_v_offsets = tuple(int(x) for x in wrist_v_offsets)

    # ── Plan the sequence (pure — no I/O) ──────────────────────────────────

    def plan_poses(self, base_joints: Optional[list[int]] = None) -> list[list[int]]:
        """Return the list of poses that will be visited.

        Useful for unit testing without invoking the ``send_pose`` callback.
        """
        current = list(base_joints if base_joints is not None else self._read())
        wr_lo, wr_hi = JOINT_LIMITS[4]
        wv_lo, wv_hi = JOINT_LIMITS[3]

        seen: set[tuple[int, ...]] = set()
        poses: list[list[int]] = []
        for dv in self._wrist_v_offsets:
            for dr in self._wrist_r_offsets:
                pose = list(current)
                pose[3] = _clamp(int(current[3]) + dv, wv_lo, wv_hi)
                pose[4] = _clamp(int(current[4]) + dr, wr_lo, wr_hi)
                key = tuple(pose)
                if key in seen:
                    continue
                seen.add(key)
                poses.append(pose)
        # Return to the original wrist position at the end so the caller
        # resumes from a predictable state.
        if not poses or tuple(poses[-1]) != tuple(current):
            poses.append(list(current))
        return poses

    # ── Execute ────────────────────────────────────────────────────────────

    def execute(
        self,
        base_joints: Optional[list[int]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        """Send every pose from ``plan_poses`` via ``send_pose`` with
        ``settle_s`` between commands."""
        poses = self.plan_poses(base_joints)
        for pose in poses:
            self._send(pose)
            if self._settle_s > 0.0:
                sleep_fn(self._settle_s)
