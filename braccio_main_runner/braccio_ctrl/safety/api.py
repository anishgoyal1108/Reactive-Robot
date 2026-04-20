"""
api.py — Single integration surface for the safety stack.

The ``SafetyAPI`` class bundles:

    * WorldModel              — obstacle point cloud + coarse grid
    * ForwardKinematics       — headless PyBullet FK over braccio.urdf
    * CollisionChecker        — capsule-based full-arm check
    * SafetyPlanner           — BiRRT wrapper
    * CascadingReplanner      — 6-strategy recovery ladder (sequence mode)
    * PolarObstacleMap        — 1D sweep skip-ahead
    * PreScanRoutine          — wrist observation routine
    * SchmittTrigger per ToF  — hysteresis on all 4 distance readings
    * EMA per joint           — smoothing on outgoing commands
    * SafetyBlackboard + BT   — orchestration

The controller / web backend / tests only ever talk to ``SafetyAPI``; the
underlying py_trees / pybullet_planning machinery is an implementation
detail. Methods are organised in three groups:

  1. Lifecycle: ``start``, ``stop``
  2. Ingest    (sensor side): ``ingest_tof_frame``, ``update_ir_severity``,
                              ``update_tof_distances``, ``update_current_q``
  3. Command   (actuator side): ``queue_manual_intent``, ``queue_sequence``,
                                ``set_mode``, ``plan_and_validate``,
                                ``emergency_stop``, ``snapshot``

The BT runs in its own daemon thread; callers interact with it only through
the SafetyBlackboard, which ``SafetyAPI`` wraps with typed setters.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .. import constants as _c
from .behavior import (
    BehaviorTreeRunner,
    RefusalInfo,
    SafetyBlackboard,
    build_root,
)
from .collision import CollisionChecker, capsule_point_cloud_distance
from .fk import BRACCIO_JOINT_NAMES, ForwardKinematics
from .hysteresis import EMA, RateClamp, SchmittTrigger
from .planner import PlanResult, SafetyPlanner
from .polar_map import PolarObstacleMap
from .pre_scan import PreScanRoutine
from .replanner import CascadingReplanner, Waypoint, WaypointType
from .world_model import WorldModel


# ── TofProjector ───────────────────────────────────────────────────────────
#
# Projects a VL53L5CX grid (N×N distance matrix, mm) into world-frame XYZ
# points. Uses the headless PyBullet FK to read the ToF mount link's world
# pose, then casts each cell along the mount's +X axis rotated by the
# sensor's (row, col) angular offset inside its FoV. The IMU's rotation
# matrix optionally rotates the *whole arm* for compensation — the mount
# pose from FK is relative to the base, so rotating that by IMU yaw gives
# world-frame.

_MOUNT_BY_CHANNEL = {
    0: "tof_up_link",      # TOP    — rays along +Z from wrist
    1: "tof_right_link",   # RIGHT  — rays along -Y from wrist
    2: "tof_left_link",    # LEFT   — rays along +Y from wrist
    3: "tof_down_link",    # BOTTOM — rays along -Z from wrist
}

_TOF_FOV_DEG = 45.0


def _quat_to_rotmat(q: tuple[float, float, float, float]) -> np.ndarray:
    """PyBullet quaternion (xyzw) → 3×3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),      2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),      2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx),  1 - 2 * (xx + yy)],
    ], dtype=np.float64)


class TofProjector:
    """Converts (grid, channel, joints, imu_R) → world-frame (N, 3) points.

    Exposed separately from WorldModel so the controller (or tests) can
    reuse the projection without paying for the full safety stack.
    """

    def __init__(self, fk: ForwardKinematics, fov_deg: float = _TOF_FOV_DEG):
        self._fk = fk
        self._fov_half_rad = math.radians(float(fov_deg) / 2.0)

    def project(
        self,
        grid_mm: np.ndarray,
        channel: int,
        joints_deg: list[int],
        imu_R: Optional[np.ndarray] = None,
        threshold_mm: Optional[float] = None,
    ) -> np.ndarray:
        if grid_mm is None or np.size(grid_mm) == 0:
            return np.empty((0, 3), dtype=np.float64)
        mount = _MOUNT_BY_CHANNEL.get(channel)
        if mount is None:
            return np.empty((0, 3), dtype=np.float64)
        pos_mm, orn = self._fk.tof_mount_pose(joints_deg, mount)
        R_mount = _quat_to_rotmat(orn)
        R_world = imu_R @ R_mount if imu_R is not None else R_mount
        origin = np.asarray(pos_mm, dtype=np.float64)
        if imu_R is not None:
            origin = imu_R @ origin

        arr = np.asarray(grid_mm, dtype=np.float64)
        rows, cols = arr.shape
        out: list[np.ndarray] = []
        half = self._fov_half_rad
        for r in range(rows):
            for c in range(cols):
                d = float(arr[r, c])
                if math.isnan(d) or d <= 0:
                    continue
                # Defensive: VL53L5CX's valid-distance floor is ~20 mm. A
                # reading below that is almost always a "masked" / saturated
                # return that the firmware's v=1/status=5 promotion should
                # have already rewritten — but if any 0-20 mm value leaks
                # through, clamp it to the synthetic-masked distance so we
                # still project a close-obstacle point instead of one at
                # the sensor origin.
                if d < 20.0:
                    d = float(_c.TOF_MASKED_SYNTHETIC_MM)
                if threshold_mm is not None and d >= threshold_mm:
                    continue
                h_ang = ((c - (cols - 1) / 2.0) / max(1, (cols - 1) / 2.0)) * half
                v_ang = ((r - (rows - 1) / 2.0) / max(1, (rows - 1) / 2.0)) * half
                # Mount-local +X is the ray axis; +Y horizontal FoV; +Z vertical.
                ps = np.array([
                    d * math.cos(v_ang) * math.cos(h_ang),
                    d * math.cos(v_ang) * math.sin(h_ang),
                    d * math.sin(v_ang),
                ])
                p_world = R_world @ ps + origin
                out.append(p_world)
        if not out:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack(out, axis=0)


# ── SafetyAPI ──────────────────────────────────────────────────────────────

@dataclass
class SafetyAPIConfig:
    """All tuning knobs for the safety stack.

    Defaults are pulled from ``braccio_ctrl.constants`` so constants.py stays
    the single source of truth — call sites that need a custom value override
    only that field, they don't re-declare the whole tuning vector.
    """
    tof_engage_mm: Sequence[float] = field(
        default_factory=lambda: tuple(_c.TOF_ENGAGE_MM))
    tof_disengage_mm: Sequence[float] = field(
        default_factory=lambda: tuple(_c.TOF_DISENGAGE_MM))
    joint_command_ema_alpha: float = _c.JOINT_COMMAND_EMA_ALPHA
    joint_rate_max_deg: float = _c.JOINT_RATE_MAX_DEG
    capsule_clearance_mm: float = _c.CAPSULE_CLEARANCE_MM
    birrt_timeout_s: float = _c.BIRRT_TIMEOUT_S
    birrt_extend_deg: float = _c.BIRRT_EXTEND_DEG
    world_max_age_s: float = _c.WORLD_MAX_AGE_S
    world_kdtree_rebuild_s: float = _c.WORLD_KDTREE_REBUILD_S
    sequence_max_retries: int = _c.SEQUENCE_MAX_RETRIES
    bt_tick_hz: float = _c.BT_TICK_HZ
    home_q: tuple[int, int, int, int, int, int] = field(
        default_factory=lambda: tuple(_c.HOME_POS))
    obstacle_sound_path: str = _c.OBSTACLE_SOUND_PATH
    obstacle_sound_cooldown_s: float = _c.OBSTACLE_SOUND_COOLDOWN_S
    tof_self_filter_margin_mm: float = _c.TOF_SELF_FILTER_MARGIN_MM
    tof_self_filter_margin_per_channel_mm: tuple[float, ...] = field(
        default_factory=lambda: tuple(_c.TOF_SELF_FILTER_MARGIN_PER_CHANNEL_MM))


class SafetyAPI:
    """The single integration surface for the curses controller and web backend."""

    def __init__(
        self,
        send_cmd: Callable[[list[int], int], None],
        config: Optional[SafetyAPIConfig] = None,
    ):
        self._cfg = config or SafetyAPIConfig()
        self._send_cmd_user = send_cmd
        self._lock = threading.RLock()

        # Core pieces.
        self.world = WorldModel(
            max_age_s=self._cfg.world_max_age_s,
            rebuild_interval_s=self._cfg.world_kdtree_rebuild_s,
        )
        self.fk = ForwardKinematics()
        self.projector = TofProjector(self.fk)
        self.collision = CollisionChecker(
            self.world, self.fk,
            clearance_mm=self._cfg.capsule_clearance_mm,
        )
        self.planner = SafetyPlanner(
            self.collision, self.fk,
            extend_step_deg=self._cfg.birrt_extend_deg,
            default_timeout_s=self._cfg.birrt_timeout_s,
        )
        self.replanner = CascadingReplanner(
            self.planner, self.world,
            home_q=list(self._cfg.home_q),
            max_retries=self._cfg.sequence_max_retries,
        )
        self.polar = PolarObstacleMap()
        self.hysteresis: dict[int, SchmittTrigger] = {
            ch: SchmittTrigger(
                engage_mm=float(self._cfg.tof_engage_mm[ch]),
                disengage_mm=float(self._cfg.tof_disengage_mm[ch]),
            )
            for ch in range(len(self._cfg.tof_engage_mm))
        }
        self.joint_ema = EMA(alpha=self._cfg.joint_command_ema_alpha)
        self.joint_rate_clamp = RateClamp(
            max_delta_per_tick=self._cfg.joint_rate_max_deg,
        )
        self.bb = SafetyBlackboard(name=f"safety_bb_{id(self)}")
        self.bb.current_q = list(self._cfg.home_q)
        self.root = build_root(
            self.bb, self.planner, self.replanner,
            self.world, self.polar, self.hysteresis,
            home_q=list(self._cfg.home_q),
            nearest_obstacle_fn=self.collision.nearest_obstacle,
            collision_checker=self.collision,
        )
        self.runner = BehaviorTreeRunner(
            self.bb, self.root,
            send_cmd=self._handle_command,
            tick_hz=self._cfg.bt_tick_hz,
        )

        # Sound-effect state — one plays per replan_event, rate-limited.
        self._sound_path = str(self._cfg.obstacle_sound_path or "")
        self._sound_player = self._resolve_sound_player()
        self._sound_cooldown_s = float(self._cfg.obstacle_sound_cooldown_s)
        self._last_sound_t = 0.0
        self._last_replan_event_seen = 0
        self._sound_thread: Optional[threading.Thread] = None

        # Poll the Blackboard's replan_event counter on a lightweight
        # daemon; a fresh increment triggers one paplay call.
        self._sound_stop = threading.Event()
        self._sound_watch_thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        self.runner.start()
        if self._sound_path and self._sound_player and self._sound_watch_thread is None:
            self._sound_stop.clear()
            self._sound_watch_thread = threading.Thread(
                target=self._sound_loop, name="safety_sound", daemon=True,
            )
            self._sound_watch_thread.start()

    def stop(self) -> None:
        self._sound_stop.set()
        if self._sound_watch_thread is not None:
            self._sound_watch_thread.join(timeout=0.5)
            self._sound_watch_thread = None
        self.runner.stop()

    # ── Strict mode ────────────────────────────────────────────────────────

    def set_strict_mode(self, enabled: bool) -> None:
        """Toggle the manual-mode refusal dialog.

        When False (default) the manual branch silently auto-replans and
        simply declines to advance when no detour is found. When True the
        refusal dialog opens on every refusal for explicit user override.
        """
        self.bb.strict_mode = bool(enabled)

    def strict_mode_enabled(self) -> bool:
        return bool(self.bb.strict_mode)

    # ── Sensor ingest ──────────────────────────────────────────────────────

    def ingest_tof_frame(
        self,
        grids: Sequence[np.ndarray],
        joints: list[int],
        imu_R: Optional[np.ndarray] = None,
        thresholds_mm: Optional[Sequence[float]] = None,
    ) -> None:
        """Project every active channel into world frame and add to WorldModel.

        ``grids`` is a list/tuple of 4 2D arrays (one per channel); ``None``
        is permitted and skipped. Every channel contributes — authority is
        encoded by the per-channel threshold in ``engage_mm``.
        """
        if thresholds_mm is None:
            thresholds_mm = list(self._cfg.tof_engage_mm)
        for ch, grid in enumerate(grids):
            if grid is None:
                continue
            pts = self.projector.project(
                grid, ch, joints, imu_R, threshold_mm=thresholds_mm[ch],
            )
            if pts.shape[0]:
                pts = self._filter_arm_self_points(pts, joints, ch)
            if pts.shape[0]:
                self.world.ingest_points(pts)

    def _filter_arm_self_points(
        self, pts_world: np.ndarray, joints: list[int],
        channel: Optional[int] = None,
    ) -> np.ndarray:
        """Drop points that land inside (or very close to) the arm's own
        capsule volume. These are reflections off the arm's own structure
        and would otherwise accumulate as fake obstacles that trap the
        planner with ``start_in_collision``.

        ``channel`` selects a per-channel margin override from the config.
        CH3 (bottom sensor) needs a wider exclusion than CH0/CH1/CH2
        because it sits directly above the TCA9548A MUX breakout and
        sees the board / wiring at close range when the wrist tilts.
        """
        if pts_world.shape[0] == 0:
            return pts_world
        per_ch = self._cfg.tof_self_filter_margin_per_channel_mm
        if channel is not None and 0 <= channel < len(per_ch):
            margin = float(per_ch[channel])
        else:
            margin = float(self._cfg.tof_self_filter_margin_mm)
        try:
            capsules = self.fk.link_endpoints(list(joints))
        except Exception:
            return pts_world
        keep_mask = np.ones(pts_world.shape[0], dtype=bool)
        for cap in capsules:
            d = capsule_point_cloud_distance(
                cap.p0, cap.p1, cap.radius_mm + margin, pts_world,
            )
            if d.size:
                keep_mask &= (d > 0.0)
        if not np.any(keep_mask):
            return np.empty((0, 3), dtype=np.float64)
        return pts_world[keep_mask]

    def update_tof_distances(self, dists_mm: Sequence[float]) -> None:
        """Feed the hysteresis triggers with per-channel min-distance readings.

        Each channel's trigger latches independently. The Blackboard's
        ``tof_filtered`` field mirrors the raw input for BT consumption —
        the trigger latches are consulted inside the sweep branch.
        """
        self.bb.tof_filtered = list(dists_mm)
        for ch, reading in enumerate(dists_mm):
            trig = self.hysteresis.get(ch)
            if trig is None:
                continue
            try:
                trig.update(float(reading))
            except (TypeError, ValueError):
                continue

    def update_ir_severity(self, severity: int) -> None:
        self.bb.ir_severity = int(severity)

    def update_current_q(self, joints: list[int]) -> None:
        self.bb.current_q = list(joints)

    # ── Command side ───────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        if mode not in ("idle", "manual", "sweep", "sequence"):
            raise ValueError(f"unknown mode: {mode}")
        self.bb.mode = mode

    def queue_manual_intent(self, target_q: list[int]) -> None:
        """Queue a manual-mode target configuration.

        The BT's ManualBranch consumes this on its next tick and plans a
        validated path. On refusal, the Blackboard's ``refusal`` field is
        populated with a ``RefusalInfo`` for the dialog.
        """
        self.bb.mode = "manual"
        self.bb.manual_intent = list(target_q)

    def queue_sequence(self, waypoints: list[Waypoint]) -> None:
        self.bb.mode = "sequence"
        self.bb.sequence_queue = list(waypoints)

    def start_sweep(self) -> None:
        self.bb.mode = "sweep"
        # Snapshot whatever Z-ladder rung the sweep starts at as the
        # "origin lane" — the sweep will return here after overtaking an
        # obstacle. Normally 0 (SWEEP_Z_DEFAULT_MM); tracks any future
        # non-default start index automatically.
        self.bb.sweep_z_origin = int(self.bb.sweep_z_index)

    def stop_sweep(self) -> None:
        self.bb.mode = "idle"
        # Leave the polar map intact — blocked regions stay remembered until
        # decay() ages them out.

    def emergency_stop(self) -> None:
        self.bb.mode = "idle"
        self.bb.manual_intent = None
        self.bb.sequence_queue = []
        # Force-send the HOME pose so the arm retreats. Bypass EMA + rate
        # clamp so the retreat snaps to HOME in one tick — the whole point
        # of emergency_stop is to skip every smoothing budget.
        self._handle_command({
            "joints": list(self._cfg.home_q), "delta": 5,
            "bypass_smoothing": True,
        })

    def plan_and_validate(
        self, current_q: list[int], goal_q: list[int],
    ) -> PlanResult:
        """Synchronous one-shot plan.

        Useful for the web backend, which needs a synchronous decision per
        DSL step rather than going through the BT. Routes through the same
        SafetyPlanner the BT uses, so the collision semantics are identical.
        """
        return self.planner.plan(current_q, goal_q)

    def consume_refusal(self) -> Optional[RefusalInfo]:
        """Return and clear the current refusal, if any."""
        info = self.bb.refusal
        if info is not None:
            self.bb.refusal = None
        return info

    # ── Diagnostics ────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        detour = list(getattr(self.bb, "sweep_detour_path", []) or [])
        detour_idx = int(getattr(self.bb, "sweep_detour_idx", 0) or 0)
        return {
            "mode": self.bb.mode,
            "bt_state": self.bb.bt_state,
            "emergency": self.bb.emergency,
            "last_strategy": self.bb.last_strategy,
            "last_failure": self.bb.last_failure,
            "sweep_direction": self.bb.sweep_direction,
            "sweep_target_deg": self.bb.sweep_target_deg,
            "sweep_z_index": int(getattr(self.bb, "sweep_z_index", 0) or 0),
            "sweep_z_origin": int(getattr(self.bb, "sweep_z_origin", 0) or 0),
            "sweep_detour_path": [list(q) for q in detour],
            "sweep_detour_idx": detour_idx,
            "polar_blocked": self.bb.polar_snapshot,
            "world": self.world.snapshot(),
            "queue_length": len(self.bb.sequence_queue or []),
        }

    # ── Internal ───────────────────────────────────────────────────────────

    # ── Sound effect helpers ───────────────────────────────────────────────

    def _resolve_sound_player(self) -> Optional[str]:
        """Pick the first available on-PATH sound player."""
        if not self._sound_path:
            return None
        if not os.path.exists(self._sound_path):
            return None
        for candidate in ("paplay", "aplay", "afplay", "play"):
            if shutil.which(candidate) is not None:
                return candidate
        return None

    def _sound_loop(self) -> None:
        """Watch ``bb.replan_event`` and fire the sound on new increments."""
        while not self._sound_stop.is_set():
            try:
                counter = int(self.bb.replan_event)
            except Exception:
                counter = self._last_replan_event_seen
            if counter != self._last_replan_event_seen:
                self._last_replan_event_seen = counter
                self._play_sound()
            self._sound_stop.wait(0.05)

    def _play_sound(self) -> None:
        now = time.monotonic()
        if now - self._last_sound_t < self._sound_cooldown_s:
            return
        if not self._sound_player or not self._sound_path:
            return
        try:
            subprocess.Popen(
                [self._sound_player, self._sound_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self._last_sound_t = now
        except Exception:
            pass

    def _handle_command(self, cmd: dict) -> None:
        """Receive a pending_command from the BT and forward to the user's
        send_cmd callback. EMA-smooths the joint list then rate-clamps the
        per-tick joint deltas so big replans (e.g. escape_collision) walk to
        their target gradually instead of snapping in one tick."""
        joints = list(cmd.get("joints", []))
        delta = int(cmd.get("delta", 3))
        bypass = bool(cmd.get("bypass_smoothing", False))
        if not joints:
            return
        if bypass:
            # Emergency retreat etc. — skip EMA + rate clamp so the command
            # lands in one tick. Reset the EMA so it re-bootstraps from the
            # snapped pose instead of lagging the arm back toward the old
            # state on the next normal tick.
            smoothed_int = list(joints)
            self.joint_ema.reset(
                initial=np.array(smoothed_int, dtype=np.float64),
            )
        else:
            prev = np.array(self.bb.current_q, dtype=np.float64)
            try:
                smoothed = self.joint_ema.update(
                    np.array(joints, dtype=np.float64),
                )
                if prev.shape == smoothed.shape:
                    smoothed = self.joint_rate_clamp.apply(prev, smoothed)
                smoothed_int = [int(round(float(v))) for v in smoothed]
            except Exception:
                smoothed_int = joints
        self.bb.current_q = smoothed_int
        try:
            self._send_cmd_user(smoothed_int, delta)
        except Exception:
            # The controller's serial bridge should survive; reset the EMA
            # on unexpected errors so we don't accumulate bad state.
            self.joint_ema.reset()


__all__ = [
    "SafetyAPI",
    "SafetyAPIConfig",
    "TofProjector",
]
