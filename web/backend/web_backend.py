"""
web_backend.py — Lightweight in-memory BraccioBackend for the web server.

The real controller has two BraccioBackend implementations today:

  * ``SerialBridge``  — talks to a real Arduino over pyserial
  * ``SimBackend``    — drives a PyBullet simulation

Neither is appropriate as the *default* target for the FastAPI
server. PyBullet is a heavy dependency that we don't want every test
to load, and the real arm is not always plugged in during
development. ``WebBackend`` fills that gap: it implements the exact
same ``BraccioBackend`` Protocol (``response_queue`` / ``connect_error`` +
``connect/ping/is_open/close/send_cmd``) but keeps joint state in a
plain dict, with zero external dependencies.

Concretely, ``WebBackend`` will:

  * Parse the four commands the controller actually emits
    (``SET ALL ...``, ``SET <joint> <deg>``, ``SET DELTA n``, ``HOME``,
    ``GET POS``, ``PING``) using the same regexes ``SimBackend`` does.
  * Clamp all incoming angles to the Braccio servo limits so that a
    malformed sequence can't produce out-of-range joint values.
  * Track a synthetic ToF / IR snapshot that REST endpoints (and the
    Phase 5 frontend) can poke into to exercise IF / WHILE conditional
    branches during a run — useful for demos where there is no real
    sensor rig attached.
  * Expose direct ``joint_snapshot()`` / ``sensor_snapshot()`` methods
    so the Phase 3 WebSocket telemetry pump can stream data without
    draining the ``response_queue``, which is reserved for the
    BraccioBackend Protocol contract.

This file stays independent of ``braccio_ctrl.controller`` so importing
it does not pull in the curses TUI, matplotlib, or any other heavy
subsystem. The only cross-module import is from ``braccio_ctrl.constants``
for ``JOINT_TOKENS`` / ``HOME_POS``, which is a tiny pure-Python file.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Any, Dict, List

from braccio_ctrl.constants import HOME_POS, JOINT_LIMITS, JOINT_TOKENS


# ── Command regexes (identical to SimBackend for drop-in compatibility) ─

_RE_SET_ALL = re.compile(
    r"^SET ALL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)
_RE_SET_JOINT = re.compile(r"^SET\s+(B|S|E|WV|WR|G)\s+(-?\d+)\s*$")
_RE_SET_DELTA = re.compile(r"^SET DELTA\s+(\d+)\s*$")

# Token → index in the 6-element joint vector.
_JOINT_TOKEN_TO_IDX: Dict[str, int] = {tok: i for i, tok in enumerate(JOINT_TOKENS)}


def _clamp_joint(idx: int, deg: int) -> int:
    """Clamp a raw joint command to the physical servo limits.

    Reuses the single source of truth in ``braccio_ctrl.constants`` so
    that updating one file propagates to every backend.
    """
    lo, hi = JOINT_LIMITS[idx]
    if deg < lo:
        return lo
    if deg > hi:
        return hi
    return deg


class WebBackend:
    """
    In-memory BraccioBackend used by the FastAPI server by default.

    All public attributes/methods listed below form the BraccioBackend
    Protocol:

        response_queue : queue.Queue[dict[str, Any]]
        connect_error  : str
        connect() / ping() / is_open() / close() / send_cmd(line)

    Extensions unique to WebBackend (used by the bridge + WebSocket):

        joint_snapshot()    -> list[int]   (current 6 joint angles)
        sensor_snapshot()   -> dict        ({"tof": {...}, "ir": int})
        set_tof(channel, mm)               (poke synthetic ToF reading)
        set_ir(severity)                   (poke synthetic IR severity)
        reset()                            (clear joint state to HOME)

    Thread-safety: a single RLock guards both the joint vector and
    the sensor snapshot. The response queue is already thread-safe.
    """

    connect_error: str
    response_queue: "queue.Queue[dict[str, Any]]"

    # Cosmetic: the existing controller uses ``_port`` for error strings.
    _port: str = "web://in-memory"

    def __init__(self) -> None:
        self.connect_error = ""
        self.response_queue = queue.Queue(maxsize=256)

        self._lock = threading.RLock()
        self._joints: List[int] = list(HOME_POS)
        self._delta: int = 1
        self._is_open = False

        # Synthetic sensor state. Defaults mean "nothing detected" so
        # IF TOF[0] < 250 { ... } evaluates False until a test or the
        # frontend calls set_tof().
        self._tof: Dict[int, float] = {0: 9999.0, 1: 9999.0, 2: 9999.0, 3: 9999.0}
        self._ir: int = 0

    # ── BraccioBackend Protocol ──────────────────────────────────────────

    def connect(self) -> bool:
        with self._lock:
            self._is_open = True
        self._push({"type": "ready"})
        return True

    def ping(self, timeout: float = 1.5) -> bool:
        self._push({"type": "pong"})
        return True

    def is_open(self) -> bool:
        with self._lock:
            return self._is_open

    def close(self) -> None:
        with self._lock:
            self._is_open = False

    def send_cmd(self, cmd: str) -> None:
        """Parse and apply one ASCII command line."""
        line = cmd.strip()
        if not line:
            return

        if line == "PING":
            self._push({"type": "pong"})
            return

        if line == "HOME":
            with self._lock:
                self._joints = list(HOME_POS)
                snap = list(self._joints)
            self._push({"type": "ok_home", "positions": snap})
            return

        if line == "GET POS":
            self._push({"type": "pos", "positions": self.joint_snapshot()})
            return

        m = _RE_SET_ALL.match(line)
        if m:
            raw = [int(g) for g in m.groups()]
            clamped_vec = [_clamp_joint(i, v) for i, v in enumerate(raw)]
            with self._lock:
                self._joints = clamped_vec
            self._push({"type": "ok_all", "positions": clamped_vec})
            return

        m = _RE_SET_JOINT.match(line)
        if m:
            tok = m.group(1)
            deg = int(m.group(2))
            idx = _JOINT_TOKEN_TO_IDX[tok]
            clamped_deg = _clamp_joint(idx, deg)
            with self._lock:
                self._joints[idx] = clamped_deg
            self._push({"type": "ok_joint", "joint": idx, "deg": clamped_deg})
            return

        m = _RE_SET_DELTA.match(line)
        if m:
            self._delta = max(1, min(5, int(m.group(1))))
            self._push({"type": "ok_delta", "delta": self._delta})
            return

        self._push({"type": "error", "message": f"unknown command: {line}"})

    # ── Extensions used by bridge / WebSocket ────────────────────────────

    def joint_snapshot(self) -> List[int]:
        """Return a copy of the current 6-joint vector."""
        with self._lock:
            return list(self._joints)

    def sensor_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the synthetic sensor state."""
        with self._lock:
            return {
                "tof": dict(self._tof),
                "ir": int(self._ir),
            }

    def set_tof(self, channel: int, distance_mm: float) -> None:
        if not 0 <= channel <= 3:
            raise ValueError(f"tof channel must be 0..3, got {channel}")
        with self._lock:
            self._tof[channel] = float(distance_mm)

    def set_ir(self, severity: int) -> None:
        if not 0 <= severity <= 3:
            raise ValueError(f"ir severity must be 0..3, got {severity}")
        with self._lock:
            self._ir = int(severity)

    def read_tof(self, channel: int) -> float:
        with self._lock:
            return float(self._tof.get(channel, 9999.0))

    def read_ir(self) -> int:
        with self._lock:
            return int(self._ir)

    def reset(self) -> None:
        with self._lock:
            self._joints = list(HOME_POS)
            self._delta = 1
            self._tof = {0: 9999.0, 1: 9999.0, 2: 9999.0, 3: 9999.0}
            self._ir = 0
        # Drain any stale responses so a subsequent connect()/ping()
        # sees a clean queue.
        while True:
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break

    # ── Internals ────────────────────────────────────────────────────────

    def _push(self, resp: Dict[str, Any]) -> None:
        try:
            self.response_queue.put_nowait(resp)
        except queue.Full:
            try:
                self.response_queue.get_nowait()
                self.response_queue.put_nowait(resp)
            except queue.Empty:
                pass


__all__ = ["WebBackend"]
