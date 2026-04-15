"""
bridge.py — The FastAPI server's session owner.

A ``WebBridge`` ties together the three things every REST/WebSocket
handler needs to touch:

  1. A BraccioBackend (``WebBackend`` by default) — where send_cmd
     goes and where joint/sensor snapshots come from.
  2. A ``StateLibrary`` — the single source of truth for named poses.
     Phase 5's Blockly dropdown reads from this exact store, and the
     curses editor writes to it, so both UIs stay in sync through the
     file on disk.
  3. A file-backed sequence store (``sequences_dir/``) — DSL text
     files persisted alongside the controller. The Phase 2 DSL is the
     lingua franca here: both the web frontend and the curses editor
     can load the same file and run it.

On top of that, the bridge owns the currently-running sequence. We
use the Phase 2 ``Interpreter`` directly (not ``SequenceRunner``,
which is built for the curses editor's run_state_fn callback) so we
can plug in an executor that speaks to the BraccioBackend.

Telemetry model
---------------
Subscribers are added via ``add_subscriber(callable)`` and removed
via ``remove_subscriber(sub_id)``. Every subscriber callable receives
a dict with shape::

    {"type": "telemetry",
     "joints": [90, 90, 90, 90, 90, 73],
     "tof":    {"0": 9999.0, "1": 9999.0, "2": 9999.0, "3": 9999.0},
     "ir":     0}

The bridge does NOT start a background broadcast thread on its own.
Instead, the FastAPI WebSocket endpoint in ``app.py`` runs its own
async loop that polls ``snapshot()`` at the desired rate and sends
it to its connected client. This keeps ``WebBridge`` pure and lets
unit tests exercise every branch without an event loop.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from braccio_ctrl.backends import BraccioBackend
from braccio_ctrl.constants import JOINT_TOKENS, HOME_POS
from braccio_ctrl.dsl import (
    BaseExecutor,
    Interpreter,
    Program,
    parse_auto,
)
from braccio_ctrl.safety import SafetyAPI
from braccio_ctrl.sequence_editor import _static_check
from braccio_ctrl.state_library import StateLibrary

from .web_backend import WebBackend


# ── Mode constants ─────────────────────────────────────────────────────
#
# Phase 7's "Run on sim / Run on hardware" toggle. The bridge exposes
# these as its public mode names; the frontend uses them verbatim when
# calling POST /mode.
MODE_SIM = "sim"
MODE_HARDWARE = "hardware"
VALID_MODES = (MODE_SIM, MODE_HARDWARE)

# Type of the factory the bridge uses to mint a backend for a given
# mode. Tests override this with a factory that returns a fresh
# WebBackend for each mode (so the "hardware" branch doesn't actually
# need a serial port).
BackendFactory = Callable[[str], BraccioBackend]


def default_backend_factory(mode: str) -> BraccioBackend:
    """
    Default factory used when no override is supplied.

    * ``sim``      → ``WebBackend`` (in-memory, no PyBullet load).
    * ``hardware`` → raises ``RuntimeError`` — production deployments
      must wire a ``SerialBridge``-based factory through
      ``WebBridge(backend_factory=...)`` because the FastAPI process
      does not know the user's serial port up front.

    Tests and the default FastAPI app never hit the hardware branch
    because they either preserve the existing backend or supply their
    own factory.
    """
    if mode == MODE_SIM:
        return WebBackend()
    if mode == MODE_HARDWARE:
        raise RuntimeError(
            "hardware mode requires a backend_factory that constructs "
            "a SerialBridge (the default factory cannot open a serial "
            "port without knowing the device path)"
        )
    raise ValueError(f"unknown backend mode: {mode!r}")


_TOKEN_TO_IDX: Dict[str, int] = {tok: i for i, tok in enumerate(JOINT_TOKENS)}


# ── Executor wired to a BraccioBackend ──────────────────────────────────


class BackendExecutor(BaseExecutor):
    """
    DSL Executor that translates AST primitives into ASCII commands on
    a ``BraccioBackend``.

    When ``safety_api`` is supplied every motion-affecting primitive
    (``move_to_state``, ``go_home``, ``set_joint`` on a major axis) is
    validated through ``SafetyAPI.plan_and_validate`` before hitting the
    backend. Failures raise a ``RuntimeError`` the interpreter surfaces to
    the web UI as a run-failure event. When ``safety_api`` is ``None`` the
    executor falls back to the legacy direct-send path — kept for the unit
    tests that mock out the backend without wiring a full safety stack.
    """

    def __init__(
        self,
        backend: BraccioBackend,
        state_lib: StateLibrary,
        stop_event: threading.Event,
        safety_api: Optional[SafetyAPI] = None,
    ) -> None:
        self._backend = backend
        self._state_lib = state_lib
        self._stop = stop_event
        self._safety = safety_api
        # Shadow of where we believe the arm is. Updated on every
        # successful move so the next plan() call starts from the
        # previous endpoint.
        self._current_q: List[int] = list(HOME_POS)
        # Best-effort sync from the backend's joint snapshot, if any.
        snap_fn = getattr(self._backend, "joint_snapshot", None)
        if callable(snap_fn):
            try:
                self._current_q = [int(x) for x in snap_fn()]
            except Exception:
                pass

    def _send_path(self, path: List[List[int]]) -> None:
        """Walk a planner-returned path one SET ALL at a time."""
        for step in path:
            if self._stop.is_set():
                return
            cmd = "SET ALL " + " ".join(str(int(j)) for j in step) + "\n"
            self._backend.send_cmd(cmd)
            self._current_q = [int(j) for j in step]

    def move_to_state(self, name: str) -> None:
        state = self._state_lib.get_state(name)
        if state is None:
            raise ValueError(f"unknown state: {name}")
        joints = [int(x) for x in state["joints"]]
        if len(joints) != 6:
            raise ValueError(
                f"state {name} has {len(joints)} joints, expected 6"
            )
        if self._safety is not None:
            result = self._safety.plan_and_validate(self._current_q, joints)
            if not result.success:
                raise RuntimeError(
                    f"safety: refused move_to_state({name!r}) — "
                    f"{result.failure_reason}"
                )
            self._send_path(result.path or [joints])
            return
        cmd = "SET ALL " + " ".join(str(j) for j in joints) + "\n"
        self._backend.send_cmd(cmd)
        self._current_q = joints

    def set_joint(self, joint_token: str, deg: int) -> None:
        if joint_token not in _TOKEN_TO_IDX:
            raise ValueError(f"unknown joint token: {joint_token}")
        idx = _TOKEN_TO_IDX[joint_token]
        # Wrist rotation and gripper never move the arm's collision
        # envelope, so they skip the planner. Base/shoulder/elbow/wrist-
        # vertical are routed through plan_and_validate when safety is
        # configured.
        if self._safety is not None and idx in (0, 1, 2, 3):
            new_q = list(self._current_q)
            new_q[idx] = int(deg)
            result = self._safety.plan_and_validate(self._current_q, new_q)
            if not result.success:
                raise RuntimeError(
                    f"safety: refused set_joint({joint_token!r}, {deg}) — "
                    f"{result.failure_reason}"
                )
            self._send_path(result.path or [new_q])
            return
        cmd = f"SET {joint_token} {deg}\n"
        self._backend.send_cmd(cmd)
        if idx is not None and 0 <= idx < 6:
            self._current_q[idx] = int(deg)

    def go_home(self) -> None:
        if self._safety is not None:
            result = self._safety.plan_and_validate(
                self._current_q, list(HOME_POS)
            )
            if not result.success:
                raise RuntimeError(
                    f"safety: refused HOME — {result.failure_reason}"
                )
            self._send_path(result.path or [list(HOME_POS)])
            return
        self._backend.send_cmd("HOME\n")
        self._current_q = list(HOME_POS)

    def wait_ms(self, ms: int) -> None:
        deadline = time.monotonic() + ms / 1000.0
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return
            time.sleep(0.02)

    def read_tof(self, channel: int) -> float:
        reader = getattr(self._backend, "read_tof", None)
        if callable(reader):
            return float(reader(channel))
        return 9999.0

    def read_ir(self) -> int:
        reader = getattr(self._backend, "read_ir", None)
        if callable(reader):
            return int(reader())
        return 0

    def read_joint(self, joint_token: str) -> int:
        idx = _TOKEN_TO_IDX.get(joint_token)
        if idx is None:
            raise ValueError(f"unknown joint token: {joint_token}")
        snap = getattr(self._backend, "joint_snapshot", None)
        if callable(snap):
            js = snap()
            if len(js) > idx:
                return int(js[idx])
        return 0

    def define_state(self, name: str, joints: tuple) -> None:
        if len(joints) != 6:
            raise ValueError(
                f"STATE {name}: expected 6 joint angles, got {len(joints)}"
            )
        snapshot = {
            "joints": list(int(x) for x in joints),
            "theta": 0.0,
            "r": 0.0,
            "z": 0.0,
            "wrist_offset": 0.0,
            "wrist_rot": 0,
            "gripper": int(joints[5]),
        }
        self._state_lib.save_state(name, snapshot)

    def define_obstacle(
        self, name: str, pos: tuple, radius: float, shape: str
    ) -> None:
        """Register a DSL-declared obstacle with the safety world model.

        When a SafetyAPI is wired, the obstacle is seeded as a cluster of
        points around ``pos`` so the capsule collision checker sees it on
        every subsequent ``plan_and_validate`` call. When there is no
        safety stack, this is a no-op (same legacy behaviour).

        If the underlying backend exposes an ``obstacle_world`` attribute
        (SimBackend does), the obstacle is *also* spawned as a PyBullet
        body so the virtual ToF rays can hit it.
        """
        if self._safety is not None:
            import numpy as _np
            # Sample points on the surface of the obstacle — enough density
            # that capsule checks see it as more than a single point.
            x, y, z = (float(v) for v in pos)
            r = float(radius)
            n_samples = 14
            rng = _np.random.default_rng(seed=abs(hash(name)) & 0xFFFFFFFF)
            dirs = rng.normal(size=(n_samples, 3))
            dirs /= _np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
            pts = _np.column_stack([
                x + dirs[:, 0] * r,
                y + dirs[:, 1] * r,
                z + dirs[:, 2] * r,
            ])
            self._safety.world.ingest_points(pts, confidence=1.0)
        world_attr = getattr(self._backend, "obstacle_world", None)
        if world_attr is None:
            getter = getattr(self._backend, "get_obstacle_world", None)
            if callable(getter):
                try:
                    world_attr = getter()
                except Exception:
                    world_attr = None
        if world_attr is not None:
            try:
                # Late import — ObstacleSpec lives under the digital-twin
                # package which isn't always importable in CI.
                from braccio_twin.obstacle_world import ObstacleSpec  # type: ignore
                world_attr.add(ObstacleSpec(
                    name=name, shape=shape,
                    position_mm=tuple(float(v) for v in pos),
                    size_mm=(float(radius), float(radius), float(radius)),
                ))
            except Exception:
                pass

    def is_stopped(self) -> bool:
        return self._stop.is_set()


# ── Bridge ───────────────────────────────────────────────────────────────


_VALID_NAME_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


def _is_valid_name(name: str) -> bool:
    """Reject names with separators so clients can't path-traverse."""
    if not name:
        return False
    if name in (".", "..") or name.startswith("."):
        return False
    return all(ch in _VALID_NAME_CHARS for ch in name)


class WebBridge:
    """
    Thread-safe facade the FastAPI handlers talk to.

    Construction parameters
    -----------------------
    backend         : BraccioBackend (defaults to a fresh WebBackend)
    state_lib       : StateLibrary   (defaults to the committed states.json)
    sequences_dir   : Path           (defaults to web/backend/sequences/)
    mode            : str            ("sim" | "hardware"). Sets the
                      initial mode label reported via ``current_mode``
                      and POST /mode. Does NOT select the ``backend``
                      argument — the caller passes that explicitly so
                      tests stay in control.
    backend_factory : BackendFactory (defaults to ``default_backend_factory``)
                      Used by ``set_mode`` to construct a fresh backend
                      when the user flips between sim and hardware.
                      Production deploys that want "Deploy to hardware"
                      to actually reach a real arm must supply a factory
                      that knows the serial port.

    The constructor opens the backend immediately so the server is
    ready to accept commands at import time. ``close()`` should be
    called from the FastAPI shutdown event.

    ─── Safety wiring (post-refactor) ─────────────────────────────────
    When ``enable_safety=True`` (the default), the bridge constructs its
    own ``SafetyAPI`` and hands it to every ``BackendExecutor`` it
    spawns. Every MoveStmt and IK-axis SetJointStmt is validated through
    ``SafetyAPI.plan_and_validate`` before the serial command lands on
    the backend, so the web path and the curses controller now share the
    same synchronous-safety contract.
    """

    def __init__(
        self,
        backend: Optional[BraccioBackend] = None,
        state_lib: Optional[StateLibrary] = None,
        sequences_dir: Optional[Path] = None,
        mode: str = MODE_SIM,
        backend_factory: Optional[BackendFactory] = None,
        enable_safety: bool = True,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(
                f"invalid mode {mode!r}; must be one of {VALID_MODES}"
            )
        self._backend: BraccioBackend = backend or WebBackend()
        self._state_lib: StateLibrary = state_lib or StateLibrary()
        self._sequences_dir: Path = (
            sequences_dir or Path(__file__).resolve().parent / "sequences"
        )
        self._sequences_dir.mkdir(parents=True, exist_ok=True)

        self._mode: str = mode
        self._backend_factory: BackendFactory = (
            backend_factory or default_backend_factory
        )
        self._mode_lock = threading.RLock()

        # Safety stack — one instance per bridge. plan_and_validate() is
        # called synchronously from BackendExecutor; the BT daemon is NOT
        # started here because the web path drives motion imperatively
        # (one plan per DSL primitive), not tick-reactively like curses.
        self._safety: Optional[SafetyAPI] = None
        if enable_safety:
            try:
                self._safety = SafetyAPI(
                    send_cmd=lambda _q, _d: None,  # BT unused for web
                )
            except Exception:
                # A stack-construction failure (missing URDF, pybullet
                # import error in a restricted env) must not kill the
                # web server. Fall back to the legacy unsafe path.
                self._safety = None

        # Currently-running interpreter + thread.
        self._run_lock = threading.RLock()
        self._run_thread: Optional[threading.Thread] = None
        self._run_interp: Optional[Interpreter] = None
        self._run_stop = threading.Event()
        self._run_status: Dict[str, Any] = {
            "running": False,
            "line": 0,
            "kind": "",
            "depth": 0,
            "pass_current": 0,
            "pass_total": 1,
            "errors": [],
        }

        # Telemetry subscribers (for future async-broadcast use cases).
        # The WebSocket endpoint in app.py uses snapshot() directly and
        # does not need to register here, but tests and background
        # publishers can.
        self._subs_lock = threading.Lock()
        self._subs: Dict[int, Callable[[Dict[str, Any]], None]] = {}
        self._next_sub_id = 1

        # Open the backend so ``send_cmd`` is legal.
        self._backend.connect()

    # ── Backend accessors ────────────────────────────────────────────────

    @property
    def backend(self) -> BraccioBackend:
        return self._backend

    @property
    def state_library(self) -> StateLibrary:
        return self._state_lib

    @property
    def sequences_dir(self) -> Path:
        return self._sequences_dir

    @property
    def current_mode(self) -> str:
        """Return the active mode label ("sim" or "hardware")."""
        with self._mode_lock:
            return self._mode

    # ── Mode switching ───────────────────────────────────────────────────

    def set_mode(self, mode: str) -> Tuple[bool, str]:
        """
        Swap the active backend to match *mode*.

        Steps:
          1. Reject unknown mode values early.
          2. If already in the requested mode, return without touching
             the backend (idempotent).
          3. Stop any in-flight sequence (safer than tearing the
             backend out from under a running interpreter).
          4. Close the current backend.
          5. Mint a new one via ``backend_factory``. If that raises,
             restore the previous backend so the server stays usable
             even when the user tried to flip to a mode the factory
             cannot satisfy (e.g. no serial port attached).
          6. Connect the new backend and swap it in under the lock.

        Returns ``(ok, detail)``. ``detail`` is a human-readable error
        message on failure and an empty string on success.
        """
        if mode not in VALID_MODES:
            return False, f"invalid mode {mode!r}; must be one of {VALID_MODES}"

        with self._mode_lock:
            if mode == self._mode:
                return True, ""

            # Don't pull the rug out from under a running interpreter.
            self.stop_sequence()

            old_backend = self._backend
            try:
                old_backend.close()
            except Exception:
                # Closing the old backend must never block the swap —
                # at worst we leak one WebBackend which is harmless.
                pass

            try:
                new_backend = self._backend_factory(mode)
            except Exception as exc:
                # Factory blew up — keep the old backend so the server
                # stays functional. We have to re-open it since we
                # closed it above.
                try:
                    old_backend.connect()
                except Exception:
                    pass
                self._backend = old_backend
                return False, f"failed to construct {mode} backend: {exc}"

            try:
                new_backend.connect()
            except Exception as exc:
                try:
                    old_backend.connect()
                except Exception:
                    pass
                self._backend = old_backend
                return False, f"failed to connect {mode} backend: {exc}"

            self._backend = new_backend
            self._mode = mode
            return True, ""

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop any running sequence and close the backend."""
        self.stop_sequence()
        try:
            self._backend.close()
        except Exception:
            pass

    # ── State library ───────────────────────────────────────────────────

    def list_states(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name in self._state_lib.names():
            st = self._state_lib.get_state(name) or {}
            out.append({"name": name, **st})
        return out

    def get_state(self, name: str) -> Optional[Dict[str, Any]]:
        return self._state_lib.get_state(name)

    def save_state(self, name: str, snapshot: Dict[str, Any]) -> None:
        if not _is_valid_name(name):
            raise ValueError(f"invalid state name: {name!r}")
        self._state_lib.save_state(name, snapshot)

    def delete_state(self, name: str) -> bool:
        return self._state_lib.delete_state(name)

    # ── Sequences (filesystem-backed) ────────────────────────────────────

    def list_sequences(self) -> List[str]:
        return sorted(
            p.stem for p in self._sequences_dir.glob("*.txt") if p.is_file()
        )

    def get_sequence(self, name: str) -> Optional[str]:
        if not _is_valid_name(name):
            return None
        path = self._sequences_dir / f"{name}.txt"
        if not path.is_file():
            return None
        return path.read_text()

    def save_sequence(self, name: str, text: str) -> Tuple[bool, List[str]]:
        """
        Validate *text* with the Phase 2 parser + static checker, and
        if it's clean, write it to ``sequences_dir/<name>.txt``.

        Returns ``(ok, errors)``. On parse failure the file is *not*
        written so the on-disk store only ever contains runnable
        sequences.
        """
        if not _is_valid_name(name):
            return False, [f"invalid sequence name: {name!r}"]

        program, parse_errs = parse_auto(text)
        if parse_errs:
            return False, list(parse_errs)

        static_errs = _static_check(program, self._state_lib)
        if static_errs:
            return False, list(static_errs)

        path = self._sequences_dir / f"{name}.txt"
        path.write_text(text)
        return True, []

    def delete_sequence(self, name: str) -> bool:
        if not _is_valid_name(name):
            return False
        path = self._sequences_dir / f"{name}.txt"
        if not path.is_file():
            return False
        path.unlink()
        return True

    # ── Run / stop the DSL interpreter ──────────────────────────────────

    def validate_program(
        self, text: str
    ) -> Tuple[Optional[Program], List[str]]:
        """Parse + static-check a DSL string without running it."""
        program, parse_errs = parse_auto(text)
        if parse_errs:
            return None, list(parse_errs)
        static_errs = _static_check(program, self._state_lib)
        if static_errs:
            return program, list(static_errs)
        return program, []

    def run_program(self, text: str) -> Tuple[bool, List[str]]:
        """
        Parse, validate, and spawn a background thread that runs the
        program through a ``BackendExecutor``. If one is already
        running, this refuses (caller should stop first).

        Returns ``(started, errors)``. ``errors`` is non-empty when
        validation fails or when a previous run is still in flight.
        """
        with self._run_lock:
            if self.is_running():
                return False, ["a sequence is already running; stop it first"]

            program, errs = self.validate_program(text)
            if errs or program is None:
                return False, errs

            self._run_stop.clear()
            executor = BackendExecutor(
                self._backend, self._state_lib, self._run_stop,
                safety_api=self._safety,
            )
            interp = Interpreter(executor, status_cb=self._on_status)
            self._run_interp = interp
            self._run_status = {
                "running": True,
                "line": 0,
                "kind": "",
                "depth": 0,
                "pass_current": 0,
                "pass_total": 1,
                "errors": [],
            }

            def _worker() -> None:
                try:
                    interp.run(program)
                finally:
                    with self._run_lock:
                        self._run_status["running"] = False
                        self._run_status["errors"] = list(interp.errors)

            thread = threading.Thread(
                target=_worker, name="web-bridge-run", daemon=True
            )
            self._run_thread = thread
            thread.start()
            return True, []

    def stop_sequence(self) -> bool:
        """Signal the currently-running interpreter (if any) to stop."""
        with self._run_lock:
            if self._run_interp is None:
                return False
            self._run_interp.stop()
            self._run_stop.set()
        # Join outside the lock to avoid deadlocking with the worker.
        thread = self._run_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._run_lock:
            self._run_thread = None
            self._run_interp = None
            self._run_status["running"] = False
        return True

    def is_running(self) -> bool:
        with self._run_lock:
            return bool(self._run_status.get("running", False))

    def run_status(self) -> Dict[str, Any]:
        with self._run_lock:
            return dict(self._run_status)

    # ── Telemetry snapshot / subscribers ────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a complete telemetry frame: joint angles, sensor state,
        running-sequence status. The WebSocket endpoint polls this at
        ~30 Hz and JSON-serialises the result to its client.
        """
        joints: List[int]
        snap_fn = getattr(self._backend, "joint_snapshot", None)
        if callable(snap_fn):
            joints = list(snap_fn())
        else:
            joints = []

        sensors: Dict[str, Any]
        sens_fn = getattr(self._backend, "sensor_snapshot", None)
        if callable(sens_fn):
            sensors = sens_fn()
        else:
            sensors = {"tof": {}, "ir": 0}

        return {
            "type": "telemetry",
            "joints": joints,
            "tof": sensors.get("tof", {}),
            "ir": int(sensors.get("ir", 0)),
            "status": self.run_status(),
        }

    def add_subscriber(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> int:
        with self._subs_lock:
            sub_id = self._next_sub_id
            self._next_sub_id += 1
            self._subs[sub_id] = callback
        return sub_id

    def remove_subscriber(self, sub_id: int) -> None:
        with self._subs_lock:
            self._subs.pop(sub_id, None)

    def _broadcast(self, frame: Dict[str, Any]) -> None:
        with self._subs_lock:
            subs = list(self._subs.values())
        for cb in subs:
            try:
                cb(frame)
            except Exception:
                # Never let a misbehaving subscriber crash the run.
                pass

    # ── Interpreter status callback ─────────────────────────────────────

    def _on_status(self, info: Dict[str, Any]) -> None:
        cp, tp = info.get("pass", (0, 0))
        with self._run_lock:
            self._run_status.update(
                {
                    "running": True,
                    "line": int(info.get("line", 0)),
                    "kind": str(info.get("kind", "")),
                    "depth": int(info.get("depth", 0)),
                    "pass_current": int(cp),
                    "pass_total": int(tp) or 1,
                }
            )
        self._broadcast(
            {
                "type": "status",
                "line": int(info.get("line", 0)),
                "kind": str(info.get("kind", "")),
                "depth": int(info.get("depth", 0)),
                "pass_current": int(cp),
                "pass_total": int(tp) or 1,
            }
        )


__all__ = [
    "WebBridge",
    "BackendExecutor",
    "MODE_SIM",
    "MODE_HARDWARE",
    "VALID_MODES",
    "BackendFactory",
    "default_backend_factory",
]
