"""
sim_backend.py — BraccioBackend implementation driven by PyBullet.

SimBackend is the twin-side seam that drops in where SerialBridge lives
on the real robot. It honours the BraccioBackend Protocol exactly:

    connect() / close() / is_open() / ping() / send_cmd()
    response_queue  (parsed response dicts, mirroring Arduino replies)
    connect_error   (populated if PyBullet import/init fails)

Internally it runs one daemon thread at a fixed step rate:

    tick_loop():
        1. SimArm.step()          — advance physics
        2. VirtualToF.sample_once() — refresh ToF grids
        3. VirtualIR.sample_once()  — refresh IR debounced bits
        4. VirtualIMU.sample_once() — refresh orientation
        5. sleep until next tick

send_cmd() parses the ASCII string (SET ALL / SET DELTA / HOME / GET POS
/ PING) and updates the joint targets or pushes a response, just like a
real Arduino would.

SimSensorBridge is a tiny wrapper that satisfies the ToFBridge-ish
"connect/close" surface so BraccioController can treat it uniformly.
Actual sampling is done by the SimBackend tick thread; SimSensorBridge
just gives the controller something to call `.connect()` / `.close()`
on, so no code changes are needed there.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from typing import Any, Optional

from braccio_ctrl.imu_state import IMUState
from braccio_ctrl.tof_sensor import ToFState

from .obstacle_world import ObstacleWorld
from .sim_arm import SimArm
from .virtual_imu import VirtualIMU
from .virtual_ir import VirtualIR
from .virtual_tof import VirtualToF


# How often the physics / sensor loop runs. 100 Hz matches the hardware's
# "~one loop tick, ~100 ms" reactive guarantee documented in CLAUDE.md.
SIM_TICK_HZ = 100.0


# ── Command parser ─────────────────────────────────────────────────────
# Matches cmd_set_all / cmd_set_joint / cmd_set_delta / cmd_home / cmd_get_pos
# / cmd_ping outputs from braccio_ctrl.protocol.

_RE_SET_ALL = re.compile(r"^SET ALL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
_RE_SET_JOINT = re.compile(r"^SET\s+(B|S|E|WV|WR|G)\s+(-?\d+)\s*$")
_RE_SET_DELTA = re.compile(r"^SET DELTA\s+(\d+)\s*$")

_JOINT_TOKEN_TO_IDX = {"B": 0, "S": 1, "E": 2, "WV": 3, "WR": 4, "G": 5}


class SimBackend:
    """
    BraccioBackend driving a PyBullet arm instead of a real Arduino.

    Parameters
    ----------
    gui        : True → open a PyBullet GUI window
    urdf_path  : override the default URDF location
    """

    # These two fields are part of the BraccioBackend Protocol contract.
    connect_error: str
    response_queue: "queue.Queue[dict[str, Any]]"

    # Cosmetic: BraccioController uses this for error messages.
    _port: str = "pybullet://sim"

    def __init__(self, gui: bool = False, urdf_path: Optional[str] = None):
        self.connect_error = ""
        self.response_queue = queue.Queue(maxsize=256)

        self._gui = gui
        self._urdf_path = urdf_path

        self._sim_arm: Optional[SimArm] = None
        self._tof_sensor: Optional[VirtualToF] = None
        self._ir_sensor: Optional[VirtualIR] = None
        self._imu_sensor: Optional[VirtualIMU] = None
        self._world: Optional[ObstacleWorld] = None

        # Shared sensor state refs — wired in by _install_sensor_state().
        self._tof_state: Optional[ToFState] = None
        self._imu_state: Optional[IMUState] = None

        # Current joint targets (Braccio degrees). Initialised to HOME.
        self._targets: list[int] = [90, 90, 90, 90, 90, 73]
        self._targets_lock = threading.Lock()

        # Delta / slew — tracked only so GET POS can echo it. The sim
        # applies PyBullet's POSITION_CONTROL with a fixed velocity cap,
        # so delta is advisory for the digital twin.
        self._delta: int = 1

        # Thread control
        self._stop = threading.Event()
        self._tick_thread: Optional[threading.Thread] = None
        self._is_open = False

    # ── Sensor wiring ────────────────────────────────────────────────────

    def install_sensor_state(
        self, tof_state: ToFState, imu_state: IMUState
    ) -> None:
        """
        Hand the SimBackend references to the ToFState/IMUState that the
        controller owns. Must be called BEFORE connect(). The sensor
        sampling threads write into these objects on every tick so the
        existing control loop reads virtual readings as if they came from
        the Teensy.
        """
        self._tof_state = tof_state
        self._imu_state = imu_state

    def get_obstacle_world(self) -> Optional[ObstacleWorld]:
        return self._world

    # ── BraccioBackend Protocol ──────────────────────────────────────────

    def connect(self) -> bool:
        """Boot PyBullet, load the URDF, and start the tick thread."""
        if self._is_open:
            return True
        try:
            # Match physics time-step to the tick loop so wall time and
            # sim time stay roughly synchronised — otherwise the motors
            # crawl at (default 240 Hz × 1/SIM_TICK_HZ) of real speed and
            # commanded moves look stuck.
            self._sim_arm = SimArm(
                urdf_path=self._urdf_path,
                gui=self._gui,
                time_step=1.0 / SIM_TICK_HZ,
            )
        except Exception as exc:
            self.connect_error = f"sim init failed: {exc}"
            return False

        self._world = ObstacleWorld(self._sim_arm)

        if self._tof_state is not None:
            self._tof_sensor = VirtualToF(self._sim_arm, self._tof_state)
            self._ir_sensor = VirtualIR(self._sim_arm, self._tof_state)
        if self._imu_state is not None:
            self._imu_sensor = VirtualIMU(self._sim_arm, self._imu_state)

        self._stop.clear()
        self._tick_thread = threading.Thread(
            target=self._tick_loop,
            name="braccio-sim-tick",
            daemon=True,
        )
        self._tick_thread.start()

        # Emit a READY response so ping() sees it if drained first.
        self._push_response({"type": "ready"})
        self._is_open = True
        return True

    def ping(self, timeout: float = 1.5) -> bool:
        """Push a PONG immediately. No round-trip needed on the sim."""
        self._push_response({"type": "pong"})
        # Give the controller's drain logic a chance to run if it
        # polls very aggressively.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = self.response_queue.queue[0]  # peek without popping
                if resp.get("type") == "pong":
                    return True
            except IndexError:
                pass
            time.sleep(0.01)
        return True  # sim always "responds"

    def is_open(self) -> bool:
        return self._is_open

    def close(self) -> None:
        self._stop.set()
        th = self._tick_thread
        if th is not None and th.is_alive():
            th.join(timeout=1.0)
        self._tick_thread = None
        if self._sim_arm is not None:
            try:
                self._sim_arm.disconnect()
            except Exception:
                pass
            self._sim_arm = None
        self._is_open = False

    def send_cmd(self, cmd: str) -> None:
        """
        Parse one ASCII command and update joint targets / emit response.
        Follows braccio_ctrl.protocol builders exactly.
        """
        line = cmd.strip()
        if not line:
            return

        if line == "PING":
            self._push_response({"type": "pong"})
            return

        if line == "HOME":
            with self._targets_lock:
                self._targets = [90, 90, 90, 90, 90, 73]
            if self._sim_arm is not None:
                self._sim_arm.set_joint_targets(self._targets)
            self._push_response({"type": "ok_home"})
            return

        if line == "GET POS":
            pos = self._sim_arm.get_joint_positions() if self._sim_arm else list(self._targets)
            self._push_response({"type": "pos", "positions": list(pos)})
            return

        m = _RE_SET_ALL.match(line)
        if m:
            vals = [int(g) for g in m.groups()]
            with self._targets_lock:
                self._targets = list(vals)
            if self._sim_arm is not None:
                self._sim_arm.set_joint_targets(vals)
            self._push_response({"type": "ok_all", "positions": vals})
            return

        m = _RE_SET_JOINT.match(line)
        if m:
            token = m.group(1)
            deg = int(m.group(2))
            idx = _JOINT_TOKEN_TO_IDX[token]
            with self._targets_lock:
                self._targets[idx] = deg
                snap = list(self._targets)
            if self._sim_arm is not None:
                self._sim_arm.set_joint_targets(snap)
            self._push_response({"type": "ok_joint", "joint": idx, "deg": deg})
            return

        m = _RE_SET_DELTA.match(line)
        if m:
            self._delta = int(m.group(1))
            self._push_response({"type": "ok_delta", "delta": self._delta})
            return

        self._push_response({"type": "error", "message": f"unknown: {line}"})

    # ── Internals ────────────────────────────────────────────────────────

    def _push_response(self, resp: dict[str, Any]) -> None:
        try:
            self.response_queue.put_nowait(resp)
        except queue.Full:
            try:
                self.response_queue.get_nowait()
                self.response_queue.put_nowait(resp)
            except queue.Empty:
                pass

    def _tick_loop(self) -> None:
        period = 1.0 / SIM_TICK_HZ
        next_wake = time.monotonic()
        tick_count = 0
        while not self._stop.is_set():
            try:
                if self._sim_arm is not None:
                    self._sim_arm.step()
                if self._tof_sensor is not None:
                    self._tof_sensor.sample_once()
                if self._ir_sensor is not None:
                    self._ir_sensor.sample_once()
                if self._imu_sensor is not None:
                    self._imu_sensor.sample_once()
            except Exception:
                # Keep ticking even if one sensor glitches — the sim must
                # never take down the control loop.
                pass

            tick_count += 1
            next_wake += period
            remaining = next_wake - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                # Fell behind — reset the clock so we don't busy-loop.
                next_wake = time.monotonic()


class SimSensorBridge:
    """
    Drop-in stand-in for ToFBridge in sim mode.

    BraccioController calls ``sensor_bridge.connect(port, baud)`` and
    ``.close()`` — that's the entire surface we need to satisfy. Sensor
    sampling is driven by the SimBackend tick thread, so there is no
    reader thread to start here.
    """

    def __init__(self, sim_backend: SimBackend):
        self._backend = sim_backend
        self._last_open_error: str = ""

    def connect(self, port: str = "", baud: int = 0) -> bool:  # noqa: ARG002
        # The SimBackend owns sensor sampling; "connecting" is a no-op
        # once the backend is up. If the backend has not been opened yet
        # (controller calls this before backend.connect()), we still
        # report success because the tick thread will pick up sampling
        # as soon as the backend is live.
        return True

    def close(self) -> None:
        # Nothing to clean up — sampling stops when SimBackend.close() runs.
        pass
