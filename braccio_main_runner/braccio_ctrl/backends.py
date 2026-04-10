"""
backends.py — Abstract seam between the controller and whatever is moving.

The controller/auto-sweep code used to talk directly to a pyserial-backed
SerialBridge. That tied the control logic to real hardware. This module
introduces a small Protocol that captures the exact surface the controller
needs, so the same control loop can also drive a PyBullet digital twin or
any other implementation that honours the contract.

Design
------
BraccioBackend is a typing.Protocol — it is purely structural, so existing
classes like SerialBridge satisfy it without inheritance. This keeps the
Phase 0 refactor risk-free: no runtime code changes, only type hints that
now accept "anything that looks like a bridge".

The two production implementations are:

    HardwareBackend  — alias for SerialBridge; talks to a real Arduino
    SimBackend       — lives in braccio_twin.sim_backend; drives PyBullet

Both expose the same `connect()/ping()/send_cmd()/response_queue/...`
surface, so the controller, AutoSweeper, and any future consumer can hold
a `BraccioBackend` reference and be oblivious to the target.
"""

from __future__ import annotations

import queue
from typing import Any, Protocol, runtime_checkable

from .serial_bridge import SerialBridge


@runtime_checkable
class BraccioBackend(Protocol):
    """
    Structural interface for anything that can drive the Braccio arm.

    Implementations must expose:

    Attributes
    ----------
    response_queue : queue.Queue
        Parsed response dicts (see protocol.parse_response) pushed by the
        backend's reader thread. The controller drains this on every tick.
    connect_error : str
        Populated on a failed connect() so the controller can surface a
        human-readable hint. Empty string on success.

    Methods
    -------
    connect()         Open the underlying transport; return True on success.
    ping(timeout)     Verify the target responds with PONG.
    is_open()         Whether the transport is currently open.
    close()           Tear the transport down.
    send_cmd(cmd)     Write an ASCII command (must end with '\\n').
    """

    response_queue: "queue.Queue[dict[str, Any]]"
    connect_error: str

    def connect(self) -> bool: ...
    def ping(self, timeout: float = 1.5) -> bool: ...
    def is_open(self) -> bool: ...
    def close(self) -> None: ...
    def send_cmd(self, cmd: str) -> None: ...


# ── Production: hardware-backed ──────────────────────────────────────────
# HardwareBackend is just SerialBridge; the alias exists to make calling
# sites read as "which implementation are we instantiating?" rather than
# leaking the transport mechanism (pyserial) into controller-layer code.
HardwareBackend = SerialBridge


__all__ = ["BraccioBackend", "HardwareBackend"]
