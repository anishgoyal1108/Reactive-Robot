"""
web.backend — FastAPI server wrapping the Phase 2 DSL runtime.

Public entry points:

    create_app(bridge=None) -> fastapi.FastAPI
        Build the FastAPI application. If no bridge is supplied, a
        default WebBridge is constructed with an in-memory WebBackend
        and the committed state library. Tests construct their own
        WebBridge so they can stub out the backend / state store.

    WebBridge
        Owns one BraccioBackend, one StateLibrary, one sequences
        store, and the currently-running SequenceRunner. Every REST
        handler and WebSocket consumer drives the arm through a
        WebBridge so that swapping the backend (Phase 7) is a single
        constructor-argument change.

    WebBackend
        Thin in-memory BraccioBackend used when the server is running
        outside of PyBullet / real hardware. Parses the same ASCII
        command grammar SerialBridge speaks, updates joint targets in
        a threadsafe dict, and exposes a direct joint_snapshot() for
        the telemetry pump.
"""

from .app import create_app
from .bridge import WebBridge
from .web_backend import WebBackend

__all__ = ["create_app", "WebBridge", "WebBackend"]
