"""
app.py — FastAPI server that wraps a ``WebBridge``.

Run in development with::

    uvicorn web.backend.app:app --reload

The top-level module-level ``app`` object is created with a default
``WebBridge``. Tests import ``create_app`` and construct their own
bridge so every test run gets an isolated in-memory backend and
an ephemeral sequences directory.

Endpoints
---------
Health::
    GET    /health                       -> {"ok": true}

State library::
    GET    /states                       -> {"states": [...]}
    GET    /states/{name}                -> StateEntry
    POST   /states/{name}                -> StateEntry   (upsert)
    DELETE /states/{name}                -> {"deleted": bool}

Sequences (DSL text files)::
    GET    /sequences                    -> {"sequences": [...]}
    GET    /sequences/{name}             -> SequenceGetResponse
    POST   /sequences/{name}             -> SaveSequenceResponse
    DELETE /sequences/{name}             -> {"deleted": bool}

DSL run control::
    POST   /dsl/validate                 -> ValidateResponse
    POST   /dsl/run                      -> RunResponse
    POST   /dsl/stop                     -> {"stopped": bool}
    GET    /dsl/status                   -> RunStatusResponse

Telemetry + sensor pokes::
    GET    /telemetry                    -> TelemetryFrame   (single snapshot)
    POST   /telemetry/tof                -> {"ok": true}     (poke)
    POST   /telemetry/ir                 -> {"ok": true}     (poke)
    WS     /ws/telemetry                 -> streamed TelemetryFrames
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .bridge import WebBridge
from .models import (
    IrPokeRequest,
    RunRequest,
    RunResponse,
    RunStatusResponse,
    SaveSequenceRequest,
    SaveSequenceResponse,
    SaveStateRequest,
    SequenceGetResponse,
    SequenceListResponse,
    StateEntry,
    StateListResponse,
    TelemetryFrame,
    TofPokeRequest,
    ValidateRequest,
    ValidateResponse,
)


# WebSocket broadcast cadence. The browser viewer runs at ~60 Hz but
# joint updates at 30 Hz look smooth and cut the WebSocket traffic in
# half, which matters when several clients are connected.
_WS_HZ = 30.0


def create_app(bridge: Optional[WebBridge] = None) -> FastAPI:
    """Build the FastAPI application, optionally with a test bridge."""

    @asynccontextmanager
    async def _lifespan(instance: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            try:
                instance.state.bridge.close()
            except Exception:
                pass

    app = FastAPI(
        title="Braccio Digital Twin API",
        version="0.3.0",
        description="Phase 3: REST + WebSocket glue around the Phase 2 DSL.",
        lifespan=_lifespan,
    )

    # CORS: Vite dev server runs on localhost:5173 by default.
    # Keep this permissive in development; a production deployment
    # would whitelist only the frontend origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Every handler pulls the bridge off app.state so tests can swap it.
    app.state.bridge = bridge or WebBridge()

    def _bridge() -> WebBridge:
        return app.state.bridge

    # ── Health ─────────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> Dict[str, Any]:
        br = _bridge()
        return {
            "ok": True,
            "backend_open": br.backend.is_open(),
            "running": br.is_running(),
        }

    # ── State library ──────────────────────────────────────────────────

    @app.get("/states", response_model=StateListResponse)
    def get_states() -> StateListResponse:
        rows = _bridge().list_states()
        return StateListResponse(
            states=[StateEntry(**_coerce_state_row(row)) for row in rows]
        )

    @app.get("/states/{name}", response_model=StateEntry)
    def get_state(name: str) -> StateEntry:
        st = _bridge().get_state(name)
        if st is None:
            raise HTTPException(status_code=404, detail=f"unknown state: {name}")
        return StateEntry(**_coerce_state_row({"name": name, **st}))

    @app.post("/states/{name}", response_model=StateEntry)
    def save_state(name: str, req: SaveStateRequest) -> StateEntry:
        try:
            _bridge().save_state(name, req.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        st = _bridge().get_state(name)
        if st is None:
            raise HTTPException(
                status_code=500, detail=f"failed to persist {name}"
            )
        return StateEntry(**_coerce_state_row({"name": name, **st}))

    @app.delete("/states/{name}")
    def delete_state(name: str) -> Dict[str, Any]:
        ok = _bridge().delete_state(name)
        if not ok:
            raise HTTPException(
                status_code=404, detail=f"unknown state: {name}"
            )
        return {"deleted": True}

    # ── Sequences ──────────────────────────────────────────────────────

    @app.get("/sequences", response_model=SequenceListResponse)
    def list_sequences() -> SequenceListResponse:
        return SequenceListResponse(sequences=_bridge().list_sequences())

    @app.get("/sequences/{name}", response_model=SequenceGetResponse)
    def get_sequence(name: str) -> SequenceGetResponse:
        text = _bridge().get_sequence(name)
        if text is None:
            raise HTTPException(
                status_code=404, detail=f"unknown sequence: {name}"
            )
        return SequenceGetResponse(name=name, text=text)

    @app.post("/sequences/{name}", response_model=SaveSequenceResponse)
    def save_sequence(
        name: str, req: SaveSequenceRequest
    ) -> SaveSequenceResponse:
        ok, errs = _bridge().save_sequence(name, req.text)
        return SaveSequenceResponse(ok=ok, errors=errs)

    @app.delete("/sequences/{name}")
    def delete_sequence(name: str) -> Dict[str, Any]:
        ok = _bridge().delete_sequence(name)
        if not ok:
            raise HTTPException(
                status_code=404, detail=f"unknown sequence: {name}"
            )
        return {"deleted": True}

    # ── DSL validate / run / stop ──────────────────────────────────────

    @app.post("/dsl/validate", response_model=ValidateResponse)
    def validate_dsl(req: ValidateRequest) -> ValidateResponse:
        _program, errs = _bridge().validate_program(req.text)
        return ValidateResponse(ok=not errs, errors=errs)

    @app.post("/dsl/run", response_model=RunResponse)
    def run_dsl(req: RunRequest) -> RunResponse:
        started, errs = _bridge().run_program(req.text)
        return RunResponse(started=started, errors=errs)

    @app.post("/dsl/stop")
    def stop_dsl() -> Dict[str, Any]:
        return {"stopped": _bridge().stop_sequence()}

    @app.get("/dsl/status", response_model=RunStatusResponse)
    def run_status() -> RunStatusResponse:
        st = _bridge().run_status()
        return RunStatusResponse(**st)

    # ── Telemetry ──────────────────────────────────────────────────────

    @app.get("/telemetry", response_model=TelemetryFrame)
    def telemetry_snapshot() -> TelemetryFrame:
        snap = _bridge().snapshot()
        return TelemetryFrame(
            joints=list(snap.get("joints", [])),
            tof=_stringify_tof(snap.get("tof", {})),
            ir=int(snap.get("ir", 0)),
            status=snap.get("status"),
        )

    @app.post("/telemetry/tof")
    def poke_tof(req: TofPokeRequest) -> Dict[str, Any]:
        setter = getattr(_bridge().backend, "set_tof", None)
        if not callable(setter):
            raise HTTPException(
                status_code=400, detail="current backend has no set_tof hook"
            )
        setter(req.channel, req.distance_mm)
        return {"ok": True}

    @app.post("/telemetry/ir")
    def poke_ir(req: IrPokeRequest) -> Dict[str, Any]:
        setter = getattr(_bridge().backend, "set_ir", None)
        if not callable(setter):
            raise HTTPException(
                status_code=400, detail="current backend has no set_ir hook"
            )
        setter(req.severity)
        return {"ok": True}

    # ── WebSocket telemetry stream ────────────────────────────────────

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket) -> None:
        await ws.accept()
        period = 1.0 / _WS_HZ
        try:
            while True:
                snap = _bridge().snapshot()
                frame = {
                    "type": "telemetry",
                    "joints": list(snap.get("joints", [])),
                    "tof": _stringify_tof(snap.get("tof", {})),
                    "ir": int(snap.get("ir", 0)),
                    "status": snap.get("status"),
                }
                await ws.send_text(json.dumps(frame))
                await asyncio.sleep(period)
        except WebSocketDisconnect:
            return

    # Default error handler for ValueError bubbled from the bridge.
    @app.exception_handler(ValueError)
    def _value_error(_req: Any, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app


# ── Helpers ────────────────────────────────────────────────────────────


def _coerce_state_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    StateLibrary returns legacy dicts that may be missing newer fields
    (theta/r/z/wrist_*). Fill them with defaults so the Pydantic model
    validates without the frontend seeing 500s on old data.
    """
    joints = list(row.get("joints", []))
    while len(joints) < 6:
        joints.append(0)
    return {
        "name": row.get("name", ""),
        "joints": [int(j) for j in joints[:6]],
        "theta": float(row.get("theta", 0.0)),
        "r": float(row.get("r", 0.0)),
        "z": float(row.get("z", 0.0)),
        "wrist_offset": float(row.get("wrist_offset", 0.0)),
        "wrist_rot": int(row.get("wrist_rot", 0)),
        "gripper": int(row.get("gripper", joints[5] if len(joints) >= 6 else 73)),
    }


def _stringify_tof(tof: Dict[Any, Any]) -> Dict[str, float]:
    """Pydantic's Dict[str, float] wants string keys — coerce them."""
    return {str(k): float(v) for k, v in tof.items()}


# Module-level default app so ``uvicorn web.backend.app:app`` works
# without an explicit factory call.
app = create_app()


__all__ = ["create_app", "app"]
