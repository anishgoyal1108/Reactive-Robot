"""
models.py — Pydantic schemas exchanged over the Phase 3 REST API.

Kept deliberately narrow: every model corresponds 1:1 to a concrete
REST payload the frontend sends or receives. Fields that only exist
server-side (e.g. interpreter internals) stay in plain dicts in
bridge.py and never leak into the schema.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── States ──────────────────────────────────────────────────────────────


class StateEntry(BaseModel):
    """A single entry in the state library."""

    name: str = Field(..., description="Unique identifier (e.g. HOME_REST)")
    joints: List[int] = Field(
        ..., min_length=6, max_length=6, description="6-DOF joint angles in deg"
    )
    theta: float = 0.0
    r: float = 0.0
    z: float = 0.0
    wrist_offset: float = 0.0
    wrist_rot: int = 0
    gripper: int = 73


class StateListResponse(BaseModel):
    states: List[StateEntry]


class SaveStateRequest(BaseModel):
    joints: List[int] = Field(..., min_length=6, max_length=6)
    theta: float = 0.0
    r: float = 0.0
    z: float = 0.0
    wrist_offset: float = 0.0
    wrist_rot: int = 0
    gripper: int = 73


# ── Sequences ───────────────────────────────────────────────────────────


class SequenceListResponse(BaseModel):
    sequences: List[str]


class SequenceGetResponse(BaseModel):
    name: str
    text: str


class SaveSequenceRequest(BaseModel):
    text: str = Field(..., description="Raw DSL source")


class SaveSequenceResponse(BaseModel):
    ok: bool
    errors: List[str] = Field(default_factory=list)


# ── Validation / run ────────────────────────────────────────────────────


class ValidateRequest(BaseModel):
    text: str


class ValidateResponse(BaseModel):
    ok: bool
    errors: List[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    text: str


class RunResponse(BaseModel):
    started: bool
    errors: List[str] = Field(default_factory=list)


class RunStatusResponse(BaseModel):
    running: bool
    line: int = 0
    kind: str = ""
    depth: int = 0
    pass_current: int = 0
    pass_total: int = 1
    errors: List[str] = Field(default_factory=list)


# ── Telemetry (snapshot / WebSocket payload shape) ─────────────────────


class TelemetryFrame(BaseModel):
    type: str = "telemetry"
    joints: List[int] = Field(default_factory=list)
    tof: Dict[str, float] = Field(default_factory=dict)
    ir: int = 0
    status: Optional[Dict[str, Any]] = None


# ── Sensor pokes (used by the frontend demo harness) ──────────────────


class TofPokeRequest(BaseModel):
    channel: int = Field(..., ge=0, le=3)
    distance_mm: float = Field(..., ge=0.0)


class IrPokeRequest(BaseModel):
    severity: int = Field(..., ge=0, le=3)


# ── Mode switching (Phase 7) ──────────────────────────────────────────


class ModeResponse(BaseModel):
    """
    Current deploy mode reported by ``GET /mode`` and the success
    case of ``POST /mode``.

    ``mode`` is one of the strings from ``bridge.VALID_MODES`` —
    currently "sim" or "hardware". The frontend banner uses the
    string verbatim to decide its styling + copy.
    """

    mode: str = Field(..., description='Active backend mode: "sim" or "hardware".')


class SetModeRequest(BaseModel):
    """Body for ``POST /mode``."""

    mode: str = Field(..., description='Target backend mode: "sim" or "hardware".')


__all__ = [
    "StateEntry",
    "StateListResponse",
    "SaveStateRequest",
    "SequenceListResponse",
    "SequenceGetResponse",
    "SaveSequenceRequest",
    "SaveSequenceResponse",
    "ValidateRequest",
    "ValidateResponse",
    "RunRequest",
    "RunResponse",
    "RunStatusResponse",
    "TelemetryFrame",
    "TofPokeRequest",
    "IrPokeRequest",
    "ModeResponse",
    "SetModeRequest",
]
