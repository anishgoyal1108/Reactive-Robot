"""Session logger geometry-emission regression tests.

Focused on the fields the replay-tab visualisation reads: world_points,
detour_path, last_strategy_entry_tick. Does NOT exercise the sampler
thread — we drive ``_emit_sample`` directly with stubs so the tests
stay fast and deterministic.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pytest

from braccio_ctrl import constants as _c
from braccio_ctrl.session_logger import SessionLogger


class _StubArmState:
    def snapshot(self) -> dict:
        return {
            "joints": [90, 90, 90, 90, 90, 73],
            "theta": 90.0,
            "r": 152.0,
            "z": 35.0,
            "obstacle_response": "clear",
            "obstacle_source": "",
            "obstacle_dist_mm": -1.0,
            "last_cmd": "",
            "last_resp": "",
            "last_error": "",
        }


class _StubTofState:
    def snapshot(self) -> dict:
        return {
            "grids": [],
            "active": [1, 1, 1, 1],
            "hz": [0, 0, 0, 0],
            "tof_thresholds_mm": [100.0, 100.0, 50.0, 50.0],
            "ir_bits": 0,
            "ir_label": "DISABLED",
            "ir_action": "",
        }


class _StubWorld:
    def __init__(self, points: np.ndarray) -> None:
        self._points = np.asarray(points, dtype=np.float64)

    def cloud_points(self) -> np.ndarray:
        return self._points.copy()


class _StubSafety:
    """Just enough of SafetyAPI for the logger to call `.world.cloud_points()`
    and `.snapshot()`."""

    def __init__(
        self,
        points: np.ndarray,
        detour_path: list[list[int]] | None = None,
        last_strategy: str = "simple_direct",
    ) -> None:
        self.world = _StubWorld(points)
        self._detour = list(detour_path or [])
        self._strategy = last_strategy

    def snapshot(self) -> dict:
        return {
            "mode": "sweep",
            "bt_state": "sweep_running",
            "emergency": False,
            "last_strategy": self._strategy,
            "last_failure": None,
            "sweep_direction": +1,
            "sweep_target_deg": 95.0,
            "sweep_detour_path": [list(q) for q in self._detour],
            "sweep_detour_idx": 0,
            "polar_blocked": [],
            "world": {
                "num_points": int(self.world._points.shape[0]),
                "oldest_age_s": 0.0,
                "grid_cells_occ": 0,
                "grid_cells_free": 0,
            },
            "queue_length": 0,
        }


def _drain(logger: SessionLogger) -> list[dict]:
    """Flush + parse every JSON line written by the stubbed logger."""
    logger._file.flush()
    logger._file.seek(0)
    out: list[dict] = []
    for line in logger._file:
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    logger._file.seek(0, io.SEEK_END)
    return out


def _make_logger(
    points: np.ndarray,
    detour: list[list[int]] | None = None,
    strategy: str = "simple_direct",
) -> tuple[SessionLogger, _StubSafety]:
    safety = _StubSafety(points, detour, last_strategy=strategy)
    logger = SessionLogger()
    logger.set_sources(_StubArmState(), _StubTofState(), safety=safety)
    logger._file = io.StringIO()
    logger._t0 = 0.0
    return logger, safety


def test_emit_includes_geometry_when_enabled(monkeypatch):
    """world_points + detour_path appear in each tick when the flag is on."""
    monkeypatch.setattr(_c, "SESSION_LOG_INCLUDE_GEOMETRY", True)
    monkeypatch.setattr(_c, "SESSION_LOG_MAX_WORLD_POINTS", 64)
    pts = np.array([[100.0, 50.0, 25.0], [120.0, 55.0, 30.0]])
    detour = [[91, 90, 90, 90, 90, 73], [93, 90, 90, 90, 90, 73]]
    logger, _ = _make_logger(pts, detour=detour,
                             strategy="sweep_detour:pull_in")
    # Monkeypatch the module-level constant used at emit time too — the
    # logger imported it at module-init, so mutate the binding.
    import braccio_ctrl.session_logger as _sl
    monkeypatch.setattr(_sl, "SESSION_LOG_INCLUDE_GEOMETRY", True)
    monkeypatch.setattr(_sl, "SESSION_LOG_MAX_WORLD_POINTS", 64)
    logger._emit_sample()
    (rec,) = _drain(logger)
    assert "world_points" in rec and len(rec["world_points"]) == 2
    for p in rec["world_points"]:
        assert isinstance(p, list) and len(p) == 3
    assert rec["detour_path"] == detour
    assert rec["last_strategy_entry_tick"] is True


def test_emit_caps_world_points_at_limit(monkeypatch):
    """world_points length must respect SESSION_LOG_MAX_WORLD_POINTS."""
    import braccio_ctrl.session_logger as _sl
    monkeypatch.setattr(_sl, "SESSION_LOG_INCLUDE_GEOMETRY", True)
    monkeypatch.setattr(_sl, "SESSION_LOG_MAX_WORLD_POINTS", 32)
    pts = np.tile(np.array([[10.0, 20.0, 30.0]]), (1000, 1))
    logger, _ = _make_logger(pts)
    logger._emit_sample()
    (rec,) = _drain(logger)
    assert len(rec["world_points"]) == 32


def test_emit_strategy_entry_tick_toggles(monkeypatch):
    """last_strategy_entry_tick is True only on the tick that transitions."""
    import braccio_ctrl.session_logger as _sl
    monkeypatch.setattr(_sl, "SESSION_LOG_INCLUDE_GEOMETRY", True)
    pts = np.empty((0, 3))
    logger, safety = _make_logger(pts, strategy="simple_direct")
    logger._emit_sample()   # first emission → change vs "" → True
    safety._strategy = "simple_direct"
    logger._emit_sample()   # unchanged → False
    safety._strategy = "sweep_detour:pull_in"
    logger._emit_sample()   # transition → True
    records = _drain(logger)
    assert [r["last_strategy_entry_tick"] for r in records] == [
        True, False, True,
    ]


def test_emit_skips_geometry_when_disabled(monkeypatch):
    import braccio_ctrl.session_logger as _sl
    monkeypatch.setattr(_sl, "SESSION_LOG_INCLUDE_GEOMETRY", False)
    pts = np.array([[10.0, 20.0, 30.0]])
    logger, _ = _make_logger(pts, detour=[[91, 90, 90, 90, 90, 73]])
    logger._emit_sample()
    (rec,) = _drain(logger)
    assert rec["world_points"] == []
    assert rec["detour_path"] == []
