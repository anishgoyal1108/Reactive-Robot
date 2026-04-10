"""
conftest.py — Shared fixtures for the Phase 3 web backend tests.

Every test that needs a bridge or an app uses these fixtures so that
each test run gets:

  * A fresh in-memory ``WebBackend`` (no leaked joint state)
  * A fresh ``StateLibrary`` pointing at a throwaway JSON file
    (populated with a handful of poses covering the main branches
    the interpreter exercises — HOME_REST + a second pose for IF/ELSE
    tests)
  * A fresh sequences directory under ``tmp_path`` so saves made by
    one test never leak into another
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import pytest

# Make braccio_main_runner importable without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRACCIO_ROOT = _REPO_ROOT / "braccio_main_runner"
if str(_BRACCIO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BRACCIO_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from braccio_ctrl.state_library import StateLibrary  # noqa: E402

from web.backend import WebBackend, WebBridge, create_app  # noqa: E402


@pytest.fixture
def fresh_state_lib(tmp_path: Path) -> StateLibrary:
    """A clean StateLibrary backed by a throwaway JSON file."""
    path = tmp_path / "states.json"
    sl = StateLibrary(path=str(path))
    sl.save_state(
        "HOME_REST",
        {
            "joints": [90, 90, 90, 90, 90, 73],
            "theta": 0.0,
            "r": 0.0,
            "z": 0.0,
            "wrist_offset": 0.0,
            "wrist_rot": 0,
            "gripper": 73,
        },
    )
    sl.save_state(
        "STOW_COMPACT",
        {
            "joints": [90, 30, 170, 90, 90, 73],
            "theta": 0.0,
            "r": 0.0,
            "z": 0.0,
            "wrist_offset": 0.0,
            "wrist_rot": 0,
            "gripper": 73,
        },
    )
    return sl


@pytest.fixture
def bridge(tmp_path: Path, fresh_state_lib: StateLibrary) -> Iterator[WebBridge]:
    """An isolated WebBridge: in-memory backend + fresh sequences dir."""
    seq_dir = tmp_path / "sequences"
    br = WebBridge(
        backend=WebBackend(),
        state_lib=fresh_state_lib,
        sequences_dir=seq_dir,
    )
    yield br
    br.close()


@pytest.fixture
def client(bridge: WebBridge):
    """A FastAPI TestClient bound to a fresh bridge."""
    from fastapi.testclient import TestClient

    app = create_app(bridge=bridge)
    with TestClient(app) as c:
        yield c
