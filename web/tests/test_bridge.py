"""
test_bridge.py — WebBridge unit tests.

Covers the session-owner layer end-to-end without going through
FastAPI. TestClient-level tests live in test_app.py; this suite
exercises the bridge's internal contracts (state / sequence / run /
telemetry) directly so that failures here have a narrower blame
surface than a 500 out of an HTTP handler.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from braccio_ctrl.state_library import StateLibrary

from web.backend import WebBackend, WebBridge
from web.backend.bridge import BackendExecutor


def _new_bridge(tmp_path: Path) -> WebBridge:
    sl_path = tmp_path / "states.json"
    sl = StateLibrary(path=str(sl_path))
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
    return WebBridge(
        backend=WebBackend(),
        state_lib=sl,
        sequences_dir=tmp_path / "sequences",
    )


# ── Construction + lifecycle ───────────────────────────────────────────


def test_bridge_opens_backend_on_construction(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        assert br.backend.is_open()
    finally:
        br.close()


def test_bridge_close_closes_backend(tmp_path):
    br = _new_bridge(tmp_path)
    br.close()
    assert not br.backend.is_open()


# ── State library ─────────────────────────────────────────────────────


def test_list_states_returns_all_poses(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        rows = br.list_states()
        names = [r["name"] for r in rows]
        assert "HOME_REST" in names
        assert "STOW_COMPACT" in names
    finally:
        br.close()


def test_save_state_rejects_path_traversal(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        with pytest.raises(ValueError):
            br.save_state(
                "../../evil",
                {"joints": [0, 0, 0, 0, 0, 0]},
            )
    finally:
        br.close()


def test_delete_state_returns_false_when_missing(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        assert not br.delete_state("NOT_A_STATE")
    finally:
        br.close()


# ── Sequences ──────────────────────────────────────────────────────────


def test_save_sequence_rejects_bad_name(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        ok, errs = br.save_sequence("../escape", "HOME_REST 100\n")
        assert not ok
        assert errs
    finally:
        br.close()


def test_save_sequence_rejects_parse_errors(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        ok, errs = br.save_sequence("bad", "HOME_REST xyz\n")
        assert not ok
        assert any("wait" in e.lower() for e in errs)
        # File must NOT have been written.
        assert not (br.sequences_dir / "bad.txt").exists()
    finally:
        br.close()


def test_save_sequence_rejects_unknown_state(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        ok, errs = br.save_sequence("bad", "MOVE NOT_REAL WAIT 10\n")
        assert not ok
        assert any("NOT_REAL" in e for e in errs)
    finally:
        br.close()


def test_save_sequence_writes_and_roundtrips(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        text = "HOME_REST 50\nSTOW_COMPACT 50\n"
        ok, errs = br.save_sequence("test", text)
        assert ok, errs
        assert br.get_sequence("test") == text
        assert "test" in br.list_sequences()
    finally:
        br.close()


def test_delete_sequence(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        br.save_sequence("to_delete", "HOME_REST 1\n")
        assert br.delete_sequence("to_delete")
        assert "to_delete" not in br.list_sequences()
        assert not br.delete_sequence("to_delete")
    finally:
        br.close()


# ── Validation ─────────────────────────────────────────────────────────


def test_validate_program_accepts_good_program(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        prog, errs = br.validate_program("MOVE HOME_REST WAIT 10")
        assert errs == []
        assert prog is not None
    finally:
        br.close()


def test_validate_program_reports_unknown_state(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        _prog, errs = br.validate_program("MOVE NONEXISTENT WAIT 10")
        assert any("NONEXISTENT" in e for e in errs)
    finally:
        br.close()


# ── Running programs ───────────────────────────────────────────────────


def _wait_until_not_running(br: WebBridge, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not br.is_running():
            return
        time.sleep(0.01)
    raise AssertionError("sequence did not finish within timeout")


def test_run_simple_move_program_drives_backend(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        started, errs = br.run_program("MOVE STOW_COMPACT WAIT 10")
        assert started, errs
        _wait_until_not_running(br)
        # STOW_COMPACT's joints should now be reflected on the backend.
        assert br.backend.joint_snapshot() == [90, 30, 170, 90, 90, 73]
    finally:
        br.close()


def test_run_program_while_running_is_rejected(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        # Long-running sequence so we can race against it.
        started, _ = br.run_program("MOVE HOME_REST WAIT 2000")
        assert started
        started2, errs = br.run_program("MOVE HOME_REST WAIT 10")
        assert not started2
        assert errs
        br.stop_sequence()
        _wait_until_not_running(br)
    finally:
        br.close()


def test_stop_sequence_interrupts_wait(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        started, _ = br.run_program("MOVE HOME_REST WAIT 30000")
        assert started
        time.sleep(0.1)
        assert br.stop_sequence()
        _wait_until_not_running(br, timeout_s=3.0)
        assert not br.is_running()
    finally:
        br.close()


def test_run_program_reports_validation_errors(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        started, errs = br.run_program("MOVE BOGUS WAIT 10")
        assert not started
        assert any("BOGUS" in e for e in errs)
    finally:
        br.close()


def test_run_program_supports_if_with_tof_poke(tmp_path):
    """
    IF TOF[0] < 250 should pick the true branch when the WebBackend's
    synthetic ToF has been poked below that threshold.
    """
    br = _new_bridge(tmp_path)
    try:
        assert hasattr(br.backend, "set_tof")
        br.backend.set_tof(0, 100.0)  # type: ignore[attr-defined]
        src = """
        IF TOF[0] < 250 {
            MOVE STOW_COMPACT WAIT 10
        } ELSE {
            MOVE HOME_REST WAIT 10
        }
        """
        started, errs = br.run_program(src)
        assert started, errs
        _wait_until_not_running(br)
        # True branch → STOW_COMPACT joints.
        assert br.backend.joint_snapshot() == [90, 30, 170, 90, 90, 73]
    finally:
        br.close()


def test_run_status_tracks_line_and_kind(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        started, _ = br.run_program("MOVE HOME_REST WAIT 20")
        assert started
        _wait_until_not_running(br)
        status = br.run_status()
        assert status["running"] is False
        # After a clean run, the last-executed line was MOVE.
        assert status["kind"] == "MoveStmt"
    finally:
        br.close()


# ── Telemetry ──────────────────────────────────────────────────────────


def test_snapshot_returns_joints_and_sensors(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        br.backend.set_tof(0, 200.0)  # type: ignore[attr-defined]
        br.backend.set_ir(1)  # type: ignore[attr-defined]
        snap = br.snapshot()
        assert snap["type"] == "telemetry"
        assert snap["joints"] == [90, 90, 90, 90, 90, 73]
        assert snap["tof"][0] == 200.0
        assert snap["ir"] == 1
    finally:
        br.close()


def test_subscribers_receive_status_broadcasts(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        received = []
        lock = threading.Lock()

        def on_msg(frame):
            with lock:
                received.append(frame)

        br.add_subscriber(on_msg)
        started, _ = br.run_program("MOVE HOME_REST WAIT 5")
        assert started
        _wait_until_not_running(br)
        # Status events are broadcast by _on_status; at least one MoveStmt
        # event should have reached the subscriber.
        with lock:
            kinds = [f.get("kind") for f in received if f.get("type") == "status"]
        assert "MoveStmt" in kinds
    finally:
        br.close()


# ── BackendExecutor ────────────────────────────────────────────────────


def test_backend_executor_move_to_state_emits_set_all(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        stop = threading.Event()
        exe = BackendExecutor(br.backend, br.state_library, stop)
        exe.move_to_state("STOW_COMPACT")
        assert br.backend.joint_snapshot() == [90, 30, 170, 90, 90, 73]
    finally:
        br.close()


def test_backend_executor_unknown_state_raises(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        exe = BackendExecutor(br.backend, br.state_library, threading.Event())
        with pytest.raises(ValueError):
            exe.move_to_state("NOPE")
    finally:
        br.close()


def test_backend_executor_read_joint_reflects_backend(tmp_path):
    br = _new_bridge(tmp_path)
    try:
        exe = BackendExecutor(br.backend, br.state_library, threading.Event())
        exe.set_joint("B", 45)
        assert exe.read_joint("B") == 45
    finally:
        br.close()
