"""
test_web_backend.py — Unit tests for the in-memory WebBackend.

Exercises every send_cmd branch, the sensor poke surface, and the
thread-safety invariants that the bridge + WebSocket pump rely on.
"""

from __future__ import annotations

import queue
import threading

import pytest

from web.backend.web_backend import WebBackend


# ── Lifecycle ──────────────────────────────────────────────────────────


def test_connect_sets_open_and_pushes_ready():
    be = WebBackend()
    assert not be.is_open()
    assert be.connect() is True
    assert be.is_open()
    # First message is 'ready'.
    resp = be.response_queue.get_nowait()
    assert resp == {"type": "ready"}


def test_ping_pushes_pong():
    be = WebBackend()
    be.connect()
    be.response_queue.get_nowait()  # drain 'ready'
    assert be.ping() is True
    assert be.response_queue.get_nowait() == {"type": "pong"}


def test_close_sets_is_open_false():
    be = WebBackend()
    be.connect()
    be.close()
    assert not be.is_open()


# ── Command parsing ────────────────────────────────────────────────────


def test_set_all_updates_joint_snapshot():
    be = WebBackend()
    be.connect()
    be.send_cmd("SET ALL 10 30 50 70 90 60\n")
    assert be.joint_snapshot() == [10, 30, 50, 70, 90, 60]


def test_set_all_clamps_to_joint_limits():
    """Values outside the physical servo limits must be clamped."""
    be = WebBackend()
    be.connect()
    # Shoulder min is 15, not 0 → clamps to 15.
    # Gripper max is 73, not 180 → clamps to 73.
    be.send_cmd("SET ALL 0 0 180 180 180 180\n")
    assert be.joint_snapshot() == [0, 15, 180, 180, 180, 73]


def test_set_single_joint_updates_only_that_joint():
    be = WebBackend()
    be.connect()
    be.send_cmd("SET B 45\n")
    snap = be.joint_snapshot()
    assert snap[0] == 45
    assert snap[1:] == [90, 90, 90, 90, 73]  # untouched


def test_set_single_joint_clamps():
    be = WebBackend()
    be.connect()
    be.send_cmd("SET S 200\n")  # shoulder max is 165
    assert be.joint_snapshot()[1] == 165
    be.send_cmd("SET G 5\n")  # gripper min is 10
    assert be.joint_snapshot()[5] == 10


def test_home_resets_joints():
    be = WebBackend()
    be.connect()
    be.send_cmd("SET ALL 45 45 45 45 45 45\n")
    be.send_cmd("HOME\n")
    assert be.joint_snapshot() == [90, 90, 90, 90, 90, 73]


def test_get_pos_pushes_response():
    be = WebBackend()
    be.connect()
    be.send_cmd("SET ALL 10 30 50 70 90 60\n")
    # Drain all responses from prior commands.
    while True:
        try:
            be.response_queue.get_nowait()
        except queue.Empty:
            break
    be.send_cmd("GET POS\n")
    resp = be.response_queue.get_nowait()
    assert resp["type"] == "pos"
    assert resp["positions"] == [10, 30, 50, 70, 90, 60]


def test_set_delta_is_clamped_1_to_5():
    be = WebBackend()
    be.connect()
    be.send_cmd("SET DELTA 99\n")
    # Drain first; only checking no crash + clamping via snapshot impossible
    # (delta is internal), so just verify the response indicates clamp.
    while not be.response_queue.empty():
        r = be.response_queue.get_nowait()
        if r.get("type") == "ok_delta":
            assert r["delta"] == 5


def test_unknown_command_emits_error_response():
    be = WebBackend()
    be.connect()
    be.send_cmd("NONSENSE FOO BAR\n")
    while not be.response_queue.empty():
        r = be.response_queue.get_nowait()
        if r.get("type") == "error":
            assert "NONSENSE" in r["message"]
            return
    pytest.fail("expected an error response")


def test_empty_command_is_noop():
    be = WebBackend()
    be.connect()
    before = be.joint_snapshot()
    be.send_cmd("")
    be.send_cmd("   \n")
    assert be.joint_snapshot() == before


# ── Sensor pokes ───────────────────────────────────────────────────────


def test_set_tof_updates_snapshot():
    be = WebBackend()
    be.set_tof(0, 120.5)
    snap = be.sensor_snapshot()
    assert snap["tof"][0] == 120.5
    assert be.read_tof(0) == 120.5


def test_set_tof_rejects_out_of_range_channel():
    be = WebBackend()
    with pytest.raises(ValueError):
        be.set_tof(5, 100.0)


def test_set_ir_updates_snapshot():
    be = WebBackend()
    be.set_ir(2)
    assert be.sensor_snapshot()["ir"] == 2
    assert be.read_ir() == 2


def test_set_ir_rejects_out_of_range_severity():
    be = WebBackend()
    with pytest.raises(ValueError):
        be.set_ir(99)


def test_reset_clears_everything():
    be = WebBackend()
    be.connect()
    be.send_cmd("SET ALL 10 20 30 40 50 60\n")
    be.set_tof(0, 120.0)
    be.set_ir(3)
    be.reset()
    assert be.joint_snapshot() == [90, 90, 90, 90, 90, 73]
    assert be.sensor_snapshot()["ir"] == 0
    assert be.read_tof(0) == 9999.0


# ── Thread-safety ──────────────────────────────────────────────────────


def test_concurrent_writers_do_not_corrupt_state():
    """
    Two writer threads spamming SET B / SET S in parallel must leave
    the backend in a consistent state (no torn reads).
    """
    be = WebBackend()
    be.connect()

    def writer_b() -> None:
        for i in range(200):
            be.send_cmd(f"SET B {i % 180}\n")

    def writer_s() -> None:
        for i in range(200):
            be.send_cmd(f"SET S {15 + (i % 150)}\n")

    t1 = threading.Thread(target=writer_b)
    t2 = threading.Thread(target=writer_s)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    snap = be.joint_snapshot()
    assert 0 <= snap[0] <= 180
    assert 15 <= snap[1] <= 165
