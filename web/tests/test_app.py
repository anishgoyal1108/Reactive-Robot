"""
test_app.py — FastAPI-level integration tests.

Exercises every REST route and the WebSocket telemetry stream via
fastapi.testclient.TestClient. Each test uses the fixtures from
conftest.py so it gets an isolated WebBridge + in-memory backend.
"""

from __future__ import annotations

import json
import time


# ── Health ─────────────────────────────────────────────────────────────


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["backend_open"] is True
    assert body["running"] is False


# ── URDF ───────────────────────────────────────────────────────────────


def test_urdf_endpoint_serves_braccio_xml(client):
    resp = client.get("/urdf")
    assert resp.status_code == 200
    # application/xml, application/xml; charset=utf-8, etc. all ok.
    assert "xml" in resp.headers["content-type"]
    body = resp.text
    # Structural sanity: the 6 real joints must all be present.
    assert '<robot name="braccio_v2">' in body
    assert 'name="joint_base"' in body
    assert 'name="joint_shoulder"' in body
    assert 'name="joint_elbow"' in body
    assert 'name="joint_wrist_vert"' in body
    assert 'name="joint_wrist_rot"' in body
    assert 'name="joint_gripper"' in body


# ── State library ──────────────────────────────────────────────────────


def test_list_states(client):
    resp = client.get("/states")
    assert resp.status_code == 200
    body = resp.json()
    names = [s["name"] for s in body["states"]]
    assert "HOME_REST" in names
    assert "STOW_COMPACT" in names


def test_get_single_state(client):
    resp = client.get("/states/HOME_REST")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "HOME_REST"
    assert body["joints"] == [90, 90, 90, 90, 90, 73]


def test_get_unknown_state_404(client):
    resp = client.get("/states/NONEXISTENT")
    assert resp.status_code == 404


def test_save_and_retrieve_state(client):
    payload = {
        "joints": [45, 90, 120, 90, 90, 50],
        "theta": 10.0,
        "r": 150.0,
        "z": 20.0,
        "wrist_offset": 0.0,
        "wrist_rot": 90,
        "gripper": 50,
    }
    resp = client.post("/states/MY_POSE", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "MY_POSE"
    assert body["joints"] == [45, 90, 120, 90, 90, 50]

    # Round-trip
    resp2 = client.get("/states/MY_POSE")
    assert resp2.status_code == 200
    assert resp2.json()["joints"] == [45, 90, 120, 90, 90, 50]


def test_delete_state(client):
    client.post(
        "/states/TEMP",
        json={"joints": [0, 30, 60, 90, 120, 50]},
    )
    resp = client.delete("/states/TEMP")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    # Second delete returns 404.
    assert client.delete("/states/TEMP").status_code == 404


# ── Sequences ──────────────────────────────────────────────────────────


def test_list_sequences_empty_initially(client):
    resp = client.get("/sequences")
    assert resp.status_code == 200
    assert resp.json() == {"sequences": []}


def test_save_and_list_sequence(client):
    text = "HOME_REST 100\nSTOW_COMPACT 100\n"
    resp = client.post("/sequences/demo", json={"text": text})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["errors"] == []

    lst = client.get("/sequences").json()
    assert "demo" in lst["sequences"]


def test_get_sequence_roundtrip(client):
    text = "HOME_REST 50\n"
    client.post("/sequences/roundtrip", json={"text": text})
    resp = client.get("/sequences/roundtrip")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "roundtrip"
    assert body["text"] == text


def test_save_sequence_rejects_parse_errors(client):
    resp = client.post(
        "/sequences/bad", json={"text": "HOME_REST not_a_number\n"}
    )
    assert resp.status_code == 200  # endpoint returns ok=false, not 400
    body = resp.json()
    assert body["ok"] is False
    assert body["errors"]

    # The file must not be listed.
    assert "bad" not in client.get("/sequences").json()["sequences"]


def test_save_sequence_rejects_unknown_state(client):
    resp = client.post(
        "/sequences/bad_state", json={"text": "MOVE NOT_A_STATE WAIT 10\n"}
    )
    body = resp.json()
    assert body["ok"] is False
    assert any("NOT_A_STATE" in e for e in body["errors"])


def test_delete_sequence(client):
    client.post("/sequences/tmp", json={"text": "HOME_REST 1\n"})
    resp = client.delete("/sequences/tmp")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
    assert client.delete("/sequences/tmp").status_code == 404


# ── DSL validate / run / stop / status ────────────────────────────────


def test_validate_accepts_good_dsl(client):
    resp = client.post(
        "/dsl/validate", json={"text": "MOVE HOME_REST WAIT 10"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["errors"] == []


def test_validate_reports_unknown_state(client):
    resp = client.post(
        "/dsl/validate", json={"text": "MOVE GHOST WAIT 10"}
    )
    body = resp.json()
    assert body["ok"] is False
    assert any("GHOST" in e for e in body["errors"])


def test_run_moves_backend_then_finishes(client):
    resp = client.post(
        "/dsl/run", json={"text": "MOVE STOW_COMPACT WAIT 10"}
    )
    assert resp.status_code == 200
    assert resp.json()["started"] is True

    # Poll status until the run finishes.
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not client.get("/dsl/status").json()["running"]:
            break
        time.sleep(0.02)
    else:
        raise AssertionError("run did not finish")

    # Backend joint state matches STOW_COMPACT.
    snap = client.get("/telemetry").json()
    assert snap["joints"] == [90, 30, 170, 90, 90, 73]


def test_stop_sequence_from_rest(client):
    client.post("/dsl/run", json={"text": "MOVE HOME_REST WAIT 20000"})
    time.sleep(0.1)
    resp = client.post("/dsl/stop")
    assert resp.status_code == 200
    assert resp.json()["stopped"] is True

    # Status should report running=False shortly after.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not client.get("/dsl/status").json()["running"]:
            break
        time.sleep(0.02)
    assert not client.get("/dsl/status").json()["running"]


def test_run_while_running_is_rejected(client):
    client.post("/dsl/run", json={"text": "MOVE HOME_REST WAIT 2000"})
    time.sleep(0.05)
    resp = client.post("/dsl/run", json={"text": "MOVE HOME_REST WAIT 10"})
    body = resp.json()
    assert body["started"] is False
    assert body["errors"]
    client.post("/dsl/stop")


# ── Telemetry REST ────────────────────────────────────────────────────


def test_telemetry_snapshot(client):
    resp = client.get("/telemetry")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "telemetry"
    assert body["joints"] == [90, 90, 90, 90, 90, 73]
    assert "tof" in body
    assert body["ir"] == 0


def test_telemetry_tof_poke_reflected_in_snapshot(client):
    resp = client.post(
        "/telemetry/tof", json={"channel": 0, "distance_mm": 150.0}
    )
    assert resp.status_code == 200
    snap = client.get("/telemetry").json()
    assert snap["tof"]["0"] == 150.0


def test_telemetry_ir_poke_reflected_in_snapshot(client):
    resp = client.post("/telemetry/ir", json={"severity": 2})
    assert resp.status_code == 200
    snap = client.get("/telemetry").json()
    assert snap["ir"] == 2


def test_telemetry_tof_poke_rejects_bad_channel(client):
    resp = client.post(
        "/telemetry/tof", json={"channel": 9, "distance_mm": 150.0}
    )
    # Pydantic validation → 422, not 400.
    assert resp.status_code == 422


# ── WebSocket ─────────────────────────────────────────────────────────


def test_websocket_streams_telemetry(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        raw = ws.receive_text()
        frame = json.loads(raw)
        assert frame["type"] == "telemetry"
        assert frame["joints"] == [90, 90, 90, 90, 90, 73]
        assert "tof" in frame
        assert "ir" in frame


def test_websocket_reflects_live_joint_changes(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        ws.receive_text()  # initial frame
        # Run a sequence that moves to STOW_COMPACT.
        resp = client.post(
            "/dsl/run", json={"text": "MOVE STOW_COMPACT WAIT 5"}
        )
        assert resp.json()["started"] is True

        # Poll frames until we see the new joint values.
        deadline = time.monotonic() + 3.0
        seen_stow = False
        while time.monotonic() < deadline:
            raw = ws.receive_text()
            frame = json.loads(raw)
            if frame.get("joints") == [90, 30, 170, 90, 90, 73]:
                seen_stow = True
                break
        assert seen_stow, "never observed STOW_COMPACT joints on the stream"
