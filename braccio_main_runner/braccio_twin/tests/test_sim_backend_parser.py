"""
Pybullet-free tests for SimBackend's command parser.

These exercise the plain-Python ASCII parser that translates
`SET ALL`, `SET DELTA`, `HOME`, `GET POS`, and `PING` into the same
response dicts the real Arduino would produce. They run without a
PyBullet install — useful as a CI smoke check for the Phase 0/1 seam.

Run with:
    python -m braccio_twin.tests.test_sim_backend_parser
"""

from __future__ import annotations

import queue

from braccio_twin.sim_backend import SimBackend


def _drain(sim: SimBackend) -> list[dict]:
    out: list[dict] = []
    while True:
        try:
            out.append(sim.response_queue.get_nowait())
        except queue.Empty:
            break
    return out


def test_ping_returns_pong() -> None:
    sim = SimBackend(gui=False)
    sim.send_cmd("PING\n")
    responses = _drain(sim)
    assert any(r["type"] == "pong" for r in responses), responses


def test_set_all_echoes_positions() -> None:
    sim = SimBackend(gui=False)
    sim.send_cmd("SET ALL 10 20 30 40 50 60\n")
    responses = _drain(sim)
    ok = [r for r in responses if r["type"] == "ok_all"]
    assert ok, responses
    assert ok[0]["positions"] == [10, 20, 30, 40, 50, 60]


def test_set_joint_updates_single_target() -> None:
    sim = SimBackend(gui=False)
    sim.send_cmd("SET B 45\n")
    sim.send_cmd("GET POS\n")
    responses = _drain(sim)
    ok_joint = [r for r in responses if r["type"] == "ok_joint"]
    pos = [r for r in responses if r["type"] == "pos"]
    assert ok_joint and ok_joint[0]["joint"] == 0 and ok_joint[0]["deg"] == 45
    assert pos and pos[0]["positions"][0] == 45


def test_home_resets_targets() -> None:
    sim = SimBackend(gui=False)
    sim.send_cmd("SET ALL 0 20 20 20 20 20\n")
    sim.send_cmd("HOME\n")
    sim.send_cmd("GET POS\n")
    responses = _drain(sim)
    assert any(r["type"] == "ok_home" for r in responses), responses
    pos = [r for r in responses if r["type"] == "pos"]
    assert pos and pos[0]["positions"] == [90, 90, 90, 90, 90, 73]


def test_set_delta_parsed() -> None:
    sim = SimBackend(gui=False)
    sim.send_cmd("SET DELTA 3\n")
    responses = _drain(sim)
    assert any(r["type"] == "ok_delta" and r["delta"] == 3 for r in responses)


def test_unknown_command_returns_error() -> None:
    sim = SimBackend(gui=False)
    sim.send_cmd("GARBAGE COMMAND\n")
    responses = _drain(sim)
    err = [r for r in responses if r["type"] == "error"]
    assert err, responses


def main() -> int:
    tests = [
        test_ping_returns_pong,
        test_set_all_echoes_positions,
        test_set_joint_updates_single_target,
        test_home_resets_targets,
        test_set_delta_parsed,
        test_unknown_command_returns_error,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  ok: {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL: {t.__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
