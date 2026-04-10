"""
test_dsl_integration.py — End-to-end checks covering DSL × other modules.

These tests import the full controller stack (minus hardware) so that
any accidental regression in the DSL wiring shows up before the
controller is launched against a real arm.
"""

from __future__ import annotations

import os

from braccio_ctrl.dsl import Program, SequenceRunner, parse_auto
from braccio_ctrl.sequence_editor import _static_check
from braccio_ctrl.state_library import StateLibrary


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── Backwards compat: every existing .txt fixture still runs ─────────────


def test_test_input_sequence_file_still_parses_and_type_checks():
    path = os.path.join(_REPO_ROOT, "TEST_INPUT_SEQUENCE.txt")
    with open(path) as f:
        text = f.read()

    program, parse_errs = parse_auto(text)
    assert parse_errs == []
    assert isinstance(program, Program)
    assert program.body  # not empty

    sl = StateLibrary()
    errors = _static_check(program, sl)
    assert errors == [], errors


def test_static_check_rejects_unknown_state_name():
    program, _ = parse_auto("MOVE NOT_A_REAL_STATE WAIT 100")
    sl = StateLibrary()
    errs = _static_check(program, sl)
    assert len(errs) == 1
    assert "NOT_A_REAL_STATE" in errs[0]


def test_static_check_accepts_states_from_library():
    # Pick whatever state is in the committed library — HOME_REST is
    # guaranteed by states.json.
    sl = StateLibrary()
    assert "HOME_REST" in sl.names()
    program, _ = parse_auto("MOVE HOME_REST WAIT 100")
    errs = _static_check(program, sl)
    assert errs == []


# ── Drop-in SequenceRunner still wires to the run_state callback ─────────


def test_sequence_runner_drives_run_state_fn_with_legacy_text():
    """What the curses F5 path does: parse text → SequenceRunner → run."""
    sl = StateLibrary()
    text = "HOME_REST 10\nSTOW_COMPACT 10\n"
    program, parse_errs = parse_auto(text)
    assert parse_errs == []
    static_errs = _static_check(program, sl)
    assert static_errs == []

    calls = []
    runner = SequenceRunner(program, run_state_fn=lambda n: calls.append(n))
    runner.start()
    # Poll briefly for completion.
    import time

    for _ in range(200):
        if not runner.running:
            break
        time.sleep(0.01)
    assert not runner.running
    assert calls == ["HOME_REST", "STOW_COMPACT"]


# ── Extended DSL programs also reach the executor ────────────────────────


def test_control_flow_program_reaches_executor_with_conditions():
    """
    A program that uses TOF + IF/ELSE must still run as long as the
    caller wires a tof_fn hook. The curses editor does not wire one
    today (sensor reads raise NotImplementedError there), but the
    Phase 3 FastAPI server will, so we exercise the positive path
    through the SequenceRunner directly.
    """
    sl = StateLibrary()
    src = """
    IF TOF[0] < 250 {
        MOVE HOME_REST WAIT 1
    } ELSE {
        MOVE STOW_COMPACT WAIT 1
    }
    """
    program, errs = parse_auto(src)
    assert errs == []
    assert _static_check(program, sl) == []

    calls = []
    runner = SequenceRunner(
        program,
        run_state_fn=lambda n: calls.append(n),
        tof_fn=lambda ch: 100.0,  # always "obstacle present"
    )
    runner.start()
    import time

    for _ in range(200):
        if not runner.running:
            break
        time.sleep(0.01)
    assert not runner.running
    assert calls == ["HOME_REST"]


def test_syntax_errors_dont_raise_from_runner_construction():
    """parse_auto must never crash; only return errors."""
    program, errs = parse_auto("WHILE { MOVE")  # malformed
    assert program.body == []
    assert errs
    # Runner construction on empty program is still safe.
    runner = SequenceRunner(program, run_state_fn=lambda n: None)
    assert runner.total_steps >= 1  # min clamp
