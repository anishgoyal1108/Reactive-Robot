"""
test_dsl_interpreter.py — Interpreter semantics for braccio_ctrl.dsl.

Uses a tiny recording executor so every test can assert on the exact
sequence of primitive calls the interpreter emitted without touching
hardware or sim.
"""

from __future__ import annotations

import threading
import time

import pytest

from braccio_ctrl.dsl import (
    BaseExecutor,
    Interpreter,
    Program,
    SequenceRunner,
    parse,
    parse_auto,
)


# ── Recording executor fixture ───────────────────────────────────────────


class RecordingExecutor(BaseExecutor):
    """In-memory stand-in that records every call and answers sensor reads
    from preset dicts. wait_ms becomes a no-op so tests run instantly."""

    def __init__(
        self,
        *,
        tof=None,
        ir=0,
        joints=None,
        stopped=False,
    ):
        self.events: list = []
        self._tof = tof or {0: 500.0, 1: 500.0, 2: 500.0, 3: 500.0}
        self._ir = ir
        self._joints = joints or {"B": 90, "S": 90, "E": 90, "WV": 90, "WR": 90, "G": 73}
        self._stopped = stopped

    def move_to_state(self, name):
        self.events.append(("MOVE", name))

    def set_joint(self, tok, deg):
        self.events.append(("SET", tok, deg))

    def go_home(self):
        self.events.append(("HOME",))

    def wait_ms(self, ms):
        self.events.append(("WAIT", ms))

    def read_tof(self, ch):
        return self._tof.get(ch, 9999.0)

    def read_ir(self):
        return self._ir

    def read_joint(self, tok):
        return self._joints.get(tok, 0)

    def define_state(self, name, joints):
        self.events.append(("DEFINE_STATE", name, tuple(joints)))

    def define_obstacle(self, name, pos, radius, shape):
        self.events.append(("DEFINE_OBS", name, tuple(pos), radius, shape))

    def is_stopped(self):
        return self._stopped


# ── Basic statement execution ────────────────────────────────────────────


def test_move_calls_executor():
    exe = RecordingExecutor()
    Interpreter(exe).run(parse("MOVE HOME_REST"))
    assert exe.events == [("MOVE", "HOME_REST")]


def test_move_with_wait():
    exe = RecordingExecutor()
    Interpreter(exe).run(parse("MOVE HOME_REST WAIT 250"))
    assert exe.events == [("MOVE", "HOME_REST"), ("WAIT", 250)]


def test_set_joint_calls_executor():
    exe = RecordingExecutor()
    Interpreter(exe).run(parse("SET B 45"))
    assert exe.events == [("SET", "B", 45)]


def test_wait_stmt():
    exe = RecordingExecutor()
    Interpreter(exe).run(parse("WAIT 200"))
    assert exe.events == [("WAIT", 200)]


def test_home_stmt_calls_go_home():
    exe = RecordingExecutor()
    Interpreter(exe).run(parse("HOME"))
    assert exe.events == [("HOME",)]


def test_define_state_calls_executor():
    exe = RecordingExecutor()
    Interpreter(exe).run(parse("STATE POSE1 { JOINTS 10 20 30 40 50 60 }"))
    assert exe.events == [("DEFINE_STATE", "POSE1", (10, 20, 30, 40, 50, 60))]


def test_define_obstacle_calls_executor():
    exe = RecordingExecutor()
    Interpreter(exe).run(
        parse("OBSTACLE O1 { POS 1 2 3 RADIUS 5 SHAPE SPHERE }")
    )
    assert exe.events == [("DEFINE_OBS", "O1", (1.0, 2.0, 3.0), 5.0, "SPHERE")]


# ── Control flow ────────────────────────────────────────────────────────


def test_repeat_flat_runs_count_times():
    exe = RecordingExecutor()
    Interpreter(exe).run(parse("REPEAT 4 { MOVE X }"))
    assert exe.events == [("MOVE", "X")] * 4


def test_nested_repeat():
    exe = RecordingExecutor()
    Interpreter(exe).run(parse("REPEAT 3 { REPEAT 2 { MOVE A } }"))
    assert exe.events == [("MOVE", "A")] * 6


def test_if_true_branch_runs_only_then():
    exe = RecordingExecutor(tof={0: 100.0})
    Interpreter(exe).run(
        parse("IF TOF[0] < 250 { MOVE STOW } ELSE { MOVE LIFT }")
    )
    assert exe.events == [("MOVE", "STOW")]


def test_if_false_branch_runs_only_else():
    exe = RecordingExecutor(tof={0: 500.0})
    Interpreter(exe).run(
        parse("IF TOF[0] < 250 { MOVE STOW } ELSE { MOVE LIFT }")
    )
    assert exe.events == [("MOVE", "LIFT")]


def test_if_no_else_branch_is_noop():
    exe = RecordingExecutor(ir=0)
    Interpreter(exe).run(parse("IF IR > 0 { MOVE DANGER }"))
    assert exe.events == []


def test_while_runs_until_condition_false():
    # IR reading drops by 1 every move — make a counter executor.
    class CountingExecutor(RecordingExecutor):
        def move_to_state(self, name):
            super().move_to_state(name)
            self._ir = max(0, self._ir - 1)

    exe = CountingExecutor(ir=3)
    Interpreter(exe).run(parse("WHILE IR > 0 { MOVE STEP }"))
    assert exe.events == [("MOVE", "STEP")] * 3


def test_while_respects_safety_cap():
    # Sensor never changes → interpreter should error out after _MAX iters.
    exe = RecordingExecutor(ir=5)
    interp = Interpreter(exe)
    interp.run(parse("WHILE IR > 0 { WAIT 0 }"))
    assert any("WHILE" in e for e in interp.errors)


# ── Sensor evaluation ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "op,threshold,reading,expect_then",
    [
        ("<", 250, 100, True),
        ("<", 250, 300, False),
        (">", 100, 150, True),
        (">", 100, 50, False),
        ("<=", 200, 200, True),
        (">=", 200, 200, True),
        ("==", 100, 100, True),
        ("!=", 100, 100, False),
    ],
)
def test_comparison_ops_evaluate_correctly(op, threshold, reading, expect_then):
    exe = RecordingExecutor(tof={0: reading})
    src = f"IF TOF[0] {op} {threshold} {{ MOVE T }} ELSE {{ MOVE F }}"
    Interpreter(exe).run(parse(src))
    assert exe.events == [("MOVE", "T" if expect_then else "F")]


def test_joint_read_expr():
    exe = RecordingExecutor(joints={"B": 120})
    Interpreter(exe).run(parse("IF JOINT[B] > 90 { MOVE TURNED }"))
    assert exe.events == [("MOVE", "TURNED")]


def test_ir_expr():
    exe = RecordingExecutor(ir=3)
    Interpreter(exe).run(parse("IF IR >= 2 { HOME }"))
    assert exe.events == [("HOME",)]


# ── Stop semantics ──────────────────────────────────────────────────────


def test_stop_between_statements_halts_run():
    class StoppingExecutor(RecordingExecutor):
        def __init__(self):
            super().__init__()
            self._calls = 0

        def move_to_state(self, name):
            super().move_to_state(name)
            self._calls += 1
            if self._calls >= 2:
                self._stopped = True

    exe = StoppingExecutor()
    Interpreter(exe).run(parse("REPEAT 10 { MOVE X }"))
    # Exactly 2 moves before is_stopped() returned True at the start of
    # iteration 3.
    assert exe.events == [("MOVE", "X"), ("MOVE", "X")]


def test_interpreter_stop_event_flag():
    """stopped reflects the event state after stop() is called.

    Note: Interpreter.run() intentionally clears the stop flag at the
    start so a single Interpreter instance can be reused across runs.
    To halt a run in progress, stop() must be called from another
    thread or from within an executor callback — see
    test_stop_between_statements_halts_run for the in-flight case.
    """
    exe = RecordingExecutor()
    interp = Interpreter(exe)
    assert not interp.stopped
    interp.stop()
    assert interp.stopped


# ── SequenceRunner shim (used by the curses editor) ────────────────────


def test_sequence_runner_runs_legacy_fixture():
    """The curses editor's F5 path must still work for a legacy file."""
    text = "HOME_REST 50\nSTOW_COMPACT 50\nREPEAT 2\n"
    program, errs = parse_auto(text)
    assert errs == []

    events = []
    runner = SequenceRunner(program, run_state_fn=lambda n: events.append(n))
    runner.start()
    # Wait up to 3 s for the background thread to finish the (fast) run.
    for _ in range(300):
        if not runner.running:
            break
        time.sleep(0.01)
    assert not runner.running
    assert events == ["HOME_REST", "STOW_COMPACT"] * 2


def test_sequence_runner_stop_is_responsive():
    """SequenceRunner.stop() must interrupt a long-running wait."""
    # 30 s wait — the test must be much faster than that.
    program, _ = parse_auto("HOME_REST 30000\n")
    events = []
    runner = SequenceRunner(program, run_state_fn=lambda n: events.append(n))
    runner.start()
    # Give it enough time to enter the first wait.
    time.sleep(0.1)
    runner.stop()
    for _ in range(200):
        if not runner.running:
            break
        time.sleep(0.01)
    assert not runner.running
    assert events == ["HOME_REST"]


def test_sequence_runner_reports_total_steps():
    program, _ = parse_auto(
        "HOME_REST 10\nSTOW_COMPACT 10\nLIFT_HIGH_CARRY 10\nREPEAT 5\n"
    )
    runner = SequenceRunner(program, run_state_fn=lambda n: None)
    # 3 moves * no visiting needed; count inside RepeatBlock is 3.
    assert runner.total_steps >= 3
    assert runner.total_passes == 5


# ── Status callback ─────────────────────────────────────────────────────


def test_status_callback_fires_once_per_statement():
    updates = []

    def on_status(info):
        updates.append((info["line"], info["kind"]))

    exe = RecordingExecutor()
    Interpreter(exe, status_cb=on_status).run(
        parse("MOVE A\nSET B 45\nHOME")
    )
    kinds = [k for _, k in updates]
    assert kinds == ["MoveStmt", "SetJointStmt", "HomeStmt"]


# ── Error handling ──────────────────────────────────────────────────────


def test_executor_exception_is_wrapped_with_line_info():
    class BrokenExecutor(RecordingExecutor):
        def move_to_state(self, name):
            raise RuntimeError("serial port closed")

    exe = BrokenExecutor()
    interp = Interpreter(exe)
    interp.run(parse("\nMOVE X"))
    assert len(interp.errors) == 1
    assert "L2" in interp.errors[0]
    assert "serial port closed" in interp.errors[0]
