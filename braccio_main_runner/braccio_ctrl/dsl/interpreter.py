"""
interpreter.py — Walk a Program AST and drive the arm through an Executor.

The interpreter is deliberately side-effect-free on its own. All
hardware (or sim) interaction goes through the ``Executor`` Protocol
so that the same program can run against:

  * ``SequenceEditorExecutor`` — the curses editor's existing
    ``run_state_fn`` callback (backwards compatible)
  * ``BackendExecutor``       — a ``BraccioBackend`` (real arm or sim)
    for the Phase 3 FastAPI server
  * a mock executor used by the test suite

Status callbacks
----------------
The interpreter calls ``status_cb(info)`` every time it begins
executing a statement. ``info`` is a small dict carrying:

    {"line": int,           # 1-based source line of the statement
     "depth": int,          # nesting depth (0 = top level)
     "kind":  str,          # class name, e.g. "MoveStmt"
     "pass":  (int, int),   # current REPEAT pass / total (outermost only)
     "total_statements": int}

The curses editor uses this to show a green "● RUNNING line N" bar;
the Phase 3 WebSocket bridge can forward it to the browser so the
Blockly workspace highlights the currently-executing block.

Stopping
--------
The interpreter polls ``executor.is_stopped()`` between every statement
and inside ``executor.wait()``. Calling ``Interpreter.stop()`` sets an
internal ``threading.Event`` that executor wait loops must honour via
their own stop predicate — see ``SequenceRunner`` below for the
reference implementation.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Optional, Protocol

from .ast_nodes import (
    Condition,
    DefineObstacle,
    DefineState,
    HomeStmt,
    IfBlock,
    IrExpr,
    JointReadExpr,
    MoveStmt,
    Program,
    RepeatBlock,
    SetJointStmt,
    Statement,
    TofExpr,
    WaitStmt,
    WhileBlock,
)


class Executor(Protocol):
    """
    Everything the interpreter needs to affect the world.

    An implementation holds the backend, state library, ToF snapshot,
    etc. and exposes just these primitives. The interpreter never
    touches BraccioBackend / ToFState / StateLibrary directly, keeping
    the DSL runtime pure.
    """

    def move_to_state(self, name: str) -> None: ...
    def set_joint(self, joint_token: str, deg: int) -> None: ...
    def go_home(self) -> None: ...
    def wait_ms(self, ms: int) -> None: ...
    def read_tof(self, channel: int) -> float: ...
    def read_ir(self) -> int: ...
    def read_joint(self, joint_token: str) -> int: ...
    def define_state(self, name: str, joints: tuple) -> None: ...
    def define_obstacle(
        self, name: str, pos: tuple, radius: float, shape: str
    ) -> None: ...
    def is_stopped(self) -> bool: ...


StatusCallback = Callable[[dict], None]


class RuntimeError_(Exception):
    """Programmatic error raised when an executor call fails mid-run."""


class Interpreter:
    """
    Walk a ``Program`` and invoke Executor primitives.

    Usage
    -----
        interp = Interpreter(executor, status_cb=my_highlighter)
        interp.run(program)

    The ``run`` call blocks the calling thread. For the curses editor
    we wrap it in a daemon thread via ``SequenceRunner`` below.
    """

    def __init__(
        self,
        executor: Executor,
        status_cb: Optional[StatusCallback] = None,
    ):
        self._exec = executor
        self._status_cb = status_cb
        self._stop = threading.Event()
        self._current_pass = 0
        self._total_passes = 1
        self._total_statements = 0
        self.errors: List[str] = []

    # ── Public ───────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def run(self, program: Program) -> None:
        """Execute every top-level statement in order."""
        self._stop.clear()
        self.errors = []
        self._total_statements = sum(1 for _ in program.walk())
        self._total_passes = 1
        self._current_pass = 1
        try:
            self._exec_statements(program.body, depth=0)
        except RuntimeError_ as exc:
            self.errors.append(str(exc))

    # ── Statement dispatch ───────────────────────────────────────────────

    def _exec_statements(self, stmts: List[Statement], depth: int) -> None:
        for stmt in stmts:
            if self._should_stop():
                return
            self._emit_status(stmt, depth)
            self._exec_one(stmt, depth)

    def _exec_one(self, stmt: Statement, depth: int) -> None:
        if isinstance(stmt, MoveStmt):
            self._run_move(stmt)
        elif isinstance(stmt, SetJointStmt):
            self._run_set_joint(stmt)
        elif isinstance(stmt, WaitStmt):
            self._run_wait(stmt)
        elif isinstance(stmt, HomeStmt):
            self._safe(lambda: self._exec.go_home(), stmt)
        elif isinstance(stmt, RepeatBlock):
            self._run_repeat(stmt, depth)
        elif isinstance(stmt, IfBlock):
            self._run_if(stmt, depth)
        elif isinstance(stmt, WhileBlock):
            self._run_while(stmt, depth)
        elif isinstance(stmt, DefineState):
            self._safe(
                lambda: self._exec.define_state(stmt.name, stmt.joints), stmt
            )
        elif isinstance(stmt, DefineObstacle):
            self._safe(
                lambda: self._exec.define_obstacle(
                    stmt.name, stmt.pos, stmt.radius, stmt.shape
                ),
                stmt,
            )
        else:
            raise RuntimeError_(
                f"L{getattr(stmt, 'line', 0)}: unknown statement kind {type(stmt).__name__}"
            )

    # ── Leaf runners ─────────────────────────────────────────────────────

    def _run_move(self, stmt: MoveStmt) -> None:
        self._safe(lambda: self._exec.move_to_state(stmt.state_name), stmt)
        if stmt.wait_ms > 0 and not self._should_stop():
            self._exec.wait_ms(stmt.wait_ms)

    def _run_set_joint(self, stmt: SetJointStmt) -> None:
        self._safe(lambda: self._exec.set_joint(stmt.joint_token, stmt.deg), stmt)
        if stmt.wait_ms > 0 and not self._should_stop():
            self._exec.wait_ms(stmt.wait_ms)

    def _run_wait(self, stmt: WaitStmt) -> None:
        if stmt.ms > 0:
            self._exec.wait_ms(stmt.ms)

    def _run_repeat(self, stmt: RepeatBlock, depth: int) -> None:
        # Only the OUTERMOST REPEAT drives the ``pass`` counter shown in
        # the editor status bar. Nested repeats are just loops.
        outer = depth == 0
        if outer:
            self._total_passes = max(1, stmt.count)
        for p in range(max(1, stmt.count)):
            if self._should_stop():
                return
            if outer:
                self._current_pass = p + 1
            self._exec_statements(stmt.body, depth + 1)

    def _run_if(self, stmt: IfBlock, depth: int) -> None:
        if self._eval_condition(stmt.cond):
            self._exec_statements(stmt.then_body, depth + 1)
        else:
            self._exec_statements(stmt.else_body, depth + 1)

    def _run_while(self, stmt: WhileBlock, depth: int) -> None:
        # Safety cap to prevent runaway loops in misconfigured programs.
        # A well-formed program that needs more iterations can use a
        # REPEAT block; kids' sequences should not loop this long.
        _MAX = 10_000
        iters = 0
        while not self._should_stop() and self._eval_condition(stmt.cond):
            self._exec_statements(stmt.body, depth + 1)
            iters += 1
            if iters >= _MAX:
                raise RuntimeError_(
                    f"L{stmt.line}: WHILE exceeded {_MAX} iterations"
                )

    # ── Expression evaluation ────────────────────────────────────────────

    def _eval_condition(self, cond: Condition) -> bool:
        lhs = self._eval_sensor(cond.lhs)
        thr = cond.threshold
        op = cond.op
        if op == "<":
            return lhs < thr
        if op == ">":
            return lhs > thr
        if op == "<=":
            return lhs <= thr
        if op == ">=":
            return lhs >= thr
        if op == "==":
            return lhs == thr
        if op == "!=":
            return lhs != thr
        raise RuntimeError_(f"L{cond.line}: unknown comparison operator '{op}'")

    def _eval_sensor(self, expr: Any) -> float:
        if isinstance(expr, TofExpr):
            return float(self._exec.read_tof(expr.channel))
        if isinstance(expr, IrExpr):
            return float(self._exec.read_ir())
        if isinstance(expr, JointReadExpr):
            return float(self._exec.read_joint(expr.joint_token))
        raise RuntimeError_(f"unknown sensor expression {type(expr).__name__}")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _should_stop(self) -> bool:
        if self._stop.is_set():
            return True
        try:
            return bool(self._exec.is_stopped())
        except Exception:
            return False

    def _safe(self, fn, stmt: Statement) -> None:
        """Run ``fn`` and wrap any raised exception with source-line context."""
        try:
            fn()
        except Exception as exc:
            raise RuntimeError_(f"L{getattr(stmt, 'line', 0)}: {exc}") from exc

    def _emit_status(self, stmt: Statement, depth: int) -> None:
        if self._status_cb is None:
            return
        try:
            self._status_cb(
                {
                    "line": int(getattr(stmt, "line", 0)),
                    "depth": int(depth),
                    "kind": type(stmt).__name__,
                    "pass": (self._current_pass, self._total_passes),
                    "total_statements": self._total_statements,
                }
            )
        except Exception:
            pass  # never let an editor UI bug crash the runner


# ── Concrete executors ────────────────────────────────────────────────────


class BaseExecutor:
    """
    Trivial base that stubs every Executor method. Test fakes and
    partial implementations can subclass this and override only the
    methods they actually use. Sim/hardware executors live in
    ``controller.py`` / ``web/backend/`` where they can reach the
    BraccioBackend.
    """

    def move_to_state(self, name: str) -> None:
        raise NotImplementedError(f"move_to_state('{name}')")

    def set_joint(self, joint_token: str, deg: int) -> None:
        raise NotImplementedError(f"set_joint('{joint_token}', {deg})")

    def go_home(self) -> None:
        raise NotImplementedError("go_home")

    def wait_ms(self, ms: int) -> None:
        time.sleep(ms / 1000.0)

    def read_tof(self, channel: int) -> float:
        raise NotImplementedError(f"read_tof({channel})")

    def read_ir(self) -> int:
        raise NotImplementedError("read_ir")

    def read_joint(self, joint_token: str) -> int:
        raise NotImplementedError(f"read_joint('{joint_token}')")

    def define_state(self, name: str, joints: tuple) -> None:
        raise NotImplementedError(f"define_state('{name}')")

    def define_obstacle(
        self, name: str, pos: tuple, radius: float, shape: str
    ) -> None:
        raise NotImplementedError(f"define_obstacle('{name}')")

    def is_stopped(self) -> bool:
        return False


class SequenceEditorExecutor(BaseExecutor):
    """
    Drop-in executor for the curses sequence editor.

    Backwards compatibility: the old ``SequenceRunner`` only ever
    called ``run_state_fn(name)``. That single callback still works
    for every legacy file (which only produces ``MoveStmt``s). The
    additional Executor methods default to NotImplementedError, so if
    a power user adds a ``SET B 45`` to their curses-edited sequence
    and the editor context doesn't wire up a set-joint hook, the
    Interpreter surfaces the error cleanly via ``errors``.

    Parameters
    ----------
    run_state_fn : callable(name) — moves the arm to a saved state;
                   wraps MotionGuard / send_move from the controller.
    stop_event   : threading.Event that the editor flips when the
                   user presses F6 (Stop). Pollable from wait_ms.
    set_joint_fn : optional callable(idx, deg) — lets sequences run
                   ``SET B 45`` if the controller hooks it up. None by
                   default for the legacy path.
    tof_fn       : optional callable(channel) -> float — lets sequences
                   run ``IF TOF[0] < 200 { ... }`` against live ToF.
    ir_fn        : optional callable() -> int
    joint_fn     : optional callable(idx) -> int
    home_fn      : optional callable() — overrides go_home().
    """

    _TOKEN_TO_IDX = {"B": 0, "S": 1, "E": 2, "WV": 3, "WR": 4, "G": 5}

    def __init__(
        self,
        run_state_fn: Callable[[str], None],
        stop_event: threading.Event,
        set_joint_fn: Optional[Callable[[int, int], None]] = None,
        tof_fn: Optional[Callable[[int], float]] = None,
        ir_fn: Optional[Callable[[], int]] = None,
        joint_fn: Optional[Callable[[int], int]] = None,
        home_fn: Optional[Callable[[], None]] = None,
    ):
        self._run_state_fn = run_state_fn
        self._stop = stop_event
        self._set_joint_fn = set_joint_fn
        self._tof_fn = tof_fn
        self._ir_fn = ir_fn
        self._joint_fn = joint_fn
        self._home_fn = home_fn

    def move_to_state(self, name: str) -> None:
        self._run_state_fn(name)

    def set_joint(self, joint_token: str, deg: int) -> None:
        if self._set_joint_fn is None:
            raise NotImplementedError(
                "SET <joint> is not wired into the curses editor; "
                "run this sequence via the web frontend instead"
            )
        idx = self._TOKEN_TO_IDX[joint_token]
        self._set_joint_fn(idx, deg)

    def go_home(self) -> None:
        if self._home_fn is not None:
            self._home_fn()
        else:
            # Fallback — move to the standard HOME_REST state.
            self._run_state_fn("HOME_REST")

    def wait_ms(self, ms: int) -> None:
        deadline = time.monotonic() + ms / 1000.0
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return
            time.sleep(0.02)

    def read_tof(self, channel: int) -> float:
        if self._tof_fn is None:
            raise NotImplementedError(
                "TOF[...] sensor expressions are not wired into the curses editor"
            )
        return float(self._tof_fn(channel))

    def read_ir(self) -> int:
        if self._ir_fn is None:
            raise NotImplementedError(
                "IR sensor expression is not wired into the curses editor"
            )
        return int(self._ir_fn())

    def read_joint(self, joint_token: str) -> int:
        if self._joint_fn is None:
            raise NotImplementedError(
                "JOINT[...] read expressions are not wired into the curses editor"
            )
        idx = self._TOKEN_TO_IDX[joint_token]
        return int(self._joint_fn(idx))

    def define_state(self, name: str, joints: tuple) -> None:
        # Not persisted from the curses editor. A future extension could
        # hook into StateLibrary, but it is out of scope for the
        # backwards-compatible path.
        raise NotImplementedError(
            f"STATE {name} {{ ... }} definitions are not supported in the "
            f"curses editor; use the web frontend to manage saved states"
        )

    def define_obstacle(
        self, name: str, pos: tuple, radius: float, shape: str
    ) -> None:
        raise NotImplementedError(
            "OBSTACLE { ... } definitions only work in sim mode via the web UI"
        )

    def is_stopped(self) -> bool:
        return self._stop.is_set()


# ── Runner shim (replaces the old SequenceRunner) ────────────────────────


class SequenceRunner:
    """
    Backwards-compatible shim that the curses editor uses.

    Owns the stop event, spawns a daemon thread, and exposes the same
    ``running`` / ``current_step`` / ``current_pass`` / ``total_steps``
    / ``total_passes`` attributes the editor's status bar reads. The
    underlying engine is the new ``Interpreter`` walking a ``Program``
    AST, which means control-flow-rich sequences just work.

    Parameters
    ----------
    program       : Program AST (produced by parser.parse_auto)
    run_state_fn  : callable(name) — same callback the old runner took
    set_joint_fn  : optional callable(idx, deg) — wired from controller
    tof_fn/ir_fn/joint_fn/home_fn : optional sensor/home hooks
    """

    def __init__(
        self,
        program: Program,
        run_state_fn: Callable[[str], None],
        set_joint_fn: Optional[Callable[[int, int], None]] = None,
        tof_fn: Optional[Callable[[int], float]] = None,
        ir_fn: Optional[Callable[[], int]] = None,
        joint_fn: Optional[Callable[[int], int]] = None,
        home_fn: Optional[Callable[[], None]] = None,
    ):
        self._program = program
        self._stop_event = threading.Event()
        self._executor = SequenceEditorExecutor(
            run_state_fn,
            self._stop_event,
            set_joint_fn=set_joint_fn,
            tof_fn=tof_fn,
            ir_fn=ir_fn,
            joint_fn=joint_fn,
            home_fn=home_fn,
        )
        self._interp = Interpreter(self._executor, status_cb=self._on_status)
        self._thread: Optional[threading.Thread] = None

        # Legacy status fields the curses editor reads directly.
        self.running: bool = False
        self.current_step: int = 0
        self.current_pass: int = 0
        self.total_steps: int = _count_direct_move_steps(program)
        self.total_passes: int = _count_outer_repeat_passes(program)
        self.current_line: int = 0
        self.current_kind: str = ""
        self.last_error: str = ""

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.running = True
        self.current_step = 0
        self.current_pass = 0
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._interp.stop()
        self.running = False

    # ── Internals ────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._interp.run(self._program)
            if self._interp.errors:
                self.last_error = self._interp.errors[0]
        finally:
            self.running = False

    def _on_status(self, info: dict) -> None:
        self.current_line = int(info.get("line", 0))
        self.current_kind = str(info.get("kind", ""))
        cp, tp = info.get("pass", (0, 0))
        self.current_pass = int(cp) or self.current_pass
        self.total_passes = int(tp) or self.total_passes
        # Step counter — advance on top-level-ish move statements so the
        # legacy "step N/M" bar keeps making sense for simple files.
        if self.current_kind in ("MoveStmt", "SetJointStmt", "HomeStmt"):
            self.current_step = min(self.current_step + 1, self.total_steps or 10**6)


# ── Static analysis helpers ──────────────────────────────────────────────


def _count_direct_move_steps(program: Program) -> int:
    """
    Count the MoveStmt / SetJointStmt / HomeStmt instances in a single
    pass — used to size the editor's step counter. For a legacy file
    this is exactly ``len(steps)`` the old parser returned.
    """
    total = 0
    for _depth, stmt in program.walk():
        if isinstance(stmt, (MoveStmt, SetJointStmt, HomeStmt)):
            total += 1
    return max(total, 1)


def _count_outer_repeat_passes(program: Program) -> int:
    """
    Return the count of the FIRST top-level RepeatBlock, or 1 if there
    isn't one. Matches the legacy ``REPEAT N`` semantics.
    """
    for stmt in program.body:
        if isinstance(stmt, RepeatBlock):
            return max(1, stmt.count)
    return 1


__all__ = [
    "Executor",
    "BaseExecutor",
    "Interpreter",
    "SequenceEditorExecutor",
    "SequenceRunner",
    "RuntimeError_",
]
