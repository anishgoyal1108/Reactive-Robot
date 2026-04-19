"""
dsl — Extended Braccio DSL package (Phase 2 of digital_twin_plan).

Public API
----------

::

    from braccio_ctrl.dsl import parse, parse_auto, Program, Interpreter
    from braccio_ctrl.dsl import SequenceRunner, SequenceEditorExecutor
    from braccio_ctrl.dsl import ParseError, RuntimeError_

``parse(text)`` is the strict Lark-based parser.
``parse_auto(text)`` auto-detects legacy vs. new format and always
returns ``(Program, errors)`` — safe to call from the curses editor
on every keystroke. Use it instead of ``parse`` anywhere backwards
compatibility with ``.txt`` files produced by the old editor matters.

``SequenceRunner`` is a drop-in replacement for the original
``SequenceRunner`` that used to live inside ``sequence_editor.py``.
It reads a ``Program`` instead of a flat step list, but exposes the
same ``running``/``current_step``/``current_pass`` fields the UI
status bar consumes.
"""

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
    SensorExpr,
    SetJointStmt,
    Statement,
    TofExpr,
    WaitStmt,
    WhileBlock,
)
from .compat import looks_legacy, parse_legacy
from .interpreter import (
    BaseExecutor,
    Executor,
    Interpreter,
    RuntimeError_,
    SequenceEditorExecutor,
    SequenceRunner,
)
from .parser import ParseError, parse, parse_auto

__all__ = [
    # AST
    "Program",
    "Statement",
    "MoveStmt",
    "SetJointStmt",
    "WaitStmt",
    "HomeStmt",
    "RepeatBlock",
    "IfBlock",
    "WhileBlock",
    "DefineState",
    "DefineObstacle",
    "Condition",
    "TofExpr",
    "IrExpr",
    "JointReadExpr",
    "SensorExpr",
    # Parsers
    "parse",
    "parse_auto",
    "ParseError",
    "looks_legacy",
    "parse_legacy",
    # Runtime
    "Executor",
    "BaseExecutor",
    "Interpreter",
    "RuntimeError_",
    "SequenceEditorExecutor",
    "SequenceRunner",
]
