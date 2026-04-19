"""
ast_nodes.py — Dataclasses for the Braccio DSL AST.

The parser (parser.py / compat.py) produces one of these trees, and the
interpreter (interpreter.py) walks it. All nodes carry a ``line`` field
(1-based line number in the source text, or 0 for synthetic nodes
emitted by the legacy compat shim) so the editor can highlight the
currently-executing statement.

Design notes
------------
- The AST intentionally mirrors the grammar 1:1. Blockly codegen can
  produce nodes directly without going through text, once Phase 5 lands.
- Integers (joint angles, wait durations) are stored as ``int``; floats
  (obstacle coordinates, comparison thresholds) as ``float``. The
  interpreter coerces on demand — see ``Condition.threshold``.
- ``Program`` is the root; every other node is a statement or a leaf
  used inside a statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


# ── Sensor expression leaves ──────────────────────────────────────────────


@dataclass
class TofExpr:
    """``TOF[<channel>]`` — current ToF reading on channel 0..3 in mm."""

    channel: int
    line: int = 0


@dataclass
class IrExpr:
    """``IR`` — current IR severity (0..3)."""

    line: int = 0


@dataclass
class JointReadExpr:
    """``JOINT[<token>]`` — current joint angle in degrees."""

    joint_token: str  # "B" | "S" | "E" | "WV" | "WR" | "G"
    line: int = 0


SensorExpr = Union[TofExpr, IrExpr, JointReadExpr]


@dataclass
class Condition:
    """Binary comparison ``sensor_expr <cmp> number`` for IF/WHILE heads."""

    lhs: SensorExpr
    op: str  # "<", ">", "<=", ">=", "==", "!="
    threshold: float
    line: int = 0


# ── Statements ────────────────────────────────────────────────────────────


@dataclass
class MoveStmt:
    """``MOVE <state_name> [WAIT <ms>]`` — go to a saved state, then wait."""

    state_name: str
    wait_ms: int = 0
    line: int = 0


@dataclass
class SetJointStmt:
    """``SET <joint_token> <deg> [WAIT <ms>]`` — single-joint absolute move."""

    joint_token: str  # "B"|"S"|"E"|"WV"|"WR"|"G"
    deg: int
    wait_ms: int = 0
    line: int = 0


@dataclass
class WaitStmt:
    """``WAIT <ms>`` — pause execution."""

    ms: int
    line: int = 0


@dataclass
class HomeStmt:
    """``HOME`` — shorthand for moving to the hardware HOME pose."""

    line: int = 0


@dataclass
class RepeatBlock:
    """``REPEAT <n> { ... }`` — run body N times."""

    count: int
    body: List["Statement"] = field(default_factory=list)
    line: int = 0


@dataclass
class IfBlock:
    """``IF <cond> { ... } [ELSE { ... }]``."""

    cond: Condition
    then_body: List["Statement"] = field(default_factory=list)
    else_body: List["Statement"] = field(default_factory=list)
    line: int = 0


@dataclass
class WhileBlock:
    """``WHILE <cond> { ... }`` — loops until the condition goes false."""

    cond: Condition
    body: List["Statement"] = field(default_factory=list)
    line: int = 0


@dataclass
class DefineState:
    """``STATE <name> { JOINTS b s e wv wr g }`` — define a named pose inline."""

    name: str
    joints: Tuple[int, int, int, int, int, int]
    line: int = 0


@dataclass
class DefineObstacle:
    """``OBSTACLE <name> { POS x y z; RADIUS r; SHAPE kind }``.

    Sim-only. The interpreter asks the executor to register it — the
    hardware backend rejects the call (there is no such thing as
    spawning a real obstacle).
    """

    name: str
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 20.0
    shape: str = "SPHERE"  # "BOX" | "SPHERE" | "CYLINDER"
    line: int = 0


Statement = Union[
    MoveStmt,
    SetJointStmt,
    WaitStmt,
    HomeStmt,
    RepeatBlock,
    IfBlock,
    WhileBlock,
    DefineState,
    DefineObstacle,
]


@dataclass
class Program:
    """Root of the AST — flat list of top-level statements."""

    body: List[Statement] = field(default_factory=list)

    def walk(self):
        """Depth-first iterator over every statement, yielding ``(depth, stmt)``."""
        stack: list = [(0, s) for s in reversed(self.body)]
        while stack:
            depth, stmt = stack.pop()
            yield depth, stmt
            if isinstance(stmt, RepeatBlock):
                for s in reversed(stmt.body):
                    stack.append((depth + 1, s))
            elif isinstance(stmt, IfBlock):
                for s in reversed(stmt.then_body):
                    stack.append((depth + 1, s))
                for s in reversed(stmt.else_body):
                    stack.append((depth + 1, s))
            elif isinstance(stmt, WhileBlock):
                for s in reversed(stmt.body):
                    stack.append((depth + 1, s))


__all__ = [
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
]
