"""
parser.py — Extended Braccio DSL parser (Lark-backed).

Exposes:

    parse(text)        — strict parser: raises ParseError on failure
    parse_auto(text)   — try legacy format first, fall back to the new
                         grammar. Returns (Program, errors) where errors
                         is a list of human-readable strings (empty on
                         success). Used by sequence_editor.py so existing
                         TEST_INPUT_SEQUENCE.txt programs keep working.

The heavy lifting is done by Lark + a Transformer that walks the raw
parse tree and emits dataclass AST nodes from ast_nodes.py.

Line tracking
-------------
Each AST node carries a 1-based ``line`` field sourced from the first
token of the corresponding parse-tree node. The editor uses this for
the "currently executing" highlight.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import LarkError, UnexpectedInput

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
    TofExpr,
    WaitStmt,
    WhileBlock,
)


class ParseError(Exception):
    """Raised when the Lark parser cannot make sense of the text."""


_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")


def _load_parser() -> Lark:
    with open(_GRAMMAR_PATH) as f:
        grammar = f.read()
    # LALR is strict but fast enough for a ~100 line sequence. If we ever
    # need ambiguity handling we can switch to Earley without changing
    # the grammar itself.
    return Lark(grammar, parser="lalr", propagate_positions=True)


# Module-level singleton so sequence-editor keystrokes don't pay the
# grammar compile cost on every re-parse.
_parser: Lark | None = None


def _get_parser() -> Lark:
    global _parser
    if _parser is None:
        _parser = _load_parser()
    return _parser


# ── Tree → AST transformer ────────────────────────────────────────────────


def _tok_line(tok) -> int:
    """Return a 1-based line number for any Lark Token or Tree, or 0."""
    line = getattr(tok, "line", None)
    if line is None and hasattr(tok, "meta"):
        line = getattr(tok.meta, "line", None)
    return int(line) if line else 0


@v_args(inline=False)
class _ToAst(Transformer):
    """
    Walk the Lark parse tree bottom-up and emit the dataclass AST.

    Every rule matches a top-level production in grammar.lark. The
    transformer methods receive the list of child results (already
    transformed for non-terminals, raw Tokens for terminals).
    """

    # ── Sensor leaves ────────────────────────────────────────────────────

    def tof_expr(self, children):
        # children = [TOF_TOKEN, LBRACK_TOKEN, INT_TOKEN, RBRACK_TOKEN]
        int_tok = _find_int(children)
        return TofExpr(channel=int(int_tok), line=_tok_line(children[0]))

    def ir_expr(self, children):
        return IrExpr(line=_tok_line(children[0]))

    def joint_read_expr(self, children):
        joint_tok = _find_joint(children)
        return JointReadExpr(joint_token=str(joint_tok), line=_tok_line(children[0]))

    def cond(self, children):
        # children = [sensor_expr, CMP_TOKEN, SIGNED_NUMBER_TOKEN]
        lhs = children[0]
        op = str(children[1])
        thresh = float(children[2])
        return Condition(lhs=lhs, op=op, threshold=thresh, line=getattr(lhs, "line", 0))

    # ── Statements ───────────────────────────────────────────────────────

    def move_stmt(self, children):
        # children = [MOVE_TOKEN, IDENT_TOKEN, (WAIT_TOKEN, INT_TOKEN)?]
        line = _tok_line(children[0])
        state_name = str(children[1])
        wait_ms = 0
        # Look for INT after a WAIT token — order in grammar is fixed.
        for i, c in enumerate(children):
            if isinstance(c, Token) and c.type == "WAIT":
                wait_ms = int(children[i + 1])
                break
        return MoveStmt(state_name=state_name, wait_ms=wait_ms, line=line)

    def set_joint_stmt(self, children):
        # children = [SET, JOINT, INT, (WAIT, INT)?]
        line = _tok_line(children[0])
        joint_tok = str(children[1])
        deg = int(children[2])
        wait_ms = 0
        for i, c in enumerate(children):
            if isinstance(c, Token) and c.type == "WAIT":
                wait_ms = int(children[i + 1])
                break
        return SetJointStmt(joint_token=joint_tok, deg=deg, wait_ms=wait_ms, line=line)

    def wait_stmt(self, children):
        line = _tok_line(children[0])
        return WaitStmt(ms=int(children[1]), line=line)

    def home_stmt(self, children):
        return HomeStmt(line=_tok_line(children[0]))

    def repeat_blk(self, children):
        # children = [REPEAT, INT, LBRACE, stmt*, RBRACE]
        line = _tok_line(children[0])
        count = int(children[1])
        body = [
            c for c in children[3:-1] if not isinstance(c, Token)
        ]
        return RepeatBlock(count=count, body=body, line=line)

    def if_blk(self, children):
        # [IF, cond, LBRACE, stmt*, RBRACE] or with optional ELSE { ... }
        line = _tok_line(children[0])
        cond = children[1]
        # Walk children to split then/else blocks around LBRACE/RBRACE markers.
        then_body: list = []
        else_body: list = []
        # Find the LBRACE after cond, then collect until matching RBRACE.
        idx = 2  # first LBRACE
        assert isinstance(children[idx], Token) and children[idx].type == "LBRACE"
        idx += 1
        while idx < len(children) and not (
            isinstance(children[idx], Token) and children[idx].type == "RBRACE"
        ):
            then_body.append(children[idx])
            idx += 1
        idx += 1  # skip RBRACE
        # Optional ELSE branch
        if (
            idx < len(children)
            and isinstance(children[idx], Token)
            and children[idx].type == "ELSE"
        ):
            idx += 1  # skip ELSE
            assert isinstance(children[idx], Token) and children[idx].type == "LBRACE"
            idx += 1
            while idx < len(children) and not (
                isinstance(children[idx], Token) and children[idx].type == "RBRACE"
            ):
                else_body.append(children[idx])
                idx += 1
        return IfBlock(cond=cond, then_body=then_body, else_body=else_body, line=line)

    def while_blk(self, children):
        line = _tok_line(children[0])
        cond = children[1]
        body = [
            c for c in children[3:-1] if not isinstance(c, Token)
        ]
        return WhileBlock(cond=cond, body=body, line=line)

    def def_state(self, children):
        # [STATE, IDENT, LBRACE, JOINTS, INT*6, RBRACE]
        line = _tok_line(children[0])
        name = str(children[1])
        # The six INTs are the last six tokens before RBRACE.
        ints = [int(c) for c in children if isinstance(c, Token) and c.type == "INT"]
        if len(ints) != 6:
            raise ParseError(
                f"L{line}: STATE {name} expected 6 joint angles, got {len(ints)}"
            )
        joints = tuple(ints)
        return DefineState(name=name, joints=joints, line=line)  # type: ignore[arg-type]

    def def_obstacle(self, children):
        # [OBSTACLE, IDENT, LBRACE, obs_field+, RBRACE]
        line = _tok_line(children[0])
        name = str(children[1])
        pos = (0.0, 0.0, 0.0)
        radius = 20.0
        shape = "SPHERE"
        for c in children:
            if isinstance(c, dict):
                if "pos" in c:
                    pos = c["pos"]
                elif "radius" in c:
                    radius = c["radius"]
                elif "shape" in c:
                    shape = c["shape"]
        return DefineObstacle(
            name=name, pos=pos, radius=radius, shape=shape, line=line
        )

    def obs_field(self, children):
        # Passes through the single child (already a dict from below).
        return children[0]

    def obs_pos(self, children):
        # [POS, SIGNED_NUMBER, SIGNED_NUMBER, SIGNED_NUMBER]
        return {"pos": (float(children[1]), float(children[2]), float(children[3]))}

    def obs_radius(self, children):
        return {"radius": float(children[1])}

    def obs_shape(self, children):
        return {"shape": str(children[1])}

    # ── Program root ─────────────────────────────────────────────────────

    def start(self, children):
        body = [c for c in children if not isinstance(c, Token)]
        return Program(body=body)


def _find_int(children) -> Token:
    for c in children:
        if isinstance(c, Token) and c.type == "INT":
            return c
    raise ParseError("internal: expected INT token not found")


def _find_joint(children) -> Token:
    for c in children:
        if isinstance(c, Token) and c.type == "JOINT":
            return c
    raise ParseError("internal: expected JOINT token not found")


# ── Public API ────────────────────────────────────────────────────────────


def parse(text: str) -> Program:
    """
    Strict parse — the text must conform to grammar.lark.

    Raises ParseError with a location hint on any failure.
    """
    parser = _get_parser()
    try:
        tree = parser.parse(text)
    except UnexpectedInput as exc:
        line = getattr(exc, "line", "?")
        col = getattr(exc, "column", "?")
        raise ParseError(f"L{line}:{col}: {exc.__class__.__name__}") from exc
    except LarkError as exc:
        raise ParseError(str(exc)) from exc

    try:
        return _ToAst().transform(tree)
    except ParseError:
        raise
    except Exception as exc:  # defensive — transformer logic bug
        raise ParseError(f"AST construction failed: {exc}") from exc


def parse_auto(text: str) -> Tuple[Program, List[str]]:
    """
    Lenient parse used by the curses editor.

    1. Run the legacy detector — if every non-empty line looks legacy,
       use compat.parse_legacy. This preserves bit-identical behaviour
       for TEST_INPUT_SEQUENCE.txt and any saved `.txt` on disk.
    2. Otherwise try the Lark grammar.
    3. On Lark failure, return an empty Program plus a list of errors
       so the editor can show a red status message.
    """
    from . import compat

    if compat.looks_legacy(text):
        return compat.parse_legacy(text)

    try:
        return parse(text), []
    except ParseError as exc:
        return Program(body=[]), [str(exc)]


__all__ = ["parse", "parse_auto", "ParseError"]
