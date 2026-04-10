"""
test_dsl_parser.py — Grammar / AST tests for braccio_ctrl.dsl.parser.

Covers the Lark-based parser entry points:
    parse(text)       — strict, raises on failure
    parse_auto(text)  — lenient, returns (Program, errors)

and the AST shapes produced for every statement kind.
"""

from __future__ import annotations

import pytest

from braccio_ctrl.dsl import (
    Condition,
    DefineObstacle,
    DefineState,
    HomeStmt,
    IfBlock,
    IrExpr,
    JointReadExpr,
    MoveStmt,
    ParseError,
    Program,
    RepeatBlock,
    SetJointStmt,
    TofExpr,
    WaitStmt,
    WhileBlock,
    parse,
    parse_auto,
)


# ── MoveStmt ─────────────────────────────────────────────────────────────


def test_move_without_wait():
    prog = parse("MOVE HOME_REST")
    assert len(prog.body) == 1
    m = prog.body[0]
    assert isinstance(m, MoveStmt)
    assert m.state_name == "HOME_REST"
    assert m.wait_ms == 0


def test_move_with_wait():
    prog = parse("MOVE STOW_COMPACT WAIT 500")
    m = prog.body[0]
    assert isinstance(m, MoveStmt)
    assert m.state_name == "STOW_COMPACT"
    assert m.wait_ms == 500


def test_move_preserves_source_line():
    prog = parse("\n\nMOVE HOME_REST WAIT 100\n")
    assert prog.body[0].line == 3


# ── SetJointStmt ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "token,deg",
    [("B", 45), ("S", 120), ("E", 0), ("WV", 180), ("WR", 90), ("G", 55)],
)
def test_set_joint_all_tokens(token, deg):
    prog = parse(f"SET {token} {deg}")
    s = prog.body[0]
    assert isinstance(s, SetJointStmt)
    assert s.joint_token == token
    assert s.deg == deg


def test_set_joint_with_wait():
    prog = parse("SET WV 135 WAIT 250")
    s = prog.body[0]
    assert s.joint_token == "WV"
    assert s.deg == 135
    assert s.wait_ms == 250


# ── WaitStmt / HomeStmt ──────────────────────────────────────────────────


def test_wait_stmt():
    prog = parse("WAIT 1000")
    w = prog.body[0]
    assert isinstance(w, WaitStmt)
    assert w.ms == 1000


def test_home_stmt():
    prog = parse("HOME")
    h = prog.body[0]
    assert isinstance(h, HomeStmt)


def test_home_vs_home_rest_disambiguation():
    """HOME_REST must lex as an IDENT even though HOME is a keyword."""
    prog = parse("MOVE HOME_REST\nHOME")
    assert isinstance(prog.body[0], MoveStmt)
    assert prog.body[0].state_name == "HOME_REST"
    assert isinstance(prog.body[1], HomeStmt)


# ── RepeatBlock ──────────────────────────────────────────────────────────


def test_repeat_block():
    prog = parse("REPEAT 3 { MOVE HOME_REST }")
    r = prog.body[0]
    assert isinstance(r, RepeatBlock)
    assert r.count == 3
    assert len(r.body) == 1
    assert isinstance(r.body[0], MoveStmt)


def test_nested_repeat():
    src = "REPEAT 2 { REPEAT 3 { MOVE A } }"
    prog = parse(src)
    outer = prog.body[0]
    assert isinstance(outer, RepeatBlock) and outer.count == 2
    inner = outer.body[0]
    assert isinstance(inner, RepeatBlock) and inner.count == 3
    assert isinstance(inner.body[0], MoveStmt)


def test_repeat_empty_body():
    prog = parse("REPEAT 5 { }")
    r = prog.body[0]
    assert isinstance(r, RepeatBlock)
    assert r.body == []


# ── IfBlock ──────────────────────────────────────────────────────────────


def test_if_with_tof():
    prog = parse("IF TOF[0] < 250 { MOVE STOW }")
    b = prog.body[0]
    assert isinstance(b, IfBlock)
    assert isinstance(b.cond.lhs, TofExpr)
    assert b.cond.lhs.channel == 0
    assert b.cond.op == "<"
    assert b.cond.threshold == 250.0
    assert len(b.then_body) == 1
    assert b.else_body == []


def test_if_else():
    src = "IF IR >= 2 { HOME } ELSE { MOVE STOW }"
    prog = parse(src)
    b = prog.body[0]
    assert isinstance(b, IfBlock)
    assert isinstance(b.cond.lhs, IrExpr)
    assert b.cond.op == ">="
    assert isinstance(b.then_body[0], HomeStmt)
    assert isinstance(b.else_body[0], MoveStmt)


@pytest.mark.parametrize("op", ["<", ">", "<=", ">=", "==", "!="])
def test_all_comparison_ops(op):
    prog = parse(f"IF TOF[1] {op} 100 {{ HOME }}")
    assert prog.body[0].cond.op == op


def test_if_with_joint_read():
    prog = parse("IF JOINT[B] > 90 { HOME }")
    b = prog.body[0]
    assert isinstance(b.cond.lhs, JointReadExpr)
    assert b.cond.lhs.joint_token == "B"


# ── WhileBlock ───────────────────────────────────────────────────────────


def test_while_block():
    prog = parse("WHILE IR > 0 { WAIT 100 }")
    w = prog.body[0]
    assert isinstance(w, WhileBlock)
    assert isinstance(w.cond.lhs, IrExpr)
    assert w.cond.op == ">"
    assert w.cond.threshold == 0.0


# ── DefineState ──────────────────────────────────────────────────────────


def test_define_state():
    prog = parse("STATE MY_POSE { JOINTS 90 90 90 90 90 73 }")
    d = prog.body[0]
    assert isinstance(d, DefineState)
    assert d.name == "MY_POSE"
    assert d.joints == (90, 90, 90, 90, 90, 73)


# ── DefineObstacle ───────────────────────────────────────────────────────


def test_define_obstacle_all_fields():
    src = "OBSTACLE B1 { POS 100 50 30 RADIUS 25 SHAPE BOX }"
    prog = parse(src)
    d = prog.body[0]
    assert isinstance(d, DefineObstacle)
    assert d.name == "B1"
    assert d.pos == (100.0, 50.0, 30.0)
    assert d.radius == 25.0
    assert d.shape == "BOX"


def test_define_obstacle_field_order_independent():
    src = "OBSTACLE S1 { RADIUS 40 SHAPE SPHERE POS -10 -20 -30 }"
    prog = parse(src)
    d = prog.body[0]
    assert d.pos == (-10.0, -20.0, -30.0)
    assert d.radius == 40.0
    assert d.shape == "SPHERE"


def test_define_obstacle_float_pos():
    prog = parse("OBSTACLE F { POS 1.5 2.25 -3.75 RADIUS 5.5 SHAPE CYLINDER }")
    d = prog.body[0]
    assert d.pos == pytest.approx((1.5, 2.25, -3.75))
    assert d.radius == pytest.approx(5.5)


# ── Comments & whitespace ────────────────────────────────────────────────


def test_comments_are_ignored():
    src = """
    # leading comment
    MOVE HOME_REST   # inline not supported, but standalone is
    # another
    MOVE STOW_COMPACT
    """
    prog = parse(src)
    assert len(prog.body) == 2
    assert prog.body[0].state_name == "HOME_REST"
    assert prog.body[1].state_name == "STOW_COMPACT"


def test_extra_whitespace_is_tolerated():
    prog = parse("  MOVE    HOME_REST     WAIT    123  ")
    assert prog.body[0].wait_ms == 123


# ── Error handling ───────────────────────────────────────────────────────


def test_bad_syntax_raises():
    with pytest.raises(ParseError):
        parse("MOVE")


def test_missing_brace_raises():
    with pytest.raises(ParseError):
        parse("REPEAT 3 { MOVE A")


def test_non_upper_ident_rejected():
    with pytest.raises(ParseError):
        parse("MOVE home_rest")  # lowercase not allowed


def test_empty_program_is_valid():
    prog = parse("")
    assert isinstance(prog, Program)
    assert prog.body == []


# ── parse_auto() legacy detection ────────────────────────────────────────


def test_parse_auto_legacy_flat():
    prog, errs = parse_auto("HOME_REST 500\nSTOW_COMPACT 200")
    assert errs == []
    assert len(prog.body) == 2
    assert isinstance(prog.body[0], MoveStmt)
    assert prog.body[0].wait_ms == 500


def test_parse_auto_legacy_with_repeat():
    prog, errs = parse_auto("HOME_REST 100\nSTOW_COMPACT 100\nREPEAT 3\n")
    assert errs == []
    assert len(prog.body) == 1
    r = prog.body[0]
    assert isinstance(r, RepeatBlock)
    assert r.count == 3
    assert len(r.body) == 2


def test_parse_auto_new_format():
    prog, errs = parse_auto("REPEAT 2 { MOVE HOME_REST WAIT 100 }")
    assert errs == []
    assert isinstance(prog.body[0], RepeatBlock)


def test_parse_auto_returns_errors_instead_of_raising():
    prog, errs = parse_auto("REPEAT 3 { MOVE ")
    assert prog.body == []
    assert len(errs) >= 1


def test_parse_auto_legacy_bad_wait():
    prog, errs = parse_auto("HOME_REST abc")
    # Legacy parser reports an error and drops the line.
    assert any("wait" in e.lower() or "HOME_REST" in e for e in errs)


# ── Real-world fixture ──────────────────────────────────────────────────


LEGACY_FIXTURE = """# META_FULL_DEMO
HOME_REST 500
STOW_COMPACT 500

# descend and grab
LOW_APPROACH_OPEN 800
LOW_GRAB_CLOSE 500

# lift
LIFT_MID_CARRY 700
LIFT_HIGH_CARRY 800

REPEAT 3
"""


def test_legacy_test_input_sequence_parses():
    prog, errs = parse_auto(LEGACY_FIXTURE)
    assert errs == []
    assert len(prog.body) == 1
    r = prog.body[0]
    assert isinstance(r, RepeatBlock)
    assert r.count == 3
    # Six moves, each with their wait_ms preserved.
    moves = [s for s in r.body if isinstance(s, MoveStmt)]
    assert len(moves) == 6
    assert moves[0].state_name == "HOME_REST"
    assert moves[0].wait_ms == 500
    assert moves[-1].state_name == "LIFT_HIGH_CARRY"
    assert moves[-1].wait_ms == 800
