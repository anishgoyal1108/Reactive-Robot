"""
test_dsl_compat.py — Legacy line-based format → Program AST.

Covers the compat shim that lets Phase 2 accept every pre-Phase-2
``.txt`` file produced by the old curses editor.
"""

from __future__ import annotations

from braccio_ctrl.dsl import MoveStmt, RepeatBlock
from braccio_ctrl.dsl.compat import looks_legacy, parse_legacy


# ── looks_legacy() heuristic ──────────────────────────────────────────────


def test_empty_text_is_legacy():
    assert looks_legacy("")
    assert looks_legacy("\n\n\n")


def test_comments_only_is_legacy():
    assert looks_legacy("# only a comment\n# and another")


def test_simple_moves_are_legacy():
    assert looks_legacy("HOME_REST 500\nSTOW_COMPACT 200")


def test_repeat_line_is_legacy():
    assert looks_legacy("HOME_REST 500\nREPEAT 3")


def test_braces_disqualify_legacy():
    assert not looks_legacy("REPEAT 3 { MOVE HOME_REST }")


def test_move_keyword_disqualifies_legacy():
    assert not looks_legacy("MOVE HOME_REST WAIT 500")


def test_if_keyword_disqualifies_legacy():
    assert not looks_legacy("IF TOF[0] < 200 { HOME }")


def test_unparseable_line_disqualifies_legacy():
    assert not looks_legacy("this is not a state name")


# ── parse_legacy() ────────────────────────────────────────────────────────


def test_empty_returns_empty_program():
    prog, errs = parse_legacy("")
    assert prog.body == []
    assert errs == []


def test_single_move_without_repeat_is_flat():
    prog, errs = parse_legacy("HOME_REST 500")
    assert errs == []
    assert len(prog.body) == 1
    m = prog.body[0]
    assert isinstance(m, MoveStmt)
    assert m.state_name == "HOME_REST"
    assert m.wait_ms == 500


def test_multiple_moves_without_repeat_are_flat():
    prog, errs = parse_legacy("A 1\nB 2\nC 3")
    assert errs == []
    assert len(prog.body) == 3
    assert all(isinstance(s, MoveStmt) for s in prog.body)


def test_repeat_wraps_entire_program():
    prog, errs = parse_legacy("A 1\nB 2\nREPEAT 4")
    assert errs == []
    assert len(prog.body) == 1
    r = prog.body[0]
    assert isinstance(r, RepeatBlock)
    assert r.count == 4
    assert len(r.body) == 2


def test_repeat_position_does_not_matter():
    """Legacy REPEAT applies to the whole program regardless of placement."""
    prog1, _ = parse_legacy("REPEAT 2\nA 1\nB 1")
    prog2, _ = parse_legacy("A 1\nREPEAT 2\nB 1")
    prog3, _ = parse_legacy("A 1\nB 1\nREPEAT 2")
    for prog in (prog1, prog2, prog3):
        assert isinstance(prog.body[0], RepeatBlock)
        assert prog.body[0].count == 2
        assert len(prog.body[0].body) == 2


def test_last_repeat_wins_matching_legacy_parser():
    prog, _ = parse_legacy("A 1\nREPEAT 2\nREPEAT 5")
    assert prog.body[0].count == 5


def test_repeat_1_does_not_wrap():
    prog, _ = parse_legacy("A 1\nREPEAT 1")
    assert not any(isinstance(s, RepeatBlock) for s in prog.body)


def test_comments_and_blank_lines_are_skipped():
    text = """
    # comment
    A 1

    # another

    B 2
    """
    prog, errs = parse_legacy(text)
    assert errs == []
    moves = [s for s in prog.body if isinstance(s, MoveStmt)]
    assert [m.state_name for m in moves] == ["A", "B"]


def test_error_for_non_numeric_wait():
    prog, errs = parse_legacy("HOME_REST abc")
    assert prog.body == []
    assert len(errs) == 1
    assert "wait" in errs[0].lower()


def test_error_for_unparseable_line():
    prog, errs = parse_legacy("this is clearly wrong")
    assert prog.body == []
    assert len(errs) == 1
    assert "L1" in errs[0]


def test_line_numbers_are_1_based():
    prog, errs = parse_legacy("\n\nHOME_REST 100")
    assert prog.body[0].line == 3


def test_real_world_test_input_fixture():
    """Verbatim copy of braccio_main_runner/TEST_INPUT_SEQUENCE.txt (trimmed)."""
    text = """# META_FULL_DEMO
HOME_REST 500
STOW_COMPACT 500

# descend and grab
LOW_APPROACH_OPEN 800
LOW_GRAB_CLOSE 500

# lift through multiple heights
LIFT_MID_CARRY 700
LIFT_HIGH_CARRY 800

# rotate
ROTATE_LEFT_SCAN 700
ROTATE_RIGHT_SCAN 700

REPEAT 3
"""
    prog, errs = parse_legacy(text)
    assert errs == []
    assert len(prog.body) == 1
    r = prog.body[0]
    assert isinstance(r, RepeatBlock)
    assert r.count == 3
    assert len(r.body) == 8
