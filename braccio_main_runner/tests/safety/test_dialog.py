"""Unit tests for RefusalDialog (pure state, no curses)."""

from dataclasses import dataclass

import pytest

from braccio_ctrl.safety.dialog import DialogChoice, RefusalDialog


@dataclass
class _Info:
    attempted_q: list
    current_q: list
    reason: str
    nearest_link: str = ""
    nearest_gap_mm: float = 0.0


def test_not_open_before_open_call():
    d = RefusalDialog()
    assert not d.is_open
    assert d.render_lines() == []
    assert d.handle_key(ord("r")) is None


def test_open_populates_render_lines():
    d = RefusalDialog()
    d.open(_Info(
        attempted_q=[90, 60, 120, 90, 90, 73],
        current_q=[90, 90, 90, 90, 90, 73],
        reason="goal_in_collision",
        nearest_link="forearm",
        nearest_gap_mm=-12.0,
    ))
    lines = d.render_lines()
    assert any("REFUSAL" in line.text for line in lines)
    assert any("goal_in_collision" in line.text for line in lines)
    assert any("forearm" in line.text for line in lines)


@pytest.mark.parametrize("key, expected", [
    (ord("r"), DialogChoice.RETREAT),
    (ord("R"), DialogChoice.RETREAT),
    (ord("a"), DialogChoice.TRY_AXIS),
    (ord("s"), DialogChoice.REDUCE_STEP),
    (ord("o"), DialogChoice.OVERRIDE),
    (ord("c"), DialogChoice.CANCEL),
    (27,       DialogChoice.CANCEL),
])
def test_handle_key_recognises_choices(key, expected):
    d = RefusalDialog()
    d.open(_Info(attempted_q=[], current_q=[], reason=""))
    assert d.handle_key(key) is expected
    assert not d.is_open


def test_unknown_key_keeps_dialog_open():
    d = RefusalDialog()
    d.open(_Info(attempted_q=[], current_q=[], reason=""))
    assert d.handle_key(ord("x")) is None
    assert d.is_open
    assert d.handle_key(ord("C")) is DialogChoice.CANCEL
    assert not d.is_open
