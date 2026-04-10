"""Small helpers for matplotlib + static typing (stub gaps, shared callers)."""

from __future__ import annotations

from typing import Protocol, TypeVar, cast

__all__ = ["figure_number"]


class HasFigureNumber(Protocol):
    """matplotlib.figure.Figure exposes .number for pyplot.fignum_exists."""

    number: int


_FigT = TypeVar("_FigT")


def figure_number(fig: _FigT) -> int:
    return cast(HasFigureNumber, fig).number
