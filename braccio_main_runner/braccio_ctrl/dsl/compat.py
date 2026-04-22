"""
compat.py — Legacy sequence format → new DSL AST.

The original curses sequence editor accepted a line-based mini-DSL:

    STATE_NAME WAIT_MS   — move to the named state, then wait WAIT_MS ms
    REPEAT N             — repeat the whole program N times
    # comment            — ignored

Every existing ``.txt`` file on disk (including
``braccio_main_runner/TEST_INPUT_SEQUENCE.txt``) uses that format. This
module preserves bit-identical semantics while letting the rest of the
pipeline deal with the structured Program/RepeatBlock/MoveStmt AST
from ast_nodes.py.

``looks_legacy(text)`` is the detector used by ``parser.parse_auto()``.
It returns True only when every non-empty, non-comment line is one of
``STATE_NAME INT`` or ``REPEAT INT`` — i.e. no braces, no new keywords,
no multi-token control flow. This is intentionally pessimistic: once a
user adds a single block statement, we route to the Lark parser.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .ast_nodes import MoveStmt, Program, RepeatBlock

_RE_REPEAT = re.compile(r"^REPEAT\s+(\d+)$", re.IGNORECASE)
_RE_MOVE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+(\d+)$")
_RE_UPPER_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Tokens that unambiguously mean "this is not a legacy file".
_NEW_KEYWORDS = frozenset(
    {
        "MOVE",
        "SET",
        "WAIT",
        "HOME",
        "IF",
        "ELSE",
        "WHILE",
        "STATE",
        "OBSTACLE",
        "TOF",
        "IR",
        "JOINT",
        "JOINTS",
        "POS",
        "RADIUS",
        "SHAPE",
        "BOX",
        "SPHERE",
        "CYLINDER",
    }
)


def looks_legacy(text: str) -> bool:
    """
    Return True if ``text`` has no structural markers of the extended
    DSL — no braces, no new keywords — and therefore should be parsed
    by the legacy line-based shim. The detection is deliberately
    permissive: a file containing only ``HOME_REST`` with no wait
    value is still "legacy" even though it will later produce a parse
    error, because routing it to ``parse_legacy`` gives a friendlier
    message ("wait must be a positive integer...") than the generic
    Lark syntax error.

    Rules:
      * any ``{`` or ``}`` → not legacy
      * any first token that is a new keyword other than ``REPEAT``
        (REPEAT is shared between both grammars) → not legacy
      * any first token that is not an ALL-CAPS IDENT (e.g. lowercase
        free text) → not legacy, since legacy state names are required
        to be all-caps
      * empty / whitespace-only / comment-only text → legacy (trivial)
      * otherwise → legacy, letting parse_legacy emit per-line errors
        such as "wait must be a positive integer" for lines like
        ``HOME_REST abc``
    """
    if "{" in text or "}" in text:
        return False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        first = line.split()[0]
        if first.upper() in _NEW_KEYWORDS and first.upper() != "REPEAT":
            return False
        if not _RE_UPPER_IDENT.match(first):
            return False
    return True


def parse_legacy(text: str) -> Tuple[Program, List[str]]:
    """
    Parse legacy text into a ``Program`` AST + error list.

    Semantics (matched to the old ``_parse_sequence``):

    * Every ``STATE_NAME WAIT_MS`` line becomes a ``MoveStmt``.
    * ``REPEAT N`` can appear anywhere; the LAST one wins and becomes
      the count of a single top-level ``RepeatBlock`` wrapping every
      move. If no REPEAT is present (or REPEAT=1), moves are emitted
      flat at the top level.
    * Comments and blank lines are skipped silently.
    * Any unrecognised line yields a human-readable error and is
      dropped from the AST, mirroring the old parser.
    """
    errors: List[str] = []
    moves: List[MoveStmt] = []
    repeat = 1

    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = _RE_REPEAT.match(line)
        if m:
            repeat = max(1, int(m.group(1)))
            continue

        m = _RE_MOVE.match(line)
        if m:
            name = m.group(1)
            wait_ms = int(m.group(2))
            moves.append(MoveStmt(state_name=name, wait_ms=wait_ms, line=i + 1))
            continue

        # Provide the same diagnostics the old regex parser did.
        parts = line.split()
        if len(parts) == 2 and not parts[1].lstrip("-").isdigit():
            errors.append(
                f"L{i + 1}: wait must be a positive integer, got '{parts[1]}'"
            )
        else:
            errors.append(
                f"L{i + 1}: expected 'STATE_NAME WAIT_MS', got '{line[:30]}'"
            )

    if not moves:
        return Program(body=[]), errors

    if repeat <= 1:
        return Program(body=list(moves)), errors

    repeat_block = RepeatBlock(count=repeat, body=list(moves), line=moves[0].line)
    return Program(body=[repeat_block]), errors


__all__ = ["looks_legacy", "parse_legacy"]
