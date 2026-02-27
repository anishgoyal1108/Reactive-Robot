"""
protocol.py — Command builders and response parser.

Pure string manipulation — no I/O occurs here.
All command functions return strings ending with '\\n' (ready to send).
"""

from .constants import JOINT_TOKENS


# ── Command builders ──────────────────────────────────────────────────────

def cmd_ping() -> str:
    return "PING\n"


def cmd_home() -> str:
    return "HOME\n"


def cmd_get_pos() -> str:
    return "GET POS\n"


def cmd_set_joint(joint_idx: int, deg: int) -> str:
    """Single-joint absolute move. E.g. 'SET B 45\\n'."""
    token = JOINT_TOKENS[joint_idx]
    return f"SET {token} {int(deg)}\n"


def cmd_set_all(positions: list) -> str:
    """All-joints move. E.g. 'SET ALL 90 90 90 90 90 73\\n'."""
    return "SET ALL " + " ".join(str(int(p)) for p in positions) + "\n"


def cmd_set_delta(delta: int) -> str:
    """Set slew rate for all joints. E.g. 'SET DELTA 2\\n'."""
    return f"SET DELTA {int(delta)}\n"


# ── Response parser ───────────────────────────────────────────────────────

def parse_response(line: str) -> dict:
    """
    Parse one Arduino response line into a structured dict.

    Return dict always has a 'type' key. Known types:
        'pong'       — PING response
        'ok_home'    — HOME response
        'ok_joint'   — single joint SET response  → 'joint': int, 'deg': int
        'ok_all'     — SET ALL response            → 'positions': list[int]
        'ok_delta'   — SET DELTA response          → 'delta': int
        'pos'        — GET POS response            → 'positions': list[int]
        'ready'      — READY startup message
        'error'      — ERR ... response            → 'message': str
        'unknown'    — anything else               → 'raw': str
    """
    line = line.strip()

    if line == "PONG":
        return {'type': 'pong'}

    if line == "OK HOME":
        return {'type': 'ok_home'}

    if line == "READY":
        return {'type': 'ready'}

    if line.startswith("OK ALL="):
        try:
            vals = [int(x) for x in line[7:].split(',')]
            return {'type': 'ok_all', 'positions': vals}
        except ValueError:
            return {'type': 'unknown', 'raw': line}

    if line.startswith("OK J="):
        try:
            parts = line[5:].split(',')
            return {'type': 'ok_joint', 'joint': int(parts[0]), 'deg': int(parts[1])}
        except (ValueError, IndexError):
            return {'type': 'unknown', 'raw': line}

    if line.startswith("OK DELTA="):
        try:
            return {'type': 'ok_delta', 'delta': int(line[9:])}
        except ValueError:
            return {'type': 'unknown', 'raw': line}

    if line.startswith("POS="):
        try:
            vals = [int(x) for x in line[4:].split(',')]
            return {'type': 'pos', 'positions': vals}
        except ValueError:
            return {'type': 'unknown', 'raw': line}

    if line.startswith("ERR"):
        return {'type': 'error', 'message': line[4:].strip() if len(line) > 4 else ''}

    return {'type': 'unknown', 'raw': line}
