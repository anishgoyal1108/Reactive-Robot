"""protocol.py - Command builders and response parser (legacy + v1)."""

from __future__ import annotations

import time

from .constants import JOINT_TOKENS


def cmd_ping() -> str:
    return "PING\n"


def cmd_home() -> str:
    return "HOME\n"


def cmd_get_pos() -> str:
    return "GET POS\n"


def cmd_set_joint(joint_idx: int, deg: int) -> str:
    token = JOINT_TOKENS[joint_idx]
    return f"SET {token} {int(deg)}\n"


def cmd_set_all(positions: list) -> str:
    return "SET ALL " + " ".join(str(int(p)) for p in positions) + "\n"


def cmd_set_delta(delta: int) -> str:
    return f"SET DELTA {int(delta)}\n"


def cmd_set_ik_polar(
    theta_deg: float,
    r_mm: float,
    z_mm: float,
    wrist_offset_deg: float,
    wrist_rot_deg: int,
    gripper_deg: int,
) -> str:
    """Delegate IK solve to MCU using a polar target."""
    return (
        "SET IKP "
        f"{float(theta_deg):.3f} {float(r_mm):.3f} {float(z_mm):.3f} "
        f"{float(wrist_offset_deg):.3f} {int(wrist_rot_deg)} {int(gripper_deg)}\n"
    )


def cmd_v1(seq: int, mode: str, speed_scale: float, positions: list[int]) -> str:
    host_ms = int(time.time() * 1000)
    vals = ",".join(str(int(v)) for v in positions)
    return f"CMD,{seq},{host_ms},{mode},{float(speed_scale):.3f},{vals}\n"


def parse_response(line: str) -> dict:
    line = line.strip()

    if line == "PONG":
        return {"type": "pong", "raw": line}
    if line == "OK HOME":
        return {"type": "ok_home", "raw": line}
    if line == "READY":
        return {"type": "ready", "raw": line}

    if line.startswith("ACK,"):
        parts = line.split(",")
        if len(parts) >= 4:
            try:
                return {
                    "type": "ack",
                    "seq": int(parts[1]),
                    "status": parts[2],
                    "last_applied_ms": int(parts[3]),
                    "raw": line,
                }
            except ValueError:
                pass

    if line.startswith("STAT,"):
        return {"type": "stat", "raw": line}

    if line.startswith("OK ALL="):
        try:
            vals = [int(x) for x in line[7:].split(",")]
            return {"type": "ok_all", "positions": vals, "raw": line}
        except ValueError:
            return {"type": "unknown", "raw": line}

    if line.startswith("OK IK="):
        try:
            vals = [int(x) for x in line[6:].split(",")]
            return {"type": "ok_ik", "positions": vals, "raw": line}
        except ValueError:
            return {"type": "unknown", "raw": line}

    if line.startswith("OK J="):
        try:
            parts = line[5:].split(",")
            return {
                "type": "ok_joint",
                "joint": int(parts[0]),
                "deg": int(parts[1]),
                "raw": line,
            }
        except (ValueError, IndexError):
            return {"type": "unknown", "raw": line}

    if line.startswith("OK DELTA="):
        try:
            return {"type": "ok_delta", "delta": int(line[9:]), "raw": line}
        except ValueError:
            return {"type": "unknown", "raw": line}

    if line.startswith("POS="):
        try:
            vals = [int(x) for x in line[4:].split(",")]
            return {"type": "pos", "positions": vals, "raw": line}
        except ValueError:
            return {"type": "unknown", "raw": line}

    if line.startswith("ERR"):
        return {
            "type": "error",
            "message": line[4:].strip() if len(line) > 4 else "",
            "raw": line,
        }

    return {"type": "unknown", "raw": line}
