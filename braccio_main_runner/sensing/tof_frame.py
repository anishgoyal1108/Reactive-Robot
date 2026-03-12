"""Host-side ToF frame contracts and parsers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ToFFrameV1:
    seq: int
    mcu_ms: int
    sensor_id: str
    mux_channel: int
    joint_id: str
    status: int
    distances_mm: List[float]
    validity: List[int]


@dataclass(frozen=True)
class MegaCommandV1:
    seq: int
    host_ms: int
    mode: str
    speed_scale: float
    joints_deg: List[int]

    def encode(self) -> str:
        joints = ",".join(str(int(v)) for v in self.joints_deg)
        return (
            f"CMD,{self.seq},{self.host_ms},{self.mode},"
            f"{self.speed_scale:.3f},{joints}\n"
        )


@dataclass(frozen=True)
class MegaAckV1:
    seq: int
    status: str
    last_applied_ms: int


def parse_tof_line(line: str) -> Optional[ToFFrameV1]:
    """Parse strict v1 TF line or legacy FRAME line into ToFFrameV1."""
    text = line.strip()
    if not text:
        return None

    if text.startswith("TF,"):
        parts = text.split(",")
        if len(parts) < 7 + 64 + 64:
            return None
        try:
            seq = int(parts[1])
            mcu_ms = int(parts[2])
            sensor_id = parts[3]
            mux_channel = int(parts[4])
            joint_id = parts[5]
            status = int(parts[6], 0)
            d0 = [float(x) for x in parts[7:71]]
            v0 = [int(x) for x in parts[71:135]]
            if len(d0) != 64 or len(v0) != 64:
                return None
            return ToFFrameV1(
                seq=seq,
                mcu_ms=mcu_ms,
                sensor_id=sensor_id,
                mux_channel=mux_channel,
                joint_id=joint_id,
                status=status,
                distances_mm=d0,
                validity=v0,
            )
        except (TypeError, ValueError):
            return None

    if text.startswith("FRAME,"):
        parts = text.split(",")
        if len(parts) < 69:
            return None
        try:
            ch = int(parts[1])
            hz = int(parts[3])
            res = int(parts[4])
            if res != 64:
                return None
            data = [float(x) for x in parts[5:69]]
            return ToFFrameV1(
                seq=-1,
                mcu_ms=-1,
                sensor_id=f"legacy_ch{ch}",
                mux_channel=ch,
                joint_id="wrist_vert",
                status=hz,
                distances_mm=data,
                validity=[1] * 64,
            )
        except (TypeError, ValueError):
            return None

    return None


def parse_mega_ack(line: str) -> Optional[MegaAckV1]:
    text = line.strip()
    if not text.startswith("ACK,"):
        return None
    parts = text.split(",")
    if len(parts) < 4:
        return None
    try:
        return MegaAckV1(seq=int(parts[1]), status=parts[2], last_applied_ms=int(parts[3]))
    except (TypeError, ValueError):
        return None
