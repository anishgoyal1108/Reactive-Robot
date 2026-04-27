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
    rows: int = 8
    cols: int = 8


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
        if len(parts) < 9:
            return None
        try:
            seq = int(parts[1])
            mcu_ms = int(parts[2])
            sensor_id = parts[3]
            mux_channel = int(parts[4])
            joint_id = parts[5]
            status = int(parts[6], 0)
            rows = 8
            cols = 8
            data_start = 7

            # New Hyperion format:
            # TF,<seq>,<mcu_ms>,<sensor_id>,<mux_channel>,<joint_id>,<status>,<rows>,<cols>,<d...>,<v...>
            if len(parts) >= 11:
                try:
                    cand_rows = int(parts[7])
                    cand_cols = int(parts[8])
                    cand_count = int(cand_rows * cand_cols)
                    if (
                        1 <= cand_rows <= 16
                        and 1 <= cand_cols <= 16
                        and cand_count > 0
                        and len(parts) == 9 + (2 * cand_count)
                    ):
                        rows = cand_rows
                        cols = cand_cols
                        data_start = 9
                except (TypeError, ValueError):
                    rows = 8
                    cols = 8
                    data_start = 7

            count = int(rows * cols)
            if count <= 0:
                return None
            if len(parts) < data_start + (2 * count):
                return None

            d0 = [float(x) for x in parts[data_start : data_start + count]]
            v0 = [int(x) for x in parts[data_start + count : data_start + (2 * count)]]
            if len(d0) != count or len(v0) != count:
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
                rows=rows,
                cols=cols,
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
            side = int(round(float(res) ** 0.5))
            if side * side != res:
                return None
            data = [float(x) for x in parts[5 : 5 + res]]
            return ToFFrameV1(
                seq=-1,
                mcu_ms=-1,
                sensor_id=f"legacy_ch{ch}",
                mux_channel=ch,
                joint_id="wrist_vert",
                status=hz,
                distances_mm=data,
                validity=[1] * res,
                rows=side,
                cols=side,
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
