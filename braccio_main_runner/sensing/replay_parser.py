"""JSONL replay loader for Teensy TF frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from sensing.tof_frame import ToFFrameV1


def load_replay(path: str | Path) -> Iterator[ToFFrameV1]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield ToFFrameV1(
                seq=int(obj["seq"]),
                mcu_ms=int(obj["mcu_ms"]),
                sensor_id=str(obj["sensor_id"]),
                mux_channel=int(obj["mux_channel"]),
                joint_id=str(obj["joint_id"]),
                status=int(obj["status"]),
                distances_mm=[float(x) for x in obj["distances_mm"]],
                validity=[int(x) for x in obj["validity"]],
            )
