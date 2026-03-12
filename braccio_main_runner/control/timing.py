"""Timing instrumentation for end-to-end chain."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TimingSample:
    teensy_mcu_ms: int = -1
    host_rx_ns: int = 0
    model_ns: int = 0
    qp_ns: int = 0
    cmd_tx_ns: int = 0
    ack_rx_ns: int = 0
    total_sensor_to_cmd_ms: float = 0.0


class TimingTracer:
    def __init__(self):
        self.last = TimingSample()

    @staticmethod
    def now_ns() -> int:
        return time.perf_counter_ns()

    def begin_from_frame(self, teensy_mcu_ms: int) -> None:
        self.last = TimingSample(teensy_mcu_ms=teensy_mcu_ms, host_rx_ns=self.now_ns())

    def mark_model_done(self) -> None:
        self.last.model_ns = self.now_ns()

    def mark_qp_done(self) -> None:
        self.last.qp_ns = self.now_ns()

    def mark_cmd_tx(self) -> None:
        self.last.cmd_tx_ns = self.now_ns()
        if self.last.host_rx_ns:
            self.last.total_sensor_to_cmd_ms = (self.last.cmd_tx_ns - self.last.host_rx_ns) / 1e6

    def mark_ack_rx(self) -> None:
        self.last.ack_rx_ns = self.now_ns()

    def append_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(self.last), separators=(",", ":")) + "\n")
