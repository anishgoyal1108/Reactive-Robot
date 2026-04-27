"""Async Teensy receiver with sequence/freshness tracking."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

from sensing.tof_frame import ToFFrameV1, parse_tof_line


@dataclass
class ReceiverStats:
    lines_total: int = 0
    parse_fail: int = 0
    frames_ok: int = 0
    last_rx_wall_s: float = 0.0


class TeensyReceiver:
    def __init__(self, serial_obj):
        self._ser = serial_obj
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames: "queue.Queue[ToFFrameV1]" = queue.Queue(maxsize=512)
        self.stats = ReceiverStats()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="teensy-rx-v1")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                self.stats.lines_total += 1
                frame = parse_tof_line(line)
                if frame is None:
                    self.stats.parse_fail += 1
                    continue

                self.stats.frames_ok += 1
                self.stats.last_rx_wall_s = time.time()
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    try:
                        self.frames.get_nowait()
                        self.frames.put_nowait(frame)
                    except queue.Empty:
                        pass
            except Exception:
                time.sleep(0.01)
