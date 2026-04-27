"""ToF stream manager: freshness, health, replay logging."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, Optional

from sensing.tof_frame import ToFFrameV1


class ToFStreamManager:
    def __init__(self, freshness_s: float = 0.4):
        self._lock = threading.RLock()
        self._latest: Dict[int, ToFFrameV1] = {}
        self._last_rx_s: Dict[int, float] = {}
        self._history: Deque[ToFFrameV1] = deque(maxlen=2048)
        self._freshness_s = freshness_s

    def ingest(self, frame: ToFFrameV1) -> None:
        now = time.time()
        with self._lock:
            self._latest[frame.mux_channel] = frame
            self._last_rx_s[frame.mux_channel] = now
            self._history.append(frame)

    def latest_by_channel(self, ch: int) -> Optional[ToFFrameV1]:
        with self._lock:
            return self._latest.get(ch)

    def all_latest(self) -> Dict[int, ToFFrameV1]:
        with self._lock:
            return dict(self._latest)

    def is_fresh(self, ch: int) -> bool:
        with self._lock:
            t = self._last_rx_s.get(ch)
            return t is not None and (time.time() - t) <= self._freshness_s

    def fresh_channels(self) -> Dict[int, bool]:
        with self._lock:
            now = time.time()
            return {ch: (now - ts) <= self._freshness_s for ch, ts in self._last_rx_s.items()}

    def log_jsonl(self, path: str | Path, frames: Optional[list[ToFFrameV1]] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = frames if frames is not None else list(self._history)
        with p.open("a", encoding="utf-8") as f:
            for fr in data:
                f.write(json.dumps(asdict(fr), separators=(",", ":")) + "\n")
