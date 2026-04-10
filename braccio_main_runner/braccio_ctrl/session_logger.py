"""
session_logger.py — JSONL telemetry logger for offline review.

Writes one JSON object per line at a fixed rate (default 10 Hz). Each line
captures everything needed to reconstruct what the robot saw and did at that
instant:

    {
      "t":          1712687234.512,     # wall-clock seconds since epoch
      "t_rel":      3.214,              # seconds since log file opened
      "joints":     [b, s, e, wv, wr, g],
      "theta":      deg,                # polar base angle
      "r":          mm,                 # reach
      "z":          mm,                 # height
      "tof": {
          "grids":        [[8x8], [8x8], [8x8], [8x8]],  # mm, null = nan
          "active":       [1,1,1,1],
          "hz":           [15, 15, 15, 15],
          "thresholds":   [250, 250, 50, 50],
          "ch_min_mm":    [123.4, 200.1, null, null],    # per-channel minimum
      },
      "ir": {
          "bits":   0-3,
          "label":  "CLEAR|FAR|CLOSE|DANGER",
          "action": ""|"advisory"|"replan"|"back_away",
      },
      "obstacle": {
          "response": "clear|replan|back_away",
          "source":   "tof_chN|ir",
          "dist_mm":  float,
      },
      "sweep": {
          "running":      bool,
          "state":        "idle|sweeping|replanning|blocked",
          "direction":    +1|-1,
          "current_theta": float,
          "sweep_r":      float,
          "sweep_z":      float,
          "bypass_theta": float|null,
      },
      "arm_io": {
          "last_cmd":   "SET ALL ...",
          "last_resp":  "OK ALL=...",
          "last_error": "",
      },
      "event": null | {"type": "action", "action": "sweep_toggle"}
    }

Events (action dispatches, obstacle triggers, sweep transitions, etc.) are
buffered and flushed inline with the next periodic sample so the reviewer
sees cause-and-effect in a single pass over the file.

Usage
-----
    logger = SessionLogger()
    logger.start()                       # opens file, begins sampling
    ...
    logger.log_event("action", {"action": "sweep_toggle"})
    logger.set_sources(
        arm_state=self._state,
        tof_state=self._tof_state,
        sweeper=self._sweeper,
    )
    ...
    logger.stop()                        # flushes + closes

The logger runs on its own daemon thread so sampling does not block the
main loop.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Optional

import numpy as np

from .constants import SESSION_LOG_DIR, SESSION_LOG_HZ


def _grid_to_list(g: np.ndarray) -> list:
    """Convert an 8x8 numpy grid to a nested list, NaN → None (JSON-safe)."""
    if g is None:
        return []
    out = []
    for row in g.tolist():
        out.append([None if (isinstance(v, float) and math.isnan(v)) else v
                    for v in row])
    return out


def _ch_min(g: np.ndarray) -> Optional[float]:
    """Minimum finite value in grid, or None if all NaN."""
    if g is None:
        return None
    try:
        if np.isnan(g).all():
            return None
        return float(np.nanmin(g))
    except Exception:
        return None


class SessionLogger:
    """
    JSONL telemetry sink. Thread-safe: `log_event` can be called from any
    thread; the sampler loop runs on its own daemon thread.
    """

    def __init__(
        self,
        hz: float = SESSION_LOG_HZ,
        out_dir: str = SESSION_LOG_DIR,
    ) -> None:
        self._hz = max(1.0, float(hz))
        self._period = 1.0 / self._hz
        self._out_dir = out_dir

        self._file = None  # type: Optional[Any]
        self._path: str = ""
        self._t0: float = 0.0

        self._lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._pending_events: list = []

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Data sources (set via set_sources)
        self._arm_state = None
        self._tof_state = None
        self._sweeper = None

    # ── Configuration ─────────────────────────────────────────────────────

    def set_sources(self, arm_state, tof_state, sweeper) -> None:
        """Attach the shared-state objects the logger samples from."""
        self._arm_state = arm_state
        self._tof_state = tof_state
        self._sweeper = sweeper

    @property
    def path(self) -> str:
        return self._path

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> str:
        """Open the JSONL file and launch the sampler thread."""
        if self.is_running:
            return self._path

        os.makedirs(self._out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(self._out_dir, f"session_{ts}.jsonl")

        self._file = open(self._path, "w", buffering=1)  # line-buffered
        self._t0 = time.time()

        # Write a header line so the reviewer knows what the file is
        header = {
            "type": "header",
            "t": self._t0,
            "hz": self._hz,
            "format": "jsonl/1",
            "description": (
                "Reactive-Robot session telemetry. Each non-header line is a "
                "sample dict; lines with 'event' set are cause-and-effect markers."
            ),
        }
        self._write_obj(header)

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sampler_loop,
            daemon=True,
            name="session-logger",
        )
        self._thread.start()
        return self._path

    def stop(self) -> None:
        """Stop the sampler and close the file."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._thread = None

        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                    self._file.close()
                except Exception:
                    pass
                self._file = None

    # ── Event hook (thread-safe) ──────────────────────────────────────────

    def log_event(self, kind: str, payload: Optional[dict] = None) -> None:
        """
        Record a one-off event. It is attached to the next periodic sample
        so every event is timestamped alongside the full state context.
        """
        if not self.is_running:
            return
        ev = {
            "t": time.time(),
            "kind": kind,
            "data": payload or {},
        }
        with self._event_lock:
            self._pending_events.append(ev)

    # ── Sampler loop ──────────────────────────────────────────────────────

    def _sampler_loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self._emit_sample()
            except Exception:
                # Never let the logger crash the main loop
                pass

            next_tick += self._period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                if self._stop.wait(sleep_s):
                    break
            else:
                # Falling behind — resync
                next_tick = time.monotonic()

    def _emit_sample(self) -> None:
        arm = self._arm_state
        tof = self._tof_state
        sweeper = self._sweeper
        if arm is None or tof is None:
            return

        now = time.time()
        arm_snap = arm.snapshot()
        tof_snap = tof.snapshot()
        sweep_snap = sweeper.get_status() if sweeper is not None else {}

        # Drain pending events
        with self._event_lock:
            events = self._pending_events
            self._pending_events = []

        grids = tof_snap.get("grids", [])
        thresholds = tof_snap.get(
            "tof_thresholds_mm", tof_snap.get("tof_threshold_mm", [])
        )
        if not isinstance(thresholds, list):
            thresholds = [thresholds] * len(grids)

        sample = {
            "t": round(now, 3),
            "t_rel": round(now - self._t0, 3),
            "joints": arm_snap.get("joints", []),
            "theta": round(float(arm_snap.get("theta", 0.0)), 2),
            "r": round(float(arm_snap.get("r", 0.0)), 2),
            "z": round(float(arm_snap.get("z", 0.0)), 2),
            "tof": {
                "grids": [_grid_to_list(g) for g in grids],
                "active": list(tof_snap.get("active", [])),
                "hz": list(tof_snap.get("hz", [])),
                "thresholds": list(thresholds),
                "ch_min_mm": [_ch_min(g) for g in grids],
            },
            "ir": {
                "bits": int(tof_snap.get("ir_bits", 0)),
                "label": tof_snap.get("ir_label", ""),
                "action": tof_snap.get("ir_action", ""),
            },
            "obstacle": {
                "response": arm_snap.get("obstacle_response", "clear"),
                "source": arm_snap.get("obstacle_source", ""),
                "dist_mm": float(arm_snap.get("obstacle_dist_mm", -1.0)),
            },
            "sweep": {
                "running": bool(sweep_snap.get("running", False)),
                "state": sweep_snap.get("sweep_state", ""),
                "direction": sweep_snap.get("direction", 0),
                "current_theta": sweep_snap.get("current_theta", None),
                "sweep_r": sweep_snap.get("sweep_r", None),
                "sweep_z": sweep_snap.get("sweep_z", None),
                "bypass_theta": sweep_snap.get("bypass_theta", None),
            },
            "arm_io": {
                "last_cmd": arm_snap.get("last_cmd", ""),
                "last_resp": arm_snap.get("last_resp", ""),
                "last_error": arm_snap.get("last_error", ""),
            },
            "events": events,
        }

        self._write_obj(sample)

    def _write_obj(self, obj: dict) -> None:
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(json.dumps(obj, default=str) + "\n")
            except Exception:
                pass
