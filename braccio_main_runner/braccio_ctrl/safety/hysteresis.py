"""
hysteresis.py — Filtering primitives for the obstacle-avoidance stack.

Three signal-conditioning primitives, each kept deliberately small so they can
be unit-tested in isolation and composed freely:

    SchmittTrigger(engage, disengage)
        Dual-threshold latch. ``update(reading)`` latches HIGH when the reading
        crosses ``engage`` and stays HIGH until the reading exceeds
        ``disengage``. Eliminates binary flapping around a single threshold —
        see research §7 (VL53L5CX ±5-15 mm noise band).

    EMA(alpha)
        Exponential moving average: ``smoothed = α·new + (1-α)·previous``.
        With α = 0.2 (DeXtreme default) it kills servo jerk from discrete
        command steps without introducing noticeable lag.

    RateClamp(max_delta_per_tick)
        Per-axis absolute-delta limiter applied *after* EMA as a safety
        backstop against pathological input spikes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SchmittTrigger:
    """Dual-threshold latching trigger.

    Parameters
    ----------
    engage_mm : float
        Reading below this value latches the trigger HIGH.
    disengage_mm : float
        Reading above this value releases the trigger back to LOW.
        Must satisfy ``disengage_mm > engage_mm``; the deadband between
        them should be comfortably larger than the sensor's peak-to-peak
        noise (research §7 recommends ~6× noise; for VL53L5CX with ±15 mm
        noise that's a 100 mm deadband).
    """

    def __init__(self, engage_mm: float, disengage_mm: float):
        if disengage_mm <= engage_mm:
            raise ValueError(
                f"disengage_mm ({disengage_mm}) must exceed engage_mm ({engage_mm})"
            )
        self._engage = float(engage_mm)
        self._disengage = float(disengage_mm)
        self._latched = False

    def update(self, reading_mm: float) -> bool:
        """Update with a new reading and return the current latch state."""
        r = float(reading_mm)
        if self._latched:
            if r > self._disengage:
                self._latched = False
        else:
            if r < self._engage:
                self._latched = True
        return self._latched

    @property
    def latched(self) -> bool:
        return self._latched

    def reset(self) -> None:
        self._latched = False


class EMA:
    """Exponential moving average filter (vectorised over NumPy arrays).

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1]. Lower → more smoothing, more lag.
        Research §7: α = 0.1–0.2 matches DeXtreme's real-hardware default.
    initial : np.ndarray | None
        Optional initial value. If ``None``, the first ``update()`` call
        bootstraps the filter with its input.
    """

    def __init__(self, alpha: float = 0.2, initial: Optional[np.ndarray] = None):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self._alpha = float(alpha)
        self._state: Optional[np.ndarray] = (
            None if initial is None else np.asarray(initial, dtype=np.float64).copy()
        )

    def update(self, new: np.ndarray) -> np.ndarray:
        x = np.asarray(new, dtype=np.float64)
        if self._state is None or self._state.shape != x.shape:
            self._state = x.copy()
        else:
            self._state = self._alpha * x + (1.0 - self._alpha) * self._state
        return self._state.copy()

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.copy()

    def reset(self, initial: Optional[np.ndarray] = None) -> None:
        self._state = (
            None if initial is None
            else np.asarray(initial, dtype=np.float64).copy()
        )


class RateClamp:
    """Per-axis absolute-delta limiter (safety backstop, applied post-EMA).

    Parameters
    ----------
    max_delta_per_tick : float | np.ndarray
        Scalar or per-axis maximum absolute change per ``apply()`` call.
    """

    def __init__(self, max_delta_per_tick: float = 5.0):
        self._max_delta = np.asarray(max_delta_per_tick, dtype=np.float64)

    def apply(self, previous: np.ndarray, target: np.ndarray) -> np.ndarray:
        prev = np.asarray(previous, dtype=np.float64)
        tgt = np.asarray(target, dtype=np.float64)
        delta = tgt - prev
        clipped = np.clip(delta, -self._max_delta, self._max_delta)
        return prev + clipped
