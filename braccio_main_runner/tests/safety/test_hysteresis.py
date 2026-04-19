"""Unit tests for braccio_ctrl.safety.hysteresis."""

import numpy as np
import pytest

from braccio_ctrl.safety.hysteresis import EMA, RateClamp, SchmittTrigger


class TestSchmittTrigger:
    def test_rejects_inverted_thresholds(self):
        with pytest.raises(ValueError):
            SchmittTrigger(engage_mm=400.0, disengage_mm=300.0)
        with pytest.raises(ValueError):
            SchmittTrigger(engage_mm=250.0, disengage_mm=250.0)

    def test_engages_below_engage_threshold(self):
        t = SchmittTrigger(engage_mm=250.0, disengage_mm=350.0)
        assert t.update(500.0) is False
        assert t.update(260.0) is False  # above engage, stays low
        assert t.update(200.0) is True   # crossed engage
        assert t.latched is True

    def test_stays_latched_in_deadband(self):
        t = SchmittTrigger(engage_mm=250.0, disengage_mm=350.0)
        t.update(100.0)
        for reading in (200.0, 260.0, 300.0, 349.0):
            assert t.update(reading) is True, f"should stay latched at {reading}"

    def test_releases_above_disengage(self):
        t = SchmittTrigger(engage_mm=250.0, disengage_mm=350.0)
        t.update(100.0)
        assert t.update(400.0) is False
        assert t.update(300.0) is False  # between thresholds, stays low

    def test_no_flapping_at_engage_boundary(self):
        """The core point of hysteresis: flapping noise around one threshold
        must not cause the trigger to oscillate."""
        t = SchmittTrigger(engage_mm=250.0, disengage_mm=350.0)
        # noisy stream hovering around 250 mm with ±15 mm jitter
        readings = [245, 255, 248, 252, 245, 258, 247]
        results = [t.update(r) for r in readings]
        # After the first sub-engage reading (245), the trigger should latch
        # and stay latched for every subsequent sample (all below disengage).
        assert results == [True] * 7

    def test_reset(self):
        t = SchmittTrigger(250.0, 350.0)
        t.update(100.0)
        assert t.latched
        t.reset()
        assert not t.latched


class TestEMA:
    def test_rejects_bad_alpha(self):
        with pytest.raises(ValueError):
            EMA(alpha=0.0)
        with pytest.raises(ValueError):
            EMA(alpha=1.5)
        with pytest.raises(ValueError):
            EMA(alpha=-0.1)

    def test_bootstrap_on_first_update(self):
        e = EMA(alpha=0.2)
        out = e.update(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_alpha_1_is_passthrough(self):
        e = EMA(alpha=1.0)
        e.update(np.zeros(3))
        out = e.update(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_allclose(out, [5.0, 5.0, 5.0])

    def test_blends_toward_new_value(self):
        e = EMA(alpha=0.5)
        e.update(np.array([0.0]))
        out = e.update(np.array([10.0]))
        np.testing.assert_allclose(out, [5.0])
        out = e.update(np.array([10.0]))
        np.testing.assert_allclose(out, [7.5])

    def test_converges_to_steady_input(self):
        e = EMA(alpha=0.2)
        for _ in range(200):
            out = e.update(np.array([42.0]))
        assert abs(out[0] - 42.0) < 1e-6

    def test_reset(self):
        e = EMA(alpha=0.2, initial=np.zeros(3))
        e.update(np.ones(3) * 5.0)
        e.reset()
        assert e.value is None
        out = e.update(np.array([3.0]))
        np.testing.assert_allclose(out, [3.0])


class TestRateClamp:
    def test_limits_absolute_delta(self):
        c = RateClamp(max_delta_per_tick=5.0)
        prev = np.array([90.0, 90.0, 90.0])
        tgt = np.array([100.0, 80.0, 92.0])  # deltas: +10, -10, +2
        out = c.apply(prev, tgt)
        np.testing.assert_allclose(out, [95.0, 85.0, 92.0])

    def test_per_axis_limits(self):
        c = RateClamp(max_delta_per_tick=np.array([3.0, 5.0, 10.0]))
        prev = np.zeros(3)
        tgt = np.array([100.0, 100.0, 100.0])
        out = c.apply(prev, tgt)
        np.testing.assert_allclose(out, [3.0, 5.0, 10.0])

    def test_zero_delta_is_noop(self):
        c = RateClamp(max_delta_per_tick=5.0)
        prev = np.array([50.0, 50.0])
        out = c.apply(prev, prev)
        np.testing.assert_allclose(out, prev)
