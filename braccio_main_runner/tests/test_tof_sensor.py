"""Tests for tof_sensor frame parsing and the masked-status promotion."""

from __future__ import annotations

import numpy as np

from braccio_ctrl.constants import (
    TOF_MASKED_SYNTHETIC_MM,
    TOF_CLOSE_STATUS_CODES,
    TOF_MUX_TO_CHANNEL,
)
from braccio_ctrl.imu_state import IMUState
from braccio_ctrl.tof_sensor import ToFBridge, ToFState


def _make_bridge() -> tuple[ToFState, ToFBridge]:
    st = ToFState()
    return st, ToFBridge(st, IMUState())


def _tf_line(ch: int, rows: int, cols: int,
             dists: list[int], valids: list[int],
             statuses: list[int] | None = None) -> str:
    parts = [
        "TF", "1", "100", f"S{ch}", str(ch), "base", "0",
        str(rows), str(cols),
        *[str(d) for d in dists],
        *[str(v) for v in valids],
    ]
    if statuses is not None:
        parts.extend(str(s) for s in statuses)
    return ",".join(parts)


def test_fully_masked_frame_injects_synthetic_close_obstacle():
    """When every zone reports invalid (the display's 'masked' state),
    a single close-return is injected at the grid centre so the safety
    stack sees an obstacle instead of a dropped frame."""
    st, bridge = _make_bridge()
    zones = 16
    mux = 0                                     # LEFT sensor
    sw = TOF_MUX_TO_CHANNEL[mux]                # software CH2
    line = _tf_line(mux, 4, 4,
                    dists=[0] * zones,          # hand flush → firmware sees 0 mm
                    valids=[0] * zones,          # every cell dropped as invalid
                    statuses=[2] * zones)        # noise code — not 5/9
    bridge._parse_tf(line)

    # Exactly one cell should have been promoted to the synthetic
    # distance; everything else stays NaN.
    assert st.diag_masked_cells[sw] == 1
    n_valid = int(np.sum(~np.isnan(st.grids[sw])))
    assert n_valid == 1
    assert float(np.nanmin(st.grids[sw])) == TOF_MASKED_SYNTHETIC_MM


def test_status_five_does_not_promote_normal_readings():
    """Regression: codes 5 and 9 are 'ranging OK' in the VL53L5CX
    datasheet. Promoting them to TOF_MASKED_SYNTHETIC_MM was the bug
    that turned every good reading into a fake 60 mm obstacle."""
    st, bridge = _make_bridge()
    mux = 2
    sw = TOF_MUX_TO_CHANNEL[mux]
    line = _tf_line(mux, 4, 4,
                    dists=[850] * 16,
                    valids=[1] * 16,
                    statuses=[5] * 16)          # all ranging-OK
    bridge._parse_tf(line)

    # Distances must pass through untouched — no "60 mm" anywhere.
    assert float(np.nanmin(st.grids[sw])) == 850.0
    assert float(np.nanmax(st.grids[sw])) == 850.0
    assert st.diag_masked_cells[sw] == 0


def test_normal_reading_passes_through_unchanged():
    """Good readings are not touched by the masked-promotion path."""
    st, bridge = _make_bridge()
    mux = 2                                      # TOP sensor on the arm
    sw = TOF_MUX_TO_CHANNEL[mux]                 # software CH0
    line = _tf_line(mux, 4, 4,
                    dists=[500] * 16,
                    valids=[1] * 16,
                    statuses=[0] * 16)
    bridge._parse_tf(line)

    assert float(np.nanmin(st.grids[sw])) == 500.0
    assert st.diag_masked_cells[sw] == 0


def test_backcompat_line_without_status_field():
    """Old firmware (no trailing s0..sN) parses identically to before."""
    st, bridge = _make_bridge()
    mux = 0
    sw = TOF_MUX_TO_CHANNEL[mux]
    line = _tf_line(mux, 4, 4,
                    dists=[800] * 16,
                    valids=[1] * 16,
                    statuses=None)
    bridge._parse_tf(line)

    assert float(np.nanmin(st.grids[sw])) == 800.0
    assert st.diag_masked_cells[sw] == 0
    assert st.statuses[sw] is None


def test_mixed_zones_partial_invalid_does_not_trigger_whole_frame_inject():
    """Partial invalidity (some valid cells remain) does not inject the
    synthetic close-return — only a *fully* masked frame does."""
    st, bridge = _make_bridge()
    mux = 3
    sw = TOF_MUX_TO_CHANNEL[mux]
    # Half the zones invalid (firmware dropped), the other half valid.
    dists = [0, 900] * 8
    valids = [0, 1] * 8
    line = _tf_line(mux, 4, 4, dists, valids)
    bridge._parse_tf(line)

    g = st.grids[sw]
    # The 8 valid cells keep their real distance; the 8 invalid cells
    # become NaN; nothing is promoted to 60 mm.
    finite_cells = g[~np.isnan(g)]
    assert finite_cells.size == 8
    assert np.all(finite_cells == 900.0)
    assert st.diag_masked_cells[sw] == 0


def test_mux_channel_0_maps_to_software_channel_2():
    """Physical MUX 0 is wired to the LEFT sensor; software CH2 is 'Left'.
    A frame labeled S0 from the firmware must land in grids[2], not grids[0]."""
    st, bridge = _make_bridge()
    line = _tf_line(0, 4, 4,               # mux_ch=0 (LEFT sensor)
                    dists=[500] * 16,
                    valids=[1] * 16)
    bridge._parse_tf(line)

    # The incoming mux_ch=0 must route to software CH2 (Left).
    assert float(np.nanmin(st.grids[2])) == 500.0
    assert np.isnan(st.grids[0]).all()     # software CH0 (Top) untouched


def test_mux_channel_2_maps_to_software_channel_0():
    """Physical MUX 2 is wired to the TOP sensor; software CH0 is 'Top'."""
    st, bridge = _make_bridge()
    line = _tf_line(2, 4, 4,               # mux_ch=2 (TOP sensor)
                    dists=[800] * 16,
                    valids=[1] * 16)
    bridge._parse_tf(line)

    assert float(np.nanmin(st.grids[0])) == 800.0
    assert np.isnan(st.grids[2]).all()


def test_mux_channels_1_and_3_pass_through_unchanged():
    """RIGHT (CH1) and BOTTOM (CH3) are already aligned; the remap is
    identity for those channels."""
    st, bridge = _make_bridge()
    for mux_ch in (1, 3):
        st.grids[mux_ch][:] = np.nan  # reset
        line = _tf_line(mux_ch, 4, 4,
                        dists=[300 + mux_ch * 100] * 16,
                        valids=[1] * 16)
        bridge._parse_tf(line)
        assert float(np.nanmin(st.grids[mux_ch])) == 300 + mux_ch * 100


def test_threshold_breach_produces_warn_not_replan():
    """Per-channel threshold breach must classify as WARN (advisory) and
    never as REPLAN. REPLAN is reserved for the BT's actual replan
    decisions; the legacy path emitting REPLAN on raw threshold breach
    was producing red 'REPLAN TRAJECTORY' banners for CH3 self-wiring
    reflections the BT correctly ignored."""
    from braccio_ctrl.tof_sensor import ObstacleResponse

    st, _ = _make_bridge()
    st.active[2] = 1
    st.grids[2] = np.full((8, 8), 30.0, dtype=np.float32)  # well below 100 mm
    st.update_obstacle_status()

    assert st.obstacle_response == ObstacleResponse.WARN
    assert st.obstacle_response != ObstacleResponse.REPLAN
    assert st.obstacle_source == "tof_ch2"
    assert st.obstacle_dist_mm == 30.0


def test_all_channels_clear_yields_clear_response():
    from braccio_ctrl.tof_sensor import ObstacleResponse

    st, _ = _make_bridge()
    for ch in range(4):
        st.active[ch] = 1
        st.grids[ch] = np.full((8, 8), 500.0, dtype=np.float32)
    st.update_obstacle_status()

    assert st.obstacle_response == ObstacleResponse.CLEAR


def test_ir_danger_still_wins_over_tof_warn():
    """BACK_AWAY priority is preserved — IR at DANGER/CLOSE still
    supersedes any ToF warning."""
    from braccio_ctrl.tof_sensor import ObstacleResponse

    st, _ = _make_bridge()
    st.ir_enabled = True
    st.ir_bits = 3   # DANGER
    st.active[0] = 1
    st.grids[0] = np.full((8, 8), 30.0, dtype=np.float32)
    st.update_obstacle_status()

    assert st.obstacle_response == ObstacleResponse.BACK_AWAY


def test_tof_close_status_codes_excludes_normal_valid_codes():
    """Codes 5 and 9 are 'ranging OK' per the VL53L5CX datasheet; they
    must never appear in TOF_CLOSE_STATUS_CODES or every normal reading
    becomes a fake 60 mm obstacle (the 2026-04-14 regression)."""
    assert isinstance(TOF_CLOSE_STATUS_CODES, tuple)
    assert 5 not in TOF_CLOSE_STATUS_CODES
    assert 9 not in TOF_CLOSE_STATUS_CODES
