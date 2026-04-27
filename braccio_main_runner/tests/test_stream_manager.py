import time

from sensing.tof_frame import ToFFrameV1
from sensing.tof_stream_manager import ToFStreamManager


def make_frame(ch: int, seq: int = 1):
    return ToFFrameV1(
        seq=seq,
        mcu_ms=10,
        sensor_id=f"S{ch}",
        mux_channel=ch,
        joint_id="wrist_vert",
        status=0,
        distances_mm=[100.0] * 64,
        validity=[1] * 64,
    )


def test_freshness_tracking():
    mgr = ToFStreamManager(freshness_s=0.1)
    mgr.ingest(make_frame(0))
    assert mgr.is_fresh(0)
    time.sleep(0.12)
    assert not mgr.is_fresh(0)
