import time

from braccio_ctrl.tof_sensor import ToFBridge, ToFState
from sensing.tof_frame import MegaCommandV1, parse_mega_ack, parse_tof_line


def test_parse_tf_v1():
    d = ",".join(["100"] * 64)
    v = ",".join(["1"] * 64)
    line = f"TF,5,1234,S0,0,wrist_vert,1,{d},{v}"
    fr = parse_tof_line(line)
    assert fr is not None
    assert fr.seq == 5
    assert fr.mux_channel == 0
    assert len(fr.distances_mm) == 64
    assert len(fr.validity) == 64


def test_parse_legacy_frame():
    d = ",".join(["200"] * 64)
    line = f"FRAME,1,1,15,64,{d}"
    fr = parse_tof_line(line)
    assert fr is not None
    assert fr.mux_channel == 1
    assert fr.seq == -1


def test_mega_command_encode_and_ack_parse():
    cmd = MegaCommandV1(
        seq=7,
        host_ms=int(time.time() * 1000),
        mode="nominal_tracking",
        speed_scale=1.0,
        joints_deg=[1, 2, 3, 4, 5, 6],
    )
    s = cmd.encode()
    assert s.startswith("CMD,7,")
    ack = parse_mega_ack("ACK,7,OK,4321")
    assert ack is not None
    assert ack.seq == 7
    assert ack.status == "OK"


def test_parse_imu_line_updates_state():
    st = ToFState(num_channels=2)
    br = ToFBridge(st)
    br._parse_imu("IMU,9,1234,0.100,-0.200,0.980,1.5,2.5,-3.5,27.125,0")

    snap = st.snapshot()["imu"]
    assert snap["online"]
    assert snap["seq"] == 9
    assert snap["mcu_ms"] == 1234
    assert abs(snap["ax_g"] - 0.1) < 1e-6
    assert abs(snap["ay_g"] + 0.2) < 1e-6
    assert abs(snap["az_g"] - 0.98) < 1e-6
    assert snap["status"] == 0
