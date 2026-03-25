"""
data_publisher.py — Fire-and-forget UDP publisher for arm and ToF data.

Sends JSON-encoded snapshots to localhost on two separate ports so that
standalone plotter scripts can consume them without any threading in the
main braccio_ctrl module.

If no listener is running the datagrams are silently dropped by the OS.
"""

import json
import socket

from .constants import ARM_DATA_PORT, TOF_DATA_PORT


class DataPublisher:
    """
    Lightweight UDP publisher.  One socket, two destination ports.

    Parameters
    ----------
    arm_port : int  — destination port for joint-angle packets
    tof_port : int  — destination port for ToF/IR/IMU packets
    """

    def __init__(self, arm_port: int = ARM_DATA_PORT,
                 tof_port: int = TOF_DATA_PORT):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._arm_addr = ('127.0.0.1', arm_port)
        self._tof_addr = ('127.0.0.1', tof_port)

    def send_arm(self, state_snap: dict) -> None:
        """Publish current joint angles + IK state."""
        try:
            payload = {
                'joints': state_snap['joints'],
                'theta':  state_snap['theta'],
                'r':      state_snap['r'],
                'z':      state_snap['z'],
            }
            self._sock.sendto(json.dumps(payload).encode(), self._arm_addr)
        except Exception:
            pass

    def send_tof(self, tof_snap: dict) -> None:
        """Publish ToF grid + IR + obstacle state."""
        try:
            payload = {
                'grids':             [g.tolist() for g in tof_snap['grids']],
                'active':            tof_snap['active'],
                'frame_cnt':         tof_snap['frame_cnt'],
                'hz':                tof_snap['hz'],
                'ir_bits':           tof_snap['ir_bits'],
                'ir_label':          tof_snap['ir_label'],
                'obstacle_response': tof_snap['obstacle_response'],
                'obstacle_dist_mm':  tof_snap['obstacle_dist_mm'],
                'tof_threshold_mm':  tof_snap['tof_threshold_mm'],
                'num_channels':      tof_snap['num_channels'],
            }
            self._sock.sendto(json.dumps(payload).encode(), self._tof_addr)
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass
