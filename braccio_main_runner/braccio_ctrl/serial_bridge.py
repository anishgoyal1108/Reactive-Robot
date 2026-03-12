"""Serial I/O bridge with background reader thread."""

from __future__ import annotations

import queue
import threading
import time

try:
    import serial
    import serial.tools.list_ports

    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

from .constants import BAUD_RATE
from .protocol import cmd_ping, parse_response


class SerialBridge:
    def __init__(self, port: str, baud: int = BAUD_RATE):
        self._port = port
        self._baud = baud
        self._ser = None
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_thread = None
        self.response_queue: queue.Queue = queue.Queue(maxsize=128)
        self.connect_error: str = ""

    def connect(self) -> bool:
        if not _SERIAL_AVAILABLE:
            self.connect_error = "pyserial not available"
            return False
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=0.1)
            time.sleep(2.0)
            self._ser.reset_input_buffer()

            self._stop_event.clear()
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                daemon=True,
                name="braccio-serial-reader",
            )
            self._reader_thread.start()
            self.connect_error = ""
            return True
        except Exception as exc:
            self.connect_error = str(exc)
            return False

    def ping(self, timeout: float = 1.5) -> bool:
        """Send `PING` and accept any valid response as link-alive."""
        if not self.is_open():
            return False

        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break

        self.send_cmd(cmd_ping())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = self.response_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if resp.get("type") in {
                "pong",
                "ready",
                "stat",
                "pos",
                "ok_all",
                "ok_ik",
                "ok_home",
                "ok_joint",
                "ok_delta",
                "ack",
            }:
                return True
        return False

    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def close(self) -> None:
        self._stop_event.set()
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass

    def send_cmd(self, cmd: str) -> None:
        if not self.is_open():
            return
        with self._write_lock:
            try:
                self._ser.write(cmd.encode("ascii"))
                self._ser.flush()
            except Exception:
                pass

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                parsed = parse_response(line)
                try:
                    self.response_queue.put_nowait(parsed)
                except queue.Full:
                    try:
                        self.response_queue.get_nowait()
                        self.response_queue.put_nowait(parsed)
                    except queue.Empty:
                        pass
            except Exception:
                if self._stop_event.is_set():
                    break
                time.sleep(0.05)

    @staticmethod
    def list_ports() -> list[tuple[str, str]]:
        if not _SERIAL_AVAILABLE:
            return []
        return [(p.device, p.description) for p in serial.tools.list_ports.comports()]
