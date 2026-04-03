"""
serial_bridge.py — Serial I/O with a daemon reader thread.

SerialBridge owns the serial.Serial object.  A daemon reader thread
continuously reads lines from the Arduino and places parsed response
dicts onto a Queue for the main thread to drain.

All writes go through a Lock so the control loop and any other caller
can safely call send_cmd() concurrently.
"""

import queue
import threading
import time

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False

from .protocol import parse_response, cmd_ping
from .constants import BAUD_RATE, ARM_DATA_PORT, TOF_DATA_PORT


def ensure_serial_device_path(port: str, what: str) -> None:
    """
    Reject mistaken use of UDP plotter ports (9870/9871) where a /dev path
    is required.  Pyserial would otherwise try to open a bogus path and
    report errno 2.
    """
    if not port or not isinstance(port, str):
        return
    p = port.strip()
    if not p.isdigit():
        return
    n = int(p)
    hint = ''
    if n in (ARM_DATA_PORT, TOF_DATA_PORT):
        hint = (
            f' {n} is the UDP port for arm_plotter_app ({ARM_DATA_PORT}) or '
            f'tof_plotter_app ({TOF_DATA_PORT}); serial devices look like '
            f'/dev/ttyACM0 or COM3.'
        )
    raise ValueError(
        f'{what} must be a USB serial device path, not the number {p!r}.{hint}'
    )


class SerialBridge:
    def __init__(self, port: str, baud: int = BAUD_RATE):
        self._port  = port
        self._baud  = baud
        self._ser   = None
        self._write_lock   = threading.Lock()
        self._stop_event   = threading.Event()
        self._reader_thread = None
        self.response_queue: queue.Queue = queue.Queue(maxsize=128)
        self.connect_error: str = ""   # set on connect() failure; empty on success

    # ── Connection ────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Open the serial port and start the reader thread.

        Returns True on success.  The Arduino resets on DTR assertion, so
        we wait 2 s after opening and then drain any startup messages.
        """
        if not _SERIAL_AVAILABLE:
            return False
        try:
            ensure_serial_device_path(self._port, "Braccio serial port")
        except ValueError as exc:
            self.connect_error = str(exc)
            return False
        try:
            self._ser = serial.Serial(
                self._port,
                self._baud,
                timeout=0.1,
            )
            # Wait for Arduino boot (DTR reset)
            time.sleep(2.0)
            self._ser.reset_input_buffer()

            self._stop_event.clear()
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                daemon=True,
                name="braccio-serial-reader",
            )
            self._reader_thread.start()
            return True
        except Exception as exc:
            self.connect_error = str(exc)
            return False

    def ping(self, timeout: float = 1.5) -> bool:
        """
        Send PING and wait up to `timeout` seconds for PONG.
        Used after connect() to verify the Arduino is alive and ready.
        """
        if not self.is_open():
            return False
        # Drain stale responses
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
                if resp['type'] == 'pong':
                    return True
            except queue.Empty:
                pass
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

    # ── Send ──────────────────────────────────────────────────────────────

    def send_cmd(self, cmd: str) -> None:
        """
        Send a command string to the Arduino.  Thread-safe.
        cmd must end with '\\n' (all protocol.py builders do this).
        """
        if not self.is_open():
            return
        with self._write_lock:
            try:
                self._ser.write(cmd.encode('ascii'))
                self._ser.flush()
            except Exception:
                pass

    # ── Reader thread ─────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Daemon thread: read lines, parse, enqueue."""
        while not self._stop_event.is_set():
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode('ascii', errors='replace').strip()
                if not line:
                    continue
                parsed = parse_response(line)
                try:
                    self.response_queue.put_nowait(parsed)
                except queue.Full:
                    # Drop oldest to make room
                    try:
                        self.response_queue.get_nowait()
                        self.response_queue.put_nowait(parsed)
                    except queue.Empty:
                        pass
            except Exception:
                if self._stop_event.is_set():
                    break
                time.sleep(0.05)   # brief back-off on transient errors

    # ── Port discovery ────────────────────────────────────────────────────

    @staticmethod
    def list_ports() -> list:
        """Return a list of (device, description) tuples for all serial ports."""
        if not _SERIAL_AVAILABLE:
            return []
        return [(p.device, p.description)
                for p in serial.tools.list_ports.comports()]
