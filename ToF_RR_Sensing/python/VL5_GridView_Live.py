import os
import sys
import time
import argparse
import numpy as np
from serial import Serial
from serial.tools import list_ports

"""

Barebones as possible console viewer for VL53L5CX 8x8 frames from Teensy.

"""


def pick_port(user_port):
    if user_port:
        return user_port
    env_port = os.environ.get("VL5_PORT")
    if env_port:
        return env_port
    ports = list(list_ports.comports())
    if not ports:
        return None
    return ports[0].device

def parse_frame(line):
    """
    Expected:
      FRAME,ch,activeFlag,hz,res,d0,d1,...,d63, defined by Teensy code in the .ino file.
    Returns (ch:int, grid:np.ndarray 8x8) or None.
    """
    if not line.startswith("FRAME,"):
        return None
    parts = line.strip().split(",")
    if len(parts) < 6:
        return None
    try:
        ch = int(parts[1])
        res = int(parts[4])
        if ch not in (0, 1) or res != 64:
            return None
        data = parts[5:5 + 64]
        if len(data) != 64:
            return None
        vals = np.array([int(x) for x in data], dtype=np.int32).reshape((8, 8))
        return ch, vals
    except Exception:
        return None

def fmt_grid(g):
    """
    Returns list[str] of 8 lines representing an 8x8 grid.

    """
    # fixed width per cell (4 chars) keeps columns aligned up to 9999 mm
    # if you expect >9999, bump width to 5.
    cell_w = 4
    lines = []
    for r in range(8):
        row = " ".join(f"{int(g[r, c]):{cell_w}d}" for c in range(8))
        lines.append(row)
    return lines

def clear_console():
    # ANSI clear works in modern terminals; fallback to newlines
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")

def main(): 
    """
    
    Making this was hell
    
    """
    ap = argparse.ArgumentParser(description="Console 8x8 grids for VL53L5CX CH0/CH1 (MUX only).")
    ap.add_argument("--port", default=None, help="Serial port (or set env VL5_PORT).")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--fps", type=float, default=10.0, help="Console refresh rate (Hz).")
    ap.add_argument("--timeout", type=float, default=0.1)
    args = ap.parse_args()

    port = pick_port(args.port)
    if not port:
        raise RuntimeError("No serial port found. Pass --port COM5 or set VL5_PORT.")

    ser = Serial(port, args.baud, timeout=args.timeout)
    print(f"Connected: {port} @ {args.baud}")
    print("Reading frames... (CTRL+C to quit)")
    time.sleep(0.3)

    # last grids + timestamps + frame counts
    grids = [np.full((8, 8), -1, dtype=np.int32), np.full((8, 8), -1, dtype=np.int32)]
    last_rx = [0.0, 0.0]
    frames = [0, 0]

    # target refresh timing
    period = 1.0 / max(1e-6, args.fps)
    next_draw = time.time()

    try:
        while True:
            # Drain serial quickly
            while ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                out = parse_frame(line)
                if out is None:
                    continue
                ch, g = out
                grids[ch] = g
                last_rx[ch] = time.time()
                frames[ch] += 1

            now = time.time()
            if now >= next_draw:
                next_draw = now + period

                # render
                clear_console()
                age0 = now - last_rx[0] if last_rx[0] else 1e9
                age1 = now - last_rx[1] if last_rx[1] else 1e9

                header = (
                    f"VL53L5CX Console Viewer (MUX) | "
                    f"CH0 frames={frames[0]} age={age0:0.2f}s | "
                    f"CH1 frames={frames[1]} age={age1:0.2f}s"
                )
                print(header)
                print("-" * len(header))

                left = fmt_grid(grids[0])
                right = fmt_grid(grids[1])

                print("CH0".ljust(4 + 8 * 5) + "    " + "CH1")
                for r in range(8):
                    print(left[r] + "    " + right[r])

                print("\n(CTRL+C to quit)")

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\nClosed.")

if __name__ == "__main__":
    main()