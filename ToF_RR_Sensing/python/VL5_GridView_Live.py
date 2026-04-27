import os
import sys
import time
import argparse
import numpy as np
from serial import Serial
from serial.tools import list_ports

"""Barebones console viewer for VL53L5CX frames from Teensy."""


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
    Accepts either:
      FRAME,ch,activeFlag,hz,res,d0,d1,...,d63
    or
      TF,seq,mcu_ms,sensor_id,mux_ch,joint_id,status,rows,cols,d...,v...

    Returns (ch:int, grid:np.ndarray, valid:np.ndarray) or None.
    """
    parts = line.strip().split(",")
    if not parts:
        return None

    try:
        if parts[0] == "FRAME":
            if len(parts) < 6:
                return None
            ch = int(parts[1])
            res = int(parts[4])
            side = int(round(float(res) ** 0.5))
            if ch not in (0, 1, 2, 3) or side * side != res:
                return None
            data = parts[5 : 5 + res]
            vals = np.array([int(float(x)) for x in data], dtype=np.int32).reshape((side, side))
            valid = np.ones((side, side), dtype=np.uint8)
            return ch, vals, valid

        if parts[0] == "TF":
            if len(parts) < 9:
                return None
            ch = int(parts[4])
            if ch not in (0, 1, 2, 3):
                return None
            rows = int(parts[7])
            cols = int(parts[8])
            count = rows * cols
            if rows <= 0 or cols <= 0 or len(parts) != 9 + count + count:
                return None
            vals = np.array([int(float(x)) for x in parts[9 : 9 + count]], dtype=np.int32).reshape((rows, cols))
            valid = np.array([int(x) for x in parts[9 + count : 9 + count + count]], dtype=np.uint8).reshape((rows, cols))
            return ch, vals, valid
    except Exception:
        return None

    return None

def fmt_grid(g, valid):
    """
    Returns list[str] representing a grid.

    """
    cell_w = 4
    lines = []
    rows, cols = g.shape
    header = "     " + " ".join(f"c{c:02d}"[-cell_w:] for c in range(cols))
    lines.append(header)
    for r in range(rows):
        cells = []
        for c in range(cols):
            if int(valid[r, c]) == 0 or int(g[r, c]) < 0:
                cells.append(f"{'--':>{cell_w}s}")
            else:
                cells.append(f"{int(g[r, c]):{cell_w}d}")
        row = f"r{r}: " + " ".join(cells)
        lines.append(row)
    return lines


def fmt_coord_map(rows, cols):
    lines = ["     " + " ".join(f"c{c:>2d}" for c in range(cols))]
    for r in range(rows):
        row = " ".join(f"{r},{c}" for c in range(cols))
        lines.append(f"r{r}: {row}")
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
    ap = argparse.ArgumentParser(description="Console grid viewer for VL53L5CX CH0..CH3.")
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
    grids = [np.full((4, 4), -1, dtype=np.int32) for _ in range(4)]
    valid = [np.zeros((4, 4), dtype=np.uint8) for _ in range(4)]
    last_rx = [0.0, 0.0, 0.0, 0.0]
    frames = [0, 0, 0, 0]

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
                ch, g, v = out
                grids[ch] = g
                valid[ch] = v
                last_rx[ch] = time.time()
                frames[ch] += 1

            now = time.time()
            if now >= next_draw:
                next_draw = now + period

                # render
                clear_console()
                header = "VL53L5CX Console Viewer (MUX)"
                print(header)
                print("-" * len(header))

                for ch in range(4):
                    age = now - last_rx[ch] if last_rx[ch] else 1e9
                    g = grids[ch]
                    v = valid[ch]
                    rows, cols = g.shape
                    print(f"\nCH{ch} | frames={frames[ch]} age={age:0.2f}s | grid={rows}x{cols}")
                    for line in fmt_grid(g, v):
                        print(line)
                    print("Coordinates:")
                    for line in fmt_coord_map(rows, cols):
                        print(line)

                print("\nOrientation reference:")
                print("  top of display = sensor row r0")
                print("  bottom of display = sensor last row")
                print("  left of display = sensor column c0")
                print("  right of display = sensor last column")
                print("  '--' means invalid / filtered zone")
                print("\n(CTRL+C to quit)")

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\nClosed.")

if __name__ == "__main__":
    main()
