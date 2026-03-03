import os
import time
import argparse
import numpy as np
import serial
from serial.tools import list_ports
Serial = serial.Serial
list_comports = list_ports.comports
serial_module_path = serial.__file__
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
This is a very intensive code, just vibecoded it for the presentation in order to get neat figures.
Realistically, we would use GridView_Live for real-time obstacle avoidance.

"""

def upsample_bilinear(z8, N): # simple 2-pass linear interpolation (first horizontally, then vertically)
    h, w = z8.shape
    x_src = np.linspace(0, w - 1, w)
    y_src = np.linspace(0, h - 1, h)
    x_tgt = np.linspace(0, w - 1, N)
    y_tgt = np.linspace(0, h - 1, N)

    zx = np.vstack([np.interp(x_tgt, x_src, z8[r, :]) for r in range(h)])
    zN = np.vstack([np.interp(y_tgt, y_src, zx[:, c]) for c in range(N)]).T
    return zN


def parse_line(line): # Parses payload lines from Teensy. Returns None if unrecognized, otherwise ("MODE", mode) or ("FRAME", {ch, active, hz, res, data})
    line = line.strip() 
    if not line:
        return None

    if line.startswith("MODE,"):
        return ("MODE", line.split(",", 1)[1].strip())

    if line.startswith("FRAME,"):
        parts = line.split(",")
        if len(parts) < 6:
            return None
        try:
            ch = int(parts[1])
            active = int(parts[2])
            hz = int(parts[3])
            res = int(parts[4])
            data = np.array(parts[5:5 + res], dtype=np.float32)
            if data.size != res:
                return None
            return ("FRAME", {"ch": ch, "active": active, "hz": hz, "res": res, "data": data})
        except Exception:
            return None

    return None


def pick_port(user_port): # priority: user arg > env var > auto-detect first serial port
    if user_port:
        return user_port
    env_port = os.environ.get("VL5_PORT")
    if env_port:
        return env_port
    ports = list(list_comports())
    if not ports:
        return None
    return ports[0].device


def main(): 
    parser = argparse.ArgumentParser(description="VL53L5CX (TCA9548A) live heatmap + 3D surface viewer.")
    parser.add_argument("--port", default=None)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--upsample", type=int, default=60)   # 80 can be heavy with 3D
    parser.add_argument("--flat", type=float, default=0.0)
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--max_lines", type=int, default=400)
    parser.add_argument("--surface_every", type=int, default=10, help="Redraw 3D surface every N frames")
    args = parser.parse_args()

    port = pick_port(args.port)
    if not port:
        raise RuntimeError("No serial ports detected. Set VL5_PORT or pass --port COM5.")

    print("pyserial resolved from: {}".format(serial_module_path))
    print("Opening serial {} @ {} ...".format(port, args.baud))
    ser = Serial(port, args.baud, timeout=0.05)

    UPSAMPLE_N = int(args.upsample)
    FLAT_Z_MM = float(args.flat)
    MAX_LINES = int(args.max_lines)
    SURFACE_EVERY = max(1, int(args.surface_every))

    # ---- Send commands to Teensy ----
    last_mode = ["MUX"]  # mutable

    def send_mode(cmd):
        cmd = cmd.strip().upper()
        try:
            ser.write((cmd + "\n").encode("utf-8"))
            ser.flush()
            # optimistic UI label
            if cmd in ("CH0", "CH1", "MUX"):
                last_mode[0] = cmd
            print("[TX] {}".format(cmd))
        except Exception as e:
            print("[TX ERROR] {} -> {}".format(cmd, e))

    # ---- Per-channel state ----
    z8 = [np.full((8, 8), np.nan, dtype=np.float32),
          np.full((8, 8), np.nan, dtype=np.float32)]
    active = [1, 1]
    last_rx = [0.0, 0.0]
    frame_count = [0, 0]
    dirty = [False, False]

    # ---- Build two windows ----
    figs, axs2d, axs3d, imshows, surfaces = [], [], [], [], []
    XN, YN = np.meshgrid(np.arange(UPSAMPLE_N), np.arange(UPSAMPLE_N))

    for ch in (0, 1):
        fig = plt.figure(figsize=(10, 4))
        try:
            fig.canvas.manager.set_window_title("VL53L5CX CH{}".format(ch))
        except Exception:
            pass

        ax2 = fig.add_subplot(1, 2, 1)
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")

        init_z = np.full((UPSAMPLE_N, UPSAMPLE_N), FLAT_Z_MM, dtype=np.float32)

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        im = ax2.imshow(init_z, origin="lower", aspect="equal")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("mm")
        surf = ax3.plot_surface(XN, YN, init_z, rstride=1, cstride=1, linewidth=0, antialiased=True)

        figs.append(fig)
        axs2d.append(ax2)
        axs3d.append(ax3)
        imshows.append(im)
        surfaces.append(surf)

    # ---- Keyboard handler ----
    def on_key(event):
        k = (event.key or "").lower()
        if k == "0":
            send_mode("CH0")
        elif k == "1":
            send_mode("CH1")
        elif k == "m":
            send_mode("MUX")
        elif k == "h":
            print("Keys: 0=CH0, 1=CH1, m=MUX, h=help")

    for fig in figs:
        fig.canvas.mpl_connect("key_press_event", on_key)

    print("Keyboard controls: 0=CH0, 1=CH1, m=MUX, h=help")

    # ---- Serial drain (shared by both animations) ----
    def pump_serial():
        lines = 0
        t0 = time.time()
        while ser.in_waiting > 0 and lines < MAX_LINES:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            lines += 1
            parsed = parse_line(raw)
            if parsed is None:
                continue

            kind, payload = parsed
            if kind == "MODE":
                last_mode[0] = payload
                continue

            if kind == "FRAME":
                ch = payload["ch"]
                if ch not in (0, 1):
                    continue
                if payload["res"] != 64:
                    continue

                active[ch] = 1 if payload["active"] else 0
                z8[ch] = payload["data"].reshape((8, 8))
                last_rx[ch] = time.time()
                frame_count[ch] += 1
                dirty[ch] = True

            # don't monopolize UI thread
            if time.time() - t0 > 0.02:
                break

    # ---- Draw helpers ----
    def compute_zN(ch):
        if active[ch] == 0 or np.isnan(z8[ch]).all():
            return np.full((UPSAMPLE_N, UPSAMPLE_N), FLAT_Z_MM, dtype=np.float32), "FLAT"
        return upsample_bilinear(z8[ch], UPSAMPLE_N), "LIVE"

    def redraw(ch):
        zN, suffix = compute_zN(ch)

        imshows[ch].set_data(zN)
        imshows[ch].set_clim(0, 3000)

        age = (time.time() - last_rx[ch]) if last_rx[ch] > 0 else 1e9
        axs2d[ch].set_title(
            "CH{} {} | MODE={} | frames={} | age={:.2f}s".format(
                ch, suffix, last_mode[0], frame_count[ch], age
            )
        )

        # 3D redraw throttled
        if (frame_count[ch] % SURFACE_EVERY) == 0:
            try:
                surfaces[ch].remove()
            except Exception:
                pass
            surfaces[ch] = axs3d[ch].plot_surface(
                XN, YN, zN, rstride=1, cstride=1, linewidth=0, antialiased=True
            )
            axs3d[ch].set_title("CH{} Surface ({})".format(ch, suffix))

    # ---- Animation callbacks ----
    # Each figure gets its OWN animation timer -> both windows repaint reliably.
    def tick_fig0(_):
        pump_serial()
        if dirty[0] or frame_count[0] == 0:
            redraw(0)
            dirty[0] = False
        figs[0].canvas.draw_idle()
        return []

    def tick_fig1(_):
        pump_serial()
        if dirty[1] or frame_count[1] == 0:
            redraw(1)
            dirty[1] = False
        figs[1].canvas.draw_idle()
        return []

    ani0 = FuncAnimation(figs[0], tick_fig0, interval=args.interval, blit=False)
    ani1 = FuncAnimation(figs[1], tick_fig1, interval=args.interval, blit=False)

    print("Viewer running. If 3D still heavy, run with --upsample 40 --surface_every 20")
    plt.show()
    ser.close()


if __name__ == "__main__":
    main()