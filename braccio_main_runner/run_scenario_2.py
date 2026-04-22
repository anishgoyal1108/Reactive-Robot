#!/usr/bin/env python3
"""
run_scenario_2.py — Automated Scenario 2 runner.

Scenario 2 (Data_Collection_Guide.pdf §2) measures ToF partial-occlusion
patterns when you hold your palm near the arm. The PDF instructs you to
start the autonomous sweep and move your hand; this script diverges slightly
— it commands the arm to a deterministic pose (or slow interpolation for 2d)
so the ToF statistics aren't mixed with sweep-motion blur. You do the palm
movements; the arm holds still; the RL recorder captures the same 74-float
obs vectors and 4-float actions that the main controller writes, in the
same ``logs/rl_transitions_*.npz`` format ``calibrate_noise.py`` consumes.

Usage
-----
    python run_scenario_2.py                       # full 15 min, all sub-scenarios
    python run_scenario_2.py --sub 2a,2c           # only 2a and 2c
    python run_scenario_2.py --no-ir               # skip IR integration
    python run_scenario_2.py --dry-run             # no hardware; exercises the timing / NPZ path

What you do
-----------
Follow the PDF's palm motions:
  2a  hold open palm at θ≈45°, z≈60 mm; slowly move from 25 cm → 12 cm (4 min)
  2b  hold palm at θ≈45°, z≈0 mm    (3 min)
  2c  hold palm at θ≈45°, z≈90 mm   (3 min)
  2d  walk palm θ=45° → 90° along the sweep axis as the arm walks there too (5 min)
Total: 15 min.

Safety
------
* The safety stack is NOT engaged here — the arm sits at an IK-computed pose
  and the only motion is the slow θ walk in 2d (~0.15°/s). Clear your workspace
  before running.
* Ctrl+C or ESC (in curses sense; this script just uses Ctrl+C) → arm returns
  HOME at the slowest slew rate, RLRecorder flushes its NPZ, process exits.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Anchor cwd at the script's directory so LOG_DIR and sibling imports behave
# regardless of where python was invoked from.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
os.chdir(_SCRIPT_DIR)

from braccio_ctrl.arm_state import ArmState  # noqa: E402
from braccio_ctrl.constants import (  # noqa: E402
    BAUD_RATE,
    DEFAULT_PORT,
    HOME_POS,
    LOG_DIR,
    TOF_BAUD_RATE,
    TOF_DEFAULT_PORT,
    TOF_NUM_CHANNELS,
    TOF_THRESHOLDS_MM,
)
from braccio_ctrl.imu_state import IMUState  # noqa: E402
from braccio_ctrl.protocol import cmd_home, cmd_set_all, cmd_set_delta  # noqa: E402
from braccio_ctrl.rl_recorder import RLRecorder  # noqa: E402
from braccio_ctrl.serial_bridge import SerialBridge  # noqa: E402
from braccio_ctrl.state_library import StateLibrary  # noqa: E402
from braccio_ctrl.tof_sensor import ToFBridge, ToFState  # noqa: E402

log = logging.getLogger("scenario_2")


# ── Sub-scenario schedule ─────────────────────────────────────────────────
# Each entry is a dict with:
#   name, desc            — UI text
#   motion                — 'hold' | 'sweep'
#   state_name            — named state in states.json whose `joints` are
#                           commanded verbatim (no IK, no clamping, no
#                           wrist-offset — the joints the user visually
#                           verified in the main controller and saved).
#   duration_s            — how long the sub-scenario runs
#   target_channels       — list of ToF channel indices the scenario is
#                           actually testing (highlighted in the panel).
#                           Other channels still stream + get recorded.
#   For motion == 'sweep' only:
#     sweep_base_lo, sweep_base_hi : base-angle (B) sweep range in degrees
#     sweep_deg_per_s              : target base-angle rate
#   Rationale: the existing IK + wrist_level_angle formula produced wrong
#   physical z for the 2a/2b/2c holds. Option B bypasses IK entirely —
#   the user drives to each desired pose via the main controller (where
#   the gripper's actual position is visible), saves it with M, and the
#   joint set is commanded byte-for-byte here. For 2d sweep, only the
#   base angle is modulated; the other 5 joints stay exactly at the
#   saved values so the verified gripper pose is preserved through the
#   whole sweep.
_SUB_SCENARIOS = [
    {"name": "2a", "desc": "Palm at ~45°, mid height — move hand 25 cm → 12 cm over 4 min",
     "motion": "hold", "state_name": "s2a_pose",
     "duration_s": 4 * 60, "target_channels": [0]},
    {"name": "2b", "desc": "Palm at ~45°, LOW height",
     "motion": "hold", "state_name": "s2b_pose",
     "duration_s": 3 * 60, "target_channels": [0]},
    {"name": "2c", "desc": "Palm at ~45°, HIGH height",
     "motion": "hold", "state_name": "s2c_pose",
     "duration_s": 3 * 60, "target_channels": [0]},
    {"name": "2d", "desc": "Arm sweeps base 0°↔180° from saved mid-height pose; walk palm θ=45°→90°",
     "motion": "sweep", "state_name": "s2d_start",
     "sweep_base_lo": 0.0, "sweep_base_hi": 180.0,
     "sweep_deg_per_s": 60.0,
     "duration_s": 5 * 60, "target_channels": [0, 1]},
]

_CMD_TICK_HZ         = 10.0   # re-issue SET ALL up to this rate during 2d sweep
_DISPLAY_HZ          = 4.0    # refresh the in-place ToF status panel at this rate
_TOF_CH_LABELS       = ["CH0 top   ", "CH1 right ", "CH2 left  ", "CH3 bottom"]
_SLEW_DELTA_HOLD     = 1      # slowest slew — for transit into a hold pose
_SLEW_DELTA_SWEEP    = 4      # faster slew — for 2d sweep so the arm tracks commanded B
_SWEEP_RESEND_DEG    = 2.0    # re-issue SET ALL when sweep base has moved by this much


# ── Helpers ───────────────────────────────────────────────────────────────

def _configure_logging() -> str:
    """Same pattern as braccio_ctrl/__main__._configure_logging, but this
    script isn't inside curses so we can also stream to stdout. Writes
    logs/scenario_2.log for post-session review."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "scenario_2.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return log_path


# NOTE: the IK-based _ik_joints helper was removed in favour of saved
# named states (see Option B in the Scenario-2 planning thread). The
# wrist_level_angle formula in ik_solver clamps to 0 for the S/E values
# typical of 2a-2c poses, producing a gripper tilt that makes commanded
# z not match physical z. Loading joints directly from a named state
# (which the user visually verified in the main controller) sidesteps
# the IK entirely.


def _format_status_panel(sub_name: str, remaining_s: float, theta: float,
                          r: float, z: float, rec_buffered: int,
                          tof_snap: dict, ir_enabled: bool,
                          target_channels: Optional[list[int]] = None) -> list[str]:
    """Build the multi-line ToF + pose + recorder panel.

    Mirrors the layout of the main controller's curses status pane:
    a header line + per-channel ToF min/valid/Hz + IR / obstacle summary.
    Returned as a list of lines so the caller can redraw in place via
    ANSI cursor-up.
    """
    m, s = divmod(int(remaining_s), 60)
    hdr = (f"[{sub_name}]  {m:02d}:{s:02d} remaining   "
           f"θ={theta:5.1f}°  r={r:5.1f} mm  z={z:+6.1f} mm   "
           f"rec {rec_buffered} samp")

    lines = [hdr]
    grids    = tof_snap.get("grids",    [None] * 4)
    hz       = tof_snap.get("hz",       [0] * 4)
    min_mm   = tof_snap.get("diag_raw_min_mm", [float("inf")] * 4)
    valid    = tof_snap.get("diag_valid_cells", [0] * 4)
    thresh   = tof_snap.get("tof_thresholds_mm", TOF_THRESHOLDS_MM)
    active   = tof_snap.get("active",   [0] * 4)

    target = set(target_channels or [])
    for ch in range(4):
        label = _TOF_CH_LABELS[ch]
        marker = "►" if ch in target else " "
        if not active[ch]:
            lines.append(f" {marker}{label}: (no frames)")
            continue
        g = grids[ch] if ch < len(grids) else None
        # Grid has 64 cells (8×8); count cells under this channel's threshold.
        below = 0
        if g is not None and getattr(g, "size", 0) > 0:
            try:
                import numpy as _np
                below = int((g < thresh[ch]).sum())
            except Exception:
                below = 0
        # CH3 is always filtered (MUX self-returns); channels not in the
        # scenario's target set are "off-target" but still streaming.
        if ch == 3:
            tag = " IGNORED"
        elif ch in target:
            tag = " ◀ TARGET"
        else:
            tag = " (off-target)"
        mm = min_mm[ch] if min_mm[ch] not in (None, float("inf")) else None
        mm_str = f"{mm:5.0f} mm" if mm is not None else "  n/a "
        lines.append(
            f" {marker}{label}: min {mm_str}  valid {valid[ch]:2d}/64  "
            f"below-thresh {below:2d}  @{hz[ch]:4.1f} Hz{tag}"
        )

    ir_bits = tof_snap.get("ir_bits", 0)
    ir_label = tof_snap.get("ir_label", "DISABLED" if not ir_enabled else "?")
    resp = tof_snap.get("obstacle_response", "clear")
    src  = tof_snap.get("obstacle_source", "")
    dist = tof_snap.get("obstacle_dist_mm", -1.0)
    dist_str = f"{dist:.0f} mm" if dist is not None and dist >= 0 else "—"
    lines.append(
        f"  IR: {ir_label:<8}  bits={ir_bits}   |   "
        f"obstacle: {resp} {f'(src={src}, {dist_str})' if src else ''}"
    )
    return lines


def _update_arm_state(arm_state: ArmState, theta: float, r: float, z: float,
                       joints: list[int]) -> None:
    """Write commanded pose into ArmState so the recorder captures it.

    The RLRecorder doesn't talk to the serial bridge — it reads theta/r/z
    from ArmState.snapshot(). In the main controller those fields are
    written by the key-dispatch path; here we write them directly after
    each SET ALL. The polar_synced latch is set so the recorder treats
    polar as authoritative (same contract as the main controller after
    first POS sync).
    """
    with arm_state._lock:
        arm_state.theta         = float(theta)
        arm_state.r             = float(r)
        arm_state.z             = float(z)
        arm_state.joints        = list(joints)
        arm_state.polar_synced  = True


def _drain_response_queue(bridge: SerialBridge, stop_event: threading.Event) -> None:
    """Silently empty the bridge's response queue so its reader thread
    doesn't block on put() after 128 unread replies.

    The controller's ``_drain_responses`` does this AND updates ArmState
    from POS replies; this script owns ``theta/r/z`` directly, so there's
    nothing to do with the parsed payloads except throw them away.
    """
    while not stop_event.is_set():
        try:
            bridge.response_queue.get(timeout=0.1)
        except queue.Empty:
            continue


# ── Main runner ───────────────────────────────────────────────────────────

class Scenario2Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dry_run = args.dry_run

        # State library — named poses saved via the main controller's M key.
        # Each sub-scenario references one of these by name; joints are
        # commanded byte-for-byte so the visually-verified gripper angle
        # is preserved exactly (no IK, no wrist auto-level guesswork).
        self.state_lib = StateLibrary()

        # Shared state objects (same types the main controller uses, so
        # RLRecorder reads them via the identical snapshot contract).
        self.arm_state = ArmState()
        self.tof_state = ToFState(num_channels=TOF_NUM_CHANNELS)
        self.tof_state.tof_threshold_mm  = TOF_THRESHOLDS_MM[0]
        self.tof_state.tof_thresholds_mm = list(TOF_THRESHOLDS_MM)
        self.tof_state.ir_enabled        = bool(args.enable_ir)
        self.imu_state = IMUState()

        # Hardware bridges — skipped in --dry-run so you can exercise timing
        # + NPZ writing without the arm plugged in.
        self.bridge: Optional[SerialBridge] = None
        self.tof_bridge: Optional[ToFBridge] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._drain_stop = threading.Event()

        # Recorder — same instance / same NPZ format as the main controller.
        self.recorder = RLRecorder(
            arm_state=self.arm_state,
            tof_state=self.tof_state,
            imu_state=self.imu_state,
            obstacle_map=None,
        )

        self._shutdown_done = False
        self._schedule = self._filter_schedule(args.sub)
        self._validate_states_or_abort()

    def _validate_states_or_abort(self) -> None:
        """Abort with instructions if any required named state is missing.

        Done BEFORE connecting to hardware so the user doesn't burn 30 s of
        Arduino boot + ToF warmup only to discover they forgot to save a
        pose. Listing every missing state in one message also beats one-
        sub-scenario-at-a-time failure during the run.
        """
        missing = []
        for entry in self._schedule:
            name = entry["state_name"]
            if self.state_lib.get_state(name) is None:
                missing.append((entry["name"], name, entry["desc"]))
        if not missing:
            return
        lines = [
            "Missing named state(s) in states.json — save them via the",
            "main controller (press M → 'Save current state') first:\n",
        ]
        for sub, state, desc in missing:
            lines.append(f"  [{sub}] needs state '{state}'  ({desc})")
        lines += [
            "",
            "Procedure:",
            "  1. python -m braccio_ctrl /dev/ttyACM0 --teensy-port /dev/ttyACM1 --no-ir",
            "  2. Use A/D/W/S/Q/E/I/K to drive the arm to the target pose.",
            "  3. Verify the gripper height is actually what you want by LOOKING at it.",
            "  4. Press M, choose 'Save current state', type the name above, Enter.",
            "  5. Repeat for each missing state. ESC when done.",
            "  6. Re-run this script.",
        ]
        raise SystemExit("\n".join(lines))

    # ── Scheduling ────────────────────────────────────────────────────────
    @staticmethod
    def _filter_schedule(sub_arg: Optional[str]) -> list[dict]:
        if not sub_arg:
            return list(_SUB_SCENARIOS)
        wanted = {s.strip().lower() for s in sub_arg.split(",") if s.strip()}
        kept = [entry for entry in _SUB_SCENARIOS
                if entry["name"].lower() in wanted]
        missing = wanted - {e["name"].lower() for e in _SUB_SCENARIOS}
        if missing:
            raise SystemExit(f"Unknown sub-scenario(s): {sorted(missing)}. "
                             f"Valid: {[e['name'] for e in _SUB_SCENARIOS]}")
        return kept

    # ── Lifecycle ─────────────────────────────────────────────────────────
    def connect(self) -> None:
        if self.dry_run:
            log.info("DRY RUN — no serial connections opened")
            return

        # Arm serial + PING handshake (same contract as the main controller's
        # _connect). The Arduino resets on DTR assertion and the
        # braccio_joint_test firmware's setup() does two 2-second safeDelays
        # + a home move before it starts reading Serial — roughly 4-5 s of
        # deaf time. SerialBridge.connect already sleeps 2 s, so we need to
        # keep trying PING for several more seconds after that to cover the
        # remainder of Arduino boot, or we hit a race that looks like a
        # totally-unresponsive board.
        self.bridge = SerialBridge(self.args.port, self.args.baud)
        if not self.bridge.connect():
            raise SystemExit(f"Cannot open arm serial {self.args.port}: "
                             f"{self.bridge.connect_error}")

        ping_deadline = time.monotonic() + 8.0
        pong = False
        log.info("Waiting for Arduino PONG on %s (up to 8 s — firmware "
                 "setup() takes ~4 s)...", self.args.port)
        while time.monotonic() < ping_deadline and not pong:
            pong = self.bridge.ping(timeout=1.0)
        if pong:
            log.info("Arm connected on %s (PONG received).", self.args.port)
        else:
            # Inspect whatever the device DID emit during the wait so we can
            # tell the user whether this is a real problem or just a sleepy
            # Arduino that'll respond to commands fine once it finishes
            # booting.
            unsolicited: list[str] = []
            try:
                while True:
                    resp = self.bridge.response_queue.get_nowait()
                    unsolicited.append(
                        resp.get("raw", str(resp)) if resp.get("type") == "unknown"
                        else f'{resp.get("type")}={resp}'
                    )
                    if len(unsolicited) >= 5:
                        break
            except queue.Empty:
                pass
            teensy_signature = any(
                s.startswith(("TF,", "IMU,", "IR,")) for s in unsolicited
            )
            if teensy_signature:
                raise SystemExit(
                    f"Device on {self.args.port} is emitting Teensy frames "
                    f"(TF/IMU/IR). Swap --port and --teensy-port."
                    f"\n  First lines: {unsolicited[:3]!r}"
                )
            # No PONG and no Teensy signature → Arduino might still be
            # booting OR firmware genuinely hung. The main controller
            # warns-and-continues here: subsequent SET ALL commands will
            # queue in the UART buffer and run once the Arduino is ready.
            log.warning(
                "No PONG from %s within 8 s — continuing anyway. If the "
                "arm doesn't move when sub-scenarios start, the firmware "
                "is not responding (re-upload braccio_joint_test.ino or "
                "check the Arduino IDE isn't holding the port).",
                self.args.port,
            )
            if unsolicited:
                log.warning("  Unsolicited lines observed: %r", unsolicited[:3])

        # Teensy serial + wait for first ToF frame. The recorder will see
        # None grids for the first few ticks if we skip this wait, which
        # encodes as all-clear sentinels (=1.0) — plausible but dishonest.
        if self.args.teensy_port:
            self.tof_bridge = ToFBridge(self.tof_state, self.imu_state)
            if not self.tof_bridge.connect(self.args.teensy_port, self.args.teensy_baud):
                log.warning("Teensy connect failed (%s) — proceeding without ToF",
                            getattr(self.tof_bridge, "_last_open_error", "unknown"))
                self.tof_bridge = None
            else:
                log.info("Teensy connected on %s — waiting for first ToF frame...",
                         self.args.teensy_port)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    snap = self.tof_state.snapshot()
                    if any(snap.get("frame_cnt", [])):
                        log.info("ToF frames arriving.")
                        break
                    time.sleep(0.05)
                else:
                    log.warning("No ToF frames within 2 s — proceeding anyway")

        # Slowest slew so the initial homing move doesn't whip.
        self.bridge.send_cmd(cmd_set_delta(_SLEW_DELTA_HOLD))

        # Drainer so the bridge's response_queue doesn't back up and block
        # its reader thread. We don't consume POS here (we own polar state).
        self._drain_thread = threading.Thread(
            target=_drain_response_queue,
            args=(self.bridge, self._drain_stop),
            daemon=True,
            name="scenario2-drain",
        )
        self._drain_thread.start()

    def go_home_and_wait(self, settle_s: float = 3.0) -> None:
        """Park at HOME and give the arm time to physically settle before
        starting recorder — otherwise the first sub-scenario's opening
        seconds capture the arm mid-transit."""
        if self.dry_run:
            _update_arm_state(self.arm_state, 90.0, 152.0, -50.0, list(HOME_POS))
            log.info("DRY RUN: pretend arm is at HOME")
            return
        assert self.bridge is not None
        self.bridge.send_cmd(cmd_home())
        _update_arm_state(self.arm_state, 90.0, 152.0, -50.0, list(HOME_POS))
        log.info("Sent HOME, settling for %.1f s...", settle_s)
        time.sleep(settle_s)

    def send_joints(self, joints: list[int], theta: float, r: float,
                     z: float) -> None:
        """Send a verbatim 6-joint SET ALL and mirror the pose metadata
        into ArmState. No IK, no clamping beyond what the Arduino does
        itself — the joints list is whatever the caller provided.
        """
        if not self.dry_run:
            assert self.bridge is not None
            self.bridge.send_cmd(cmd_set_all(joints))
        _update_arm_state(self.arm_state, theta, r, z, joints)

    def _set_slew(self, delta: int) -> None:
        """Thin wrapper around SET DELTA for clarity at call sites."""
        if self.dry_run or self.bridge is None:
            return
        self.bridge.send_cmd(cmd_set_delta(int(delta)))

    def run_sub_scenario(self, entry: dict) -> None:
        name     = entry["name"]
        desc     = entry["desc"]
        duration = entry["duration_s"]
        motion   = entry["motion"]
        targets  = entry.get("target_channels", [])
        state    = self.state_lib.get_state(entry["state_name"])
        # Already validated at __init__; keep an assert for clarity.
        assert state is not None, f"state '{entry['state_name']}' missing"
        saved_joints = list(state["joints"])
        # Metadata for the ArmState mirror + panel. These are whatever the
        # main controller recorded when the pose was saved; they may be IK
        # nonsense but they stay internally consistent with the rest of
        # the session (and with any other code that reads arm_state polar).
        meta_theta = float(state.get("theta", 0.0))
        meta_r     = float(state.get("r",     0.0))
        meta_z     = float(state.get("z",     0.0))

        print("\n" + "=" * 68)
        print(f"[{name}] {desc}")
        print(f"    source state: '{entry['state_name']}'  joints={saved_joints}")
        if motion == "hold":
            print(f"    arm holding saved pose")
        elif motion == "sweep":
            lo = entry["sweep_base_lo"]; hi = entry["sweep_base_hi"]
            speed = entry["sweep_deg_per_s"]
            period = 2 * (hi - lo) / speed
            print(f"    arm sweeping base {lo:.0f}° ↔ {hi:.0f}° at "
                  f"{speed:.0f}°/s (full cycle {period:.1f} s); "
                  f"S/E/WV/WR/G held at saved values")
        else:
            raise RuntimeError(f"Unknown motion mode: {motion}")
        print(f"    duration: {duration // 60}:{duration % 60:02d}   "
              f"target ToF: {', '.join(f'CH{c}' for c in targets) or 'none'}")
        print("=" * 68)

        # Move to starting pose at slow slew, then wait for the arm to
        # arrive before the panel + recorder timing begins.
        self._set_slew(_SLEW_DELTA_HOLD)
        if motion == "hold":
            initial_joints = list(saved_joints)
        else:  # sweep — start at the low end of the base-angle range
            initial_joints = list(saved_joints)
            initial_joints[0] = int(round(entry["sweep_base_lo"]))
        self.send_joints(initial_joints, meta_theta, meta_r, meta_z)
        print(f"    moving into position...", flush=True)
        time.sleep(2.5)

        if motion == "sweep":
            self._set_slew(_SLEW_DELTA_SWEEP)

        start = time.monotonic()
        end   = start + duration
        last_base_sent = float(initial_joints[0])
        last_cmd_t = start
        cmd_period = 1.0 / _CMD_TICK_HZ
        display_period = 1.0 / _DISPLAY_HZ
        last_display_t = 0.0
        panel_lines_drawn = 0
        use_ansi = sys.stdout.isatty()
        if use_ansi:
            sys.stdout.write("\033[?25l")  # hide cursor
            sys.stdout.flush()

        try:
            while True:
                now = time.monotonic()
                if now >= end:
                    break
                remaining = end - now

                if motion == "hold":
                    current_joints = saved_joints
                    current_base = float(saved_joints[0])
                    need_resend = (now - last_cmd_t) >= 5.0
                else:
                    # Triangle wave on the base angle only.
                    lo = entry["sweep_base_lo"]; hi = entry["sweep_base_hi"]
                    speed = entry["sweep_deg_per_s"]
                    span = hi - lo
                    period = 2 * span / speed
                    t = (now - start) % period
                    b = (lo + speed * t) if t < period / 2 else \
                        (hi - speed * (t - period / 2))
                    current_base = b
                    current_joints = list(saved_joints)
                    current_joints[0] = int(round(b))
                    need_resend = abs(current_base - last_base_sent) >= _SWEEP_RESEND_DEG \
                        or (now - last_cmd_t) >= 1.0

                if need_resend:
                    # Use the saved meta theta if holding, or the commanded
                    # base angle if sweeping, so obs[0] reflects reality.
                    display_theta = current_base if motion == "sweep" else meta_theta
                    self.send_joints(current_joints, display_theta, meta_r, meta_z)
                    last_base_sent = current_base
                    last_cmd_t = now

                # Redraw ToF panel at _DISPLAY_HZ.
                if (now - last_display_t) >= display_period:
                    last_display_t = now
                    rec_st = self.recorder.status()
                    tof_snap = self.tof_state.snapshot()
                    panel_theta = current_base if motion == "sweep" else meta_theta
                    panel = _format_status_panel(
                        name, remaining, panel_theta, meta_r, meta_z,
                        rec_st["buffered"], tof_snap,
                        ir_enabled=self.args.enable_ir,
                        target_channels=targets,
                    )
                    if use_ansi and panel_lines_drawn:
                        sys.stdout.write(f"\033[{panel_lines_drawn}A")
                    for line in panel:
                        if use_ansi:
                            sys.stdout.write("\033[2K" + line + "\n")
                        else:
                            sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                    panel_lines_drawn = len(panel)

                time.sleep(cmd_period)
        finally:
            if use_ansi:
                sys.stdout.write("\033[?25h")   # restore cursor
                sys.stdout.flush()
            # Always return to the slow slew before the next sub-scenario
            # (or before we park at HOME in shutdown).
            self._set_slew(_SLEW_DELTA_HOLD)

    # ── Top-level driver ─────────────────────────────────────────────────
    def run(self) -> None:
        self.connect()
        self.go_home_and_wait(settle_s=3.0)

        # Start recorder AFTER arm is at HOME so the opening frames are
        # honest (arm in a real known pose, ToF streaming real frames).
        self.recorder.start()
        log.info("RL recorder running — NPZ will be written to %s", LOG_DIR)

        total = sum(e["duration_s"] for e in self._schedule)
        print(f"\nTotal session time: {total // 60}:{total % 60:02d}")
        print(f"Writing NPZ to: {LOG_DIR}/rl_transitions_<timestamp>.npz\n")

        try:
            for entry in self._schedule:
                self.run_sub_scenario(entry)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        # Belt-and-suspenders cursor restore — the panel loop's finally block
        # handles normal exits, but SIGINT inside time.sleep() unwinds via
        # this path and may have left the terminal with the cursor hidden.
        if sys.stdout.isatty():
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
        log.info("Shutting down...")

        # Stop recorder FIRST so it can flush while the arm state snapshot
        # path is still intact. Surface success/failure loudly — this is
        # the artifact the whole session exists to produce.
        try:
            self.recorder.stop()
        except Exception as exc:
            print(f"\nERROR: rl_recorder flush failed: {exc}", file=sys.stderr)
            log.exception("recorder.stop() failed")
        else:
            st = self.recorder.status()
            if st["last_save_path"]:
                print(f"\n✓ NPZ saved: {st['last_save_count']} transitions → "
                      f"{st['last_save_path']}")
            elif st["tick_fatal"]:
                print(f"\nERROR: recorder died: {st['tick_fatal']}", file=sys.stderr)
            else:
                print("\nWARNING: recorder stopped with 0 transitions buffered.",
                      file=sys.stderr)

        # Park at HOME so the arm is in a known-safe pose next time.
        if not self.dry_run and self.bridge is not None:
            try:
                self.bridge.send_cmd(cmd_set_delta(1))
                self.bridge.send_cmd(cmd_home())
            except Exception:
                pass

        self._drain_stop.set()
        if self.bridge is not None:
            try:
                self.bridge.close()
            except Exception:
                pass
        if self.tof_bridge is not None:
            try:
                self.tof_bridge.close()
            except Exception:
                pass


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Scenario 2 data collection (see "
                    "Data_Collection_Guide.pdf §2).",
    )
    # Accept the arm port both positionally (matches `python -m braccio_ctrl
     # /dev/ttyACM1 ...`) and as --port (explicit). Positional wins if both.
    parser.add_argument("arm_port", nargs="?", default=None,
                        help=f"Arm serial port (positional, like the main "
                             f"runner). Default: {DEFAULT_PORT}")
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Arm serial port (alternative to positional). "
                             f"Default: {DEFAULT_PORT}")
    parser.add_argument("--baud", type=int, default=BAUD_RATE,
                        help=f"Arm baud (default: {BAUD_RATE})")
    parser.add_argument("--teensy-port", default=TOF_DEFAULT_PORT,
                        help=f"Teensy serial port (default: {TOF_DEFAULT_PORT}); "
                             "pass empty string to disable ToF entirely")
    parser.add_argument("--teensy-baud", type=int, default=TOF_BAUD_RATE)
    parser.add_argument("--no-ir", action="store_true",
                        help="Disable IR channel processing")
    parser.add_argument("--sub", default=None,
                        help="Comma-separated sub-scenarios to run "
                             "(default: 2a,2b,2c,2d)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip hardware — exercise timing + NPZ writer only")
    parser.add_argument("--list-ports", action="store_true",
                        help="List available serial ports and exit (identify "
                             "which /dev/ttyACMx is the arm vs. the Teensy).")
    args = parser.parse_args()
    if args.list_ports:
        ports = SerialBridge.list_ports()
        if not ports:
            print("No serial ports found.")
        else:
            print("Available serial ports:")
            for device, desc in ports:
                print(f"  {device:<20s}  {desc}")
        return
    args.enable_ir = not args.no_ir
    # Resolve positional vs flag.
    if args.arm_port is not None:
        args.port = args.arm_port

    log_path = _configure_logging()
    print(f"Logging to {log_path}\n")

    runner = Scenario2Runner(args)

    # Park at HOME + flush NPZ on any exit path: normal completion, Ctrl+C,
    # SIGTERM, uncaught exception, or terminal close. Matches BraccioController's
    # shutdown contract.
    atexit.register(runner.shutdown)
    def _sig(signum, _frame):
        log.info("Signal %s received — shutting down", signum)
        runner.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)
    try:
        signal.signal(signal.SIGHUP, _sig)
    except (AttributeError, ValueError):
        pass

    try:
        runner.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
