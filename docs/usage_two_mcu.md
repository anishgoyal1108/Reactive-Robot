# Usage: Two-MCU Stack

## Flash Teensy
1. Open `ToF_RR_Sensing/arduino/vl5_tca_RR/vl5_tca_RR/vl5_tca_RR.ino` in Arduino IDE / Teensyduino.
2. Select Teensy board and serial port.
3. Compile/upload.
4. Verify serial output includes `INFO,Streaming,Started` and `TF,...` frames.

## Flash Mega
1. Open `braccio_main_runner/braccio_joint_test.ino`.
2. Select Arduino Mega board and port.
3. Compile/upload.
4. Verify startup prints `READY` and periodic `STAT,...`.

## Run Host
```bash
cd braccio_main_runner
python run_braccio.py <mega_port> --teensy-port <teensy_port> --baud 115200 --teensy-baud 115200
```

## Port Identification
- Use `python run_braccio.py --list-ports`.
- Mega should answer `PING` with `PONG`.
- Teensy should stream `TF` lines immediately.

## Replay Mode
- Log frames from `ToFStreamManager.log_jsonl(...)`.
- Replay with `sensing.replay_parser.load_replay(path)`.

## Optimizer Toggle
- Optimizer auto-activates on obstacle trigger.
- Disable by forcing threshold very low or setting mode policy in host control logic.

## Tune Safety
- ToF threshold: runtime keys `f` / `F`.
- Mount/FOV assumptions: `planning/sensor_config.py`.
- QP bounds: `planning/qp_replanner.py` (`max_step_deg`, `min_clearance_m`).
- Mega watchdog: `CMD_TIMEOUT_MS` in firmware.
