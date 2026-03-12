# Communication Interfaces Assessment

## Physical Links
- **Host <-> Teensy**: USB serial (sensor telemetry)
- **Host <-> Mega**: USB serial (commands + ack/status)
- **Teensy <-> Mega**: no direct link

## Teensy Sensor Link (Host Receiver)
- Baud: `115200`
- Encoding: ASCII lines
- Main packets:
  - `TF,<...>` for ToF CH0/CH1
  - `IMU,<...>` for MPU6050 on direct I2C (logical CH2 stream)
  - `IR,<bits>` for IR danger input
  - `MODE,...`, `CFG,ACT,...` for runtime config
- Sequencing/timestamp:
  - `TF` and `IMU` include `seq` and `mcu_ms`
- Ack:
  - none (streaming telemetry)
- Sensor rates in current firmware:
  - ToF round-robin across CH0/CH1 at ~15 Hz per channel (target)
  - IMU streamed at 100 Hz (10 ms interval)

## Mega Command Link (Host Sender)
- Baud: `115200`
- Encoding: ASCII lines
- Command packets:
  - Primary runtime path: `SET IKP ...` (MCU IK)
  - Compatible path: `CMD,<...>`
- Return telemetry:
  - `ACK,<...>` and `STAT,<...>` (plus legacy responses)
- Sequencing/timestamp:
  - `CMD`/`ACK` carry sequence IDs and timestamps

## Robustness Assessment
- Host can operate both links in parallel using separate serial reader threads.
- Bounded queues prevent one link from starving the control loop.
- Mega watchdog timeout provides low-level hold safety if host commands stall.
- Host freshness checks gate stale ToF/IMU use.

## Coexistence Risks and Mitigations
- Risk: high-rate display/plotting can steal loop time.
  - Mitigation: control loop decoupled from display cadence; plots disabled by default.
- Risk: malformed or partial serial lines.
  - Mitigation: strict parsing with safe rejection.
- Risk: stale sensing but continued motion.
  - Mitigation: host fallback hold mode when freshness fails.
- Risk: command link loss.
  - Mitigation: Mega timeout hold + host mode `comms_fault`.

## Link Identification (explicit)
- Teensy publishes ToF data on **Host-Teensy USB serial** via `TF` lines.
- Host sends arm commands to Mega on **Host-Mega USB serial** via `SET IKP`/`CMD` lines.
- Host runs both links concurrently and non-blocking in current architecture.
