# End-to-End Timing Budget

## Chain Definition
1. Teensy acquisition
2. Teensy serialization/transmission
3. Host parse/ingest
4. Obstacle model update
5. QP solve
6. Host transmit to Mega
7. Mega parse/apply

## Instrumentation Hooks Implemented
- Host tracer marks:
  - frame receive
  - model done
  - QP done
  - command tx
  - optional ACK receive
- Logs written to `logs/end_to_end_timing.jsonl`.

## Budget Targets (v1)
- Host model + trigger: <= 5 ms
- QP solve: <= 30 ms
- Host sensor-to-command total: <= 50 ms nominal
- Command watchdog on Mega: 750 ms hard safety bound

## Current Status
- Software instrumentation implemented.
- Hardware measured avg/p95/worst pending bench run.
- Deadline miss counter exposed in operator diagnostics.

## Benchmark Procedure
- Run 10-minute session with nominal + obstacle events.
- Compute avg/p95/worst for each stage and total latency.
- Compare against control cycle and sensor update cadence.
