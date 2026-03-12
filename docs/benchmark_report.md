# Benchmark Report (Current Workspace Run)

## Scope
This report summarizes what was measured in this workspace without live hardware.

## Measured in Software
- Unit tests: 8 passed.
- Host QP module solve path runs and reports solve time.
- End-to-end tracer writes JSONL timing records at command-send points.

## Not Measured Yet (Hardware Required)
- True Teensy per-sensor Hz and drop rate.
- Mega command apply latency over USB.
- Full sensor-to-command latency percentiles under load.

## Next Hardware Run Checklist
1. Run host with Teensy + Mega connected for >=10 minutes.
2. Export `logs/end_to_end_timing.jsonl` and compute avg/p95/worst.
3. Capture Teensy frame sequence gap stats.
4. Validate watchdog fallback by unplugging host link.
