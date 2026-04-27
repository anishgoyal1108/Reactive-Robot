# Teensy Sensor Timing

## Measurement Method
- Firmware emits v1 `TF` frames containing `seq` and `mcu_ms`.
- On host, parse stream and compute:
  - per-sensor frame delta (`mcu_ms`/`seq` gaps)
  - round-robin cycle time across deployed channels
  - serialization overhead from line size and baud rate

## Current Deployment Assumptions
- Active channels: CH0 and CH1
- Supported channels: CH0..CH3 (future)
- Baud: 115200
- Matrix: 64 distances + 64 validity bits per frame

## Analytical Estimate (Pre-HIL)
- TF line payload roughly 700-900 bytes depending on values.
- At 115200 baud (~11.5 kB/s effective), serialization costs ~60-80 ms/frame worst case.
- With 2 active sensors, pure serial overhead suggests ~6-8 Hz/sensor if every frame is full-length.
- Claimed 15 Hz/sensor is unlikely at 115200 with full 64+64 payload unless:
  - baud is increased, or
  - packet is compressed/binary, or
  - frame payload is reduced.

## Required Hardware Validation
Run with target hardware and capture 60+ seconds:
- mean/95th/worst per-sensor period
- mean/95th/worst round-robin cycle
- parse fail and invalid-zone rate
- sequence gap count (drops)

## Conclusion
- v1 protocol supports measurement and diagnosis now.
- Real achieved rate must be measured on bench; current payload/baud combination likely constrains per-sensor rate below 15 Hz for two active channels.
