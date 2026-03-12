# Obstacle Detection Pipeline

1. Teensy polls active ToF sensors (CH0/CH1 deployed) through TCA9548A.
2. Teensy validates zones and emits `TF` packet with sequence, timestamp, distances, validity mask.
3. Host parses packet and updates per-channel freshness/health.
4. Host projects valid cells from sensor frame to base frame using configured mount assumptions.
5. Host builds conservative obstacle model with inflation and smoothing.
6. Host computes trigger result (`optimizer_active`, threat level, nearest metric, speed scale).
7. Host transitions control mode and updates operator diagnostics.

## Mount Assumptions (v1)
- Two sensors on sides of wrist-vertical joint.
- Side-looking beams, configurable yaw/pitch/FOV in `planning/sensor_config.py`.
