# Obstacle Detection Pipeline

1. Host keeps the Teensy in `ACT 2` during normal sweep detection.
2. Only `CH0` and `CH1` are allowed to initiate a new obstacle event.
3. A threshold crossing below the configured detection threshold, default `250 mm`, stops the arm.
4. Host switches the Teensy to `ACT 4` and confirms the obstacle with a `3 of 5` filtered sample window.
5. Confirmed samples are projected into base frame and clustered into one temporary obstacle sector.
6. Host performs a vertical opening probe: the source channel must clear above threshold and `CH2/CH3` must stay above the support clearance gate.
7. Host generates deterministic polar waypoint candidates, preferring the confirmed vertical opening and allowing egress when the starting pose is already inside the inflated detected sector.
8. Host commits one safe candidate, sends `SET IKP ...` commands to the Mega, and ignores new obstacle initiations while the plan is executing.
9. All ToF channels stay active to track/confirm the stored obstacle during avoidance.
10. After the current sweep leg completes, the host clears temporary obstacle memory and returns to `ACT 2` detection mode.

## Mount Assumptions

- `CH0`: west/left side of wrist
- `CH1`: east/right side of wrist
- `CH2`: north/top support channel
- `CH3`: south/bottom support channel
- IMU correction is used for wrist pose when calibrated and fresh.
- The temporary obstacle sector radius is robust-capped to the physical demo scale, and planner clearance is intentionally Braccio-sized rather than QP-sized.
