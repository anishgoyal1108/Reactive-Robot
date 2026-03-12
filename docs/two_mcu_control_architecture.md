# Two-MCU Control Architecture

## Roles
- Teensy MCU: ToF sensing, MUX control, frame validation, stream publishing.
- Arduino Mega MCU: actuator command parsing and execution, watchdog safety fallback.
- Host: IK, obstacle projection/modeling, trigger, QP replanning, mode management.

## Modes
- `nominal_tracking`
- `obstacle_aware_tracking`
- `fallback_hold`
- `comms_fault`

## Safety
- Mega holds current position on command timeout.
- Host enters fallback hold on stale sensing, QP failure, or comm faults.

## Compatibility
- Legacy text protocol retained during migration.
- v1 packet contracts available on both links.
