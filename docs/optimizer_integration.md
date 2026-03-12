# Optimizer Integration

## Inputs
- Current joint state
- Nominal IK command
- Trigger/threat and nearest obstacle metric
- Speed scaling recommendation

## Outputs
- Safe command compatible with Mega interface
- Solver status
- Solve time
- Fallback indicator
- Predicted clearance

## Runtime Behavior
- No threat: send nominal command, optimizer inactive.
- Threat: solve QP with bounded step and speed scaling.
- Failure/stale data/comms fault: send fallback hold command.

## Diagnostics
- Current mode
- Optimizer active flag
- QP solve time
- End-to-end latency
- Deadline misses
