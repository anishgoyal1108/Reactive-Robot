# QP Formulation (Two-MCU)

## Decision Variable
- `u in R^n`: next joint command (degrees) for n=6 joints.

## Objective
Minimize tracking error to nominal IK command with regularization:
`min 0.5 * ||u - u_nom||_W^2 + lambda_slack * s`

Implemented v1 objective is quadratic tracking with bounded step from current command.

## Constraints
- Joint bounds: `u_min <= u <= u_max`
- Step/velocity bound per cycle: `|u - u_cur| <= delta_max * speed_scale`
- Obstacle safety:
  - Triggered if nearest modeled distance is below threshold.
  - If distance is critically low, immediate fallback hold.
  - Otherwise solve bounded tracking QP with reduced speed scale.

## Obstacle Avoidance Relation
- Host derives conservative obstacle distance from ToF-derived local model.
- Trigger selects threat level and speed scale.
- QP enforces bounded motion so command does not aggressively approach obstacle.

## Failure Handling
If any of the following occurs, send `fallback_hold`:
- infeasible/invalid solve
- solve exceeds timing budget
- stale or missing sensor data during obstacle mode
- Mega link fault or ACK error stream

## Solver Assumptions
- Preferred backend: OSQP (warm-start capable).
- If OSQP unavailable, deterministic projection fallback is used to preserve safety.
