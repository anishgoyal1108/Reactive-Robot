# braccio_description

Local vendored copy of the Braccio URDF + STL meshes.

## Source

Fetched from:
`https://github.com/lots-of-things/braccio_moveit_gazebo/tree/main/braccio_description`

Specifically:
- `urdf/braccio_model.urdf`
- `urdf/braccio_planning_model.urdf`
- `urdf/xacros/braccio_arm.xacro`
- `stl/braccio_base.stl`
- `stl/braccio_shoulder.stl`
- `stl/braccio_elbow.stl`
- `stl/braccio_wrist_pitch.stl`
- `stl/braccio_wrist_roll.stl`
- `stl/braccio_left_gripper.stl`
- `stl/braccio_right_gripper.stl`

## Changes

`package://braccio_description/stl/` references in the URDFs have been
rewritten to `../stl/` so the URDFs can be loaded directly by PyBullet /
three.js without needing a ROS workspace.

## Usage

- PyBullet sim (`braccio_ctrl/sim/braccio_env.py`, Phase 2):
  `p.loadURDF("assets/braccio_description/urdf/braccio_model.urdf")`
- Web viewer (if ported from naive primitives to realistic meshes):
  point urdf-loaders at the same URDF and host the STLs statically.
