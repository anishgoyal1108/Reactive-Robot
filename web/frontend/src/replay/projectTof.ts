// projectTof.ts — Client-side replica of the Python TofProjector.
//
// Old session logs (pre-SESSION_LOG_INCLUDE_GEOMETRY) don't contain the
// world-frame [x,y,z] obstacle cloud, but they DO contain the raw
// per-tick 4×N×N ToF distance grids plus the joint vector. That's
// enough to reconstruct the points client-side: run forward kinematics
// along the URDF chain to get each ToF mount's world pose, then cast
// one ray per grid cell inside the sensor FoV.
//
// Matches braccio_ctrl/safety/api.py::TofProjector.project line-for-
// line, with URDF extrinsics baked in from braccio.urdf (the mount
// joints are fixed, so their offsets are compile-time constants).
//
// Units: returned points are in arm-base frame, METRES (the frontend
// convention). Divide by 1000 from mm before consuming.

import { Matrix4, Vector3 } from "three";
import { braccioDegreesToUrdfRadians } from "../state/urdf";
import type { JointVector } from "../state/types";

// VL53L5CX field-of-view — matches _TOF_FOV_DEG in api.py.
const FOV_HALF_RAD = (45.0 / 2) * (Math.PI / 180);

// Distance floor — mirrors the Python TOF_MASKED_SYNTHETIC_MM promotion
// so cells reporting < 20 mm are clamped rather than projected to the
// sensor origin. Used as the default when the constant isn't available.
const MASKED_SYNTHETIC_MM = 200.0;

// URDF extrinsics: fixed offsets from wrist_roll_link to each ToF
// mount. All values copied verbatim from
// braccio_main_runner/braccio_twin/urdf/braccio.urdf. Mount-local +X is
// always the ray axis (matches the Python projector), so the rpy
// rotations below are what take (1,0,0) into the physical outward
// direction of each sensor.
interface MountExtrinsic {
  xyz: [number, number, number];  // metres
  rpy: [number, number, number];  // radians
  channel: number;
}

const TOF_MOUNTS: MountExtrinsic[] = [
  { channel: 0, xyz: [0.010, 0, 0.020], rpy: [0, -Math.PI / 2, 0] },   // up
  { channel: 1, xyz: [0.010, -0.025, 0], rpy: [0, 0, -Math.PI / 2] },  // right
  { channel: 2, xyz: [0.010, 0.025, 0], rpy: [0, 0, Math.PI / 2] },    // left
  { channel: 3, xyz: [0.010, 0, -0.020], rpy: [0, Math.PI / 2, 0] },   // down
];

// URDF chain: fixed origins and rotation axes from braccio.urdf. The
// TypeScript side of the digital twin already encodes these same link
// lengths (Arm.tsx L1/L2/L3), but the sensor projection needs the
// actual transform chain rather than just the tip — wrist_roll_link's
// pose is what the mount extrinsics hang off.
const CHAIN_ORIGINS_M: Array<[number, number, number]> = [
  [0, 0, 0.060],    // base_link → shoulder_pillar
  [0, 0, 0.010],    // shoulder_pillar → upper_arm
  [0.125, 0, 0],    // upper_arm → forearm
  [0.125, 0, 0],    // forearm → wrist_pitch
  [0.030, 0, 0],    // wrist_pitch → wrist_roll
];

const CHAIN_JOINT_NAMES = [
  "joint_base",
  "joint_shoulder",
  "joint_elbow",
  "joint_wrist_vert",
  "joint_wrist_rot",
] as const;

const CHAIN_AXES: Array<"x" | "y" | "z"> = ["z", "y", "y", "y", "x"];

/** Build a world-frame transform for wrist_roll_link given the 6-DOF
 *  servo joint vector. Returns a THREE.Matrix4 whose column vectors are
 *  the mount link's basis in metres. */
function wristRollWorldMatrix(joints: JointVector): Matrix4 {
  const accum = new Matrix4();
  for (let i = 0; i < CHAIN_ORIGINS_M.length; i++) {
    const origin = new Matrix4().makeTranslation(...CHAIN_ORIGINS_M[i]);
    const jointName = CHAIN_JOINT_NAMES[i];
    const rad = braccioDegreesToUrdfRadians(jointName, joints[i]);
    const rot = new Matrix4();
    if (CHAIN_AXES[i] === "z") rot.makeRotationZ(rad);
    else if (CHAIN_AXES[i] === "y") rot.makeRotationY(rad);
    else rot.makeRotationX(rad);
    accum.multiply(origin).multiply(rot);
  }
  return accum;
}

/** Build the (roll, pitch, yaw) = rpy rotation matrix URDF-style (ZYX
 *  extrinsic — equivalent to R = Rz(yaw) · Ry(pitch) · Rx(roll) applied
 *  in that multiplication order in three.js). */
function rpyMatrix(rpy: [number, number, number]): Matrix4 {
  const [roll, pitch, yaw] = rpy;
  return new Matrix4()
    .makeRotationZ(yaw)
    .multiply(new Matrix4().makeRotationY(pitch))
    .multiply(new Matrix4().makeRotationX(roll));
}

/** World-frame transform for one ToF mount link. */
function mountWorldMatrix(joints: JointVector, mount: MountExtrinsic): Matrix4 {
  const wrist = wristRollWorldMatrix(joints);
  const localOrigin = new Matrix4().makeTranslation(...mount.xyz);
  const localRot = rpyMatrix(mount.rpy);
  return wrist.multiply(localOrigin).multiply(localRot);
}

/**
 * Replica of TofProjector.project for one grid. Returns world-frame
 * points in MILLIMETRES so downstream code (ReplayOverlays) can treat
 * the output identically to logged `world_points`. Cells with NaN /
 * non-positive distance are skipped; cells ≥ threshold are skipped so
 * the cloud only contains "close" returns worth visualising.
 */
export function projectGridToWorldPoints(
  grid: number[][],
  channel: number,
  joints: JointVector,
  thresholdMm: number,
): Array<[number, number, number]> {
  const mount = TOF_MOUNTS.find((m) => m.channel === channel);
  if (!mount || grid.length === 0) return [];
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  if (cols === 0) return [];

  const M = mountWorldMatrix(joints, mount);
  const out: Array<[number, number, number]> = [];
  const v = new Vector3();
  for (let r = 0; r < rows; r++) {
    const row = grid[r];
    if (!row) continue;
    for (let c = 0; c < cols; c++) {
      let d = Number(row[c]);
      if (!Number.isFinite(d) || d <= 0) continue;
      if (d < 20) d = MASKED_SYNTHETIC_MM;
      if (d >= thresholdMm) continue;
      const hAng =
        cols === 1
          ? 0
          : ((c - (cols - 1) / 2) / ((cols - 1) / 2)) * FOV_HALF_RAD;
      const vAng =
        rows === 1
          ? 0
          : ((r - (rows - 1) / 2) / ((rows - 1) / 2)) * FOV_HALF_RAD;
      // Mount-local frame: +X ray, +Y horizontal FoV, +Z vertical.
      // d is millimetres — convert to metres, transform, convert back.
      const dm = d / 1000;
      v.set(
        dm * Math.cos(vAng) * Math.cos(hAng),
        dm * Math.cos(vAng) * Math.sin(hAng),
        dm * Math.sin(vAng),
      );
      v.applyMatrix4(M);
      out.push([v.x * 1000, v.y * 1000, v.z * 1000]);
    }
  }
  return out;
}

/** Project every channel in a tick's grid bundle, using each
 *  channel's own threshold. Channel 3 (ground) is excluded by default
 *  to avoid floor-splatter clouds — callers can widen via the
 *  `channels` argument. */
export function projectTickTofToWorld(
  joints: JointVector,
  grids: number[][][],
  thresholds: number[],
  channels: number[] = [0, 1, 2],
): Array<[number, number, number]> {
  const out: Array<[number, number, number]> = [];
  for (const ch of channels) {
    if (ch < 0 || ch >= grids.length) continue;
    const grid = grids[ch];
    const thr = Number.isFinite(thresholds[ch]) ? thresholds[ch] : 250;
    out.push(...projectGridToWorldPoints(grid, ch, joints, thr));
  }
  return out;
}
