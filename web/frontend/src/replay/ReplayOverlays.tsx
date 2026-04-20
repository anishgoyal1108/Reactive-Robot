// ReplayOverlays.tsx — 3D overlays driven by the ReplayStore.
//
// Rendered as a child of the existing <Scene/> so it inherits the
// scene-root rotation (URDF Z-up → three.js Y-up). Every overlay
// position is computed in arm-base-frame METRES (divide mm-based log
// fields by 1000), matching the units used by Arm.tsx link lengths.
//
// Four overlays:
//   1. Obstacle cloud  — red spheres at each tick's world_points.
//   2. Detour path     — orange polyline through IK-projected detour
//                        waypoints. Only drawn during an active detour.
//   3. Trail           — last N tip positions across the tick history
//                        (white for normal sweep, amber mid-detour,
//                        red on reverse).
//   4. Sweep target    — thin green line from tip to the nominal
//                        (r, target_theta, z) point.

import { useMemo } from "react";
import { Line } from "@react-three/drei";
import { useReplayStore } from "./ReplayStore";
import { projectTickTofToWorld } from "./projectTof";
import type { JointVector } from "../state/types";

// Link lengths (metres) — match Arm.tsx exactly.
const L1 = 0.125;
const L2 = 0.125;
const L3 = 0.060;
const SHOULDER_Z = 0.07; // base-top → shoulder-pivot height (metres)
const TRAIL_MAX = 40;

// Nominal sweep (r, z) target line; must match constants.py defaults.
const SWEEP_R_M = 0.152;
const SWEEP_Z_M = 0.035;

type Vec3 = [number, number, number];

/** Forward kinematics for the gripper tip in arm-base frame (metres).
 *  Mirrors ik_solver.py's polar forward: base rotates the whole chain
 *  around Z; shoulder/elbow/wrist_vert all hinge in the radial plane.
 *  The convention matches the IK solver so tip positions plotted here
 *  line up with where the solver commanded the arm to go. */
function fkTipXYZ(joints: JointVector): Vec3 {
  const toRad = (d: number) => (d * Math.PI) / 180;
  const base = toRad(joints[0]);
  // Braccio servo convention: shoulder/elbow/wrist measured from the
  // chassis; the ik_solver subtracts 90° / 180° to bring them into a
  // standard planar-arm frame. Match those offsets here so the tip
  // placement matches what solve_ik() produced.
  const shoulderInternal = toRad(joints[1] - 90);
  const elbowInternal = toRad(joints[2] - 90 + (joints[1] - 90));
  const wristInternal = toRad(
    joints[3] - 90 + (joints[2] - 90) + (joints[1] - 90),
  );

  const r =
    L1 * Math.cos(shoulderInternal) +
    L2 * Math.cos(elbowInternal) +
    L3 * Math.cos(wristInternal);
  const z =
    SHOULDER_Z +
    L1 * Math.sin(shoulderInternal) +
    L2 * Math.sin(elbowInternal) +
    L3 * Math.sin(wristInternal);
  const x = r * Math.cos(base);
  const y = r * Math.sin(base);
  return [x, y, z];
}

function strategyColour(last_strategy: string): string {
  if (last_strategy.startsWith("sweep_detour:")) return "#f59e0b"; // amber
  if (last_strategy.startsWith("simple_reverse")) return "#ef4444"; // red
  return "#e2e8f0"; // slate-200 / white
}

export function ReplayOverlays() {
  const active = useReplayStore((s) => s.active);
  const ticks = useReplayStore((s) => s.ticks);
  const cursor = useReplayStore((s) => s.cursor);
  const projectTofFallback = useReplayStore((s) => s.projectTofFallback);

  const tick = ticks[cursor];

  // Obstacle cloud — prefer logged world_points (new-format logs). For
  // old-format logs (no geometry field), optionally fall back to
  // client-side projection of tof.grids through the URDF chain so the
  // user can still see what the arm "saw". All mm → m + shoulder-
  // offset so the spheres land at the right height in the scene.
  const obstacleSpheres = useMemo<Vec3[]>(() => {
    if (!tick) return [];
    let pointsMm: Array<[number, number, number]> = tick.world_points;
    if (pointsMm.length === 0 && projectTofFallback &&
        tick.tof.grids.length > 0) {
      pointsMm = projectTickTofToWorld(
        tick.joints,
        tick.tof.grids,
        tick.tof.thresholds,
      );
    }
    return pointsMm.map(([x, y, z]) => [
      x / 1000,
      y / 1000,
      z / 1000,
    ]);
  }, [tick, projectTofFallback]);

  // Detour path polyline — IK-project each cached waypoint to a tip
  // XYZ. Skipped when no detour is active this tick.
  const detourPolyline = useMemo<Vec3[]>(() => {
    if (!tick || tick.detour_path.length === 0) return [];
    const tipNow = fkTipXYZ(tick.joints);
    const pts: Vec3[] = [tipNow];
    for (const q of tick.detour_path) {
      const fix = [
        q[0] ?? 90, q[1] ?? 90, q[2] ?? 90,
        q[3] ?? 90, q[4] ?? 90, q[5] ?? 73,
      ] as unknown as JointVector;
      pts.push(fkTipXYZ(fix));
    }
    return pts;
  }, [tick]);

  // Rolling tip trail — last TRAIL_MAX ticks behind the cursor.
  const trailSegments = useMemo(() => {
    if (!tick) return [] as Array<{ from: Vec3; to: Vec3; color: string }>;
    const start = Math.max(0, cursor - TRAIL_MAX);
    const segs: Array<{ from: Vec3; to: Vec3; color: string }> = [];
    let prevTip: Vec3 | null = null;
    for (let i = start; i <= cursor; i++) {
      const t = ticks[i];
      const tip = fkTipXYZ(t.joints);
      if (prevTip !== null) {
        segs.push({
          from: prevTip,
          to: tip,
          color: strategyColour(t.bt.last_strategy),
        });
      }
      prevTip = tip;
    }
    return segs;
  }, [ticks, cursor, tick]);

  // Sweep-target line — from current tip to the nominal-line goal.
  const sweepTargetLine = useMemo<Vec3[]>(() => {
    if (!tick) return [];
    const tipNow = fkTipXYZ(tick.joints);
    const base = (tick.sweep.target_theta * Math.PI) / 180;
    const goal: Vec3 = [
      SWEEP_R_M * Math.cos(base),
      SWEEP_R_M * Math.sin(base),
      SWEEP_Z_M + SHOULDER_Z,
    ];
    return [tipNow, goal];
  }, [tick]);

  if (!active || !tick) return null;

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* Obstacle points */}
      {obstacleSpheres.map((p, i) => (
        <mesh key={`obs-${i}`} position={p}>
          <sphereGeometry args={[0.012, 12, 12]} />
          <meshStandardMaterial
            color="#ef4444"
            emissive="#7f1d1d"
            emissiveIntensity={0.4}
          />
        </mesh>
      ))}

      {/* Detour path polyline */}
      {detourPolyline.length >= 2 && (
        <Line
          points={detourPolyline}
          color="#f59e0b"
          lineWidth={3}
          dashed={false}
        />
      )}

      {/* Tip trail */}
      {trailSegments.map((s, i) => (
        <Line
          key={`trail-${i}`}
          points={[s.from, s.to]}
          color={s.color}
          lineWidth={2}
          transparent
          opacity={0.85}
        />
      ))}

      {/* Sweep-target line */}
      {sweepTargetLine.length === 2 && (
        <Line
          points={sweepTargetLine}
          color="#22c55e"
          lineWidth={1}
          dashed
          dashSize={0.012}
          gapSize={0.008}
        />
      )}
    </group>
  );
}
