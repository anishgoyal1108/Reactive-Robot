// SensorRays.tsx — Placeholder ToF + IR visualisations for Phase 4.
//
// The Phase 1 sim already reasons about 4 ToF cones (front/back/up/
// down) mounted on the wrist block and 4 IR pockets on the base. At
// this phase we draw coloured indicator meshes around the base whose
// colour reflects the latest telemetry (ToF distance thresholds and
// IR severity). Animating them to follow the arm requires live
// world-space transforms from the Arm component, which we leave for
// a future phase. This keeps the viewer functional while we move on
// to Blockly.

import { useTelemetryStore } from "../state/telemetry";

// Distance (mm) below which a ToF reading is considered "blocked".
// Matches braccio_ctrl.constants.TOF_THRESHOLDS_MM (CH0-2 = 100 mm,
// CH3 = 50 mm); we use the conservative side-channel value for the
// viewer so CH3 turns yellow around the same relative distance as the
// others without a per-channel colour ramp.
const BLOCK_THRESHOLD_MM = 100;

function tofColor(distanceMm: number): string {
  if (distanceMm < BLOCK_THRESHOLD_MM * 0.5) return "#ef4444";
  if (distanceMm < BLOCK_THRESHOLD_MM) return "#facc15";
  return "#22c55e";
}

function irColor(severity: number): string {
  if (severity >= 3) return "#ef4444";
  if (severity === 2) return "#f97316";
  if (severity === 1) return "#facc15";
  return "#22c55e";
}

/**
 * Four floating disks around the base showing the 4 ToF channels
 * and an IR-severity ring overlay. Updates pull directly from the
 * zustand store so they re-render as frames arrive.
 */
export function SensorRays() {
  const tof = useTelemetryStore((s) => s.tof);
  const ir = useTelemetryStore((s) => s.ir);

  // Channel → physical direction (matches ``safety/api.py`` _MOUNT_BY_CHANNEL
  // and the URDF tof_*_link definitions).
  //   CH0 = Top, CH1 = Right, CH2 = Left, CH3 = Bottom
  // World frame here is three.js: +X right, +Y up, +Z out of screen. So
  // top/bottom stack on Y, right/left on X, around the base.
  const TOF_LAYOUT: Array<{
    channel: number;
    position: [number, number, number];
    label: string;
  }> = [
    { channel: 0, position: [0.0, 0.15, 0.0],  label: "CH0 Top" },
    { channel: 1, position: [0.20, 0.07, 0.0], label: "CH1 Right" },
    { channel: 2, position: [-0.20, 0.07, 0.0], label: "CH2 Left" },
    { channel: 3, position: [0.0, -0.02, 0.0], label: "CH3 Bottom" },
  ];

  return (
    <group>
      {TOF_LAYOUT.map((l) => (
        <mesh key={l.channel} position={l.position}>
          <sphereGeometry args={[0.015, 16, 16]} />
          <meshStandardMaterial
            color={tofColor(tof[l.channel] ?? 9999)}
            emissive={tofColor(tof[l.channel] ?? 9999)}
            emissiveIntensity={0.4}
          />
        </mesh>
      ))}
      {/* IR severity ring — lays flat on the platform so it reads as
          a halo around the arm base rather than a vertical hula-hoop.
          torusGeometry defaults to the XY plane (normal +Z); rotating
          -π/2 around X puts it in the XZ plane (normal +Y = world up). */}
      <mesh position={[0, 0.062, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.080, 0.005, 8, 32]} />
        <meshStandardMaterial
          color={irColor(ir)}
          emissive={irColor(ir)}
          emissiveIntensity={0.3}
        />
      </mesh>
    </group>
  );
}

// Exported for tests so the pure colour-mapping logic has coverage
// without mounting the scene graph.
export const _internal = {
  tofColor,
  irColor,
  BLOCK_THRESHOLD_MM,
};
