// Arm.tsx — Procedural Braccio arm fed by the telemetry store.
//
// Rather than running urdf-loaders (which would add a DOMParser
// runtime dependency and doesn't help us since the URDF uses
// primitive shapes anyway), we construct the arm from R3F groups
// whose origins, rotations, and lengths mirror braccio.urdf
// line-for-line. The joint angles come from the zustand store and
// feed through the same servo→URDF mapping the PyBullet sim uses,
// so the on-screen pose matches the simulated pose exactly.

import { useRef } from "react";
import type { Group } from "three";
import { useFrame } from "@react-three/fiber";
import { useTelemetryStore } from "../state/telemetry";
import { braccioDegreesToUrdfRadians } from "../state/urdf";

// Link lengths in metres — identical to braccio.urdf (which in turn
// mirrors braccio_ctrl/constants.py L1 = L2 = 125 mm, L3 = 60 mm).
const L1 = 0.125;
const L2 = 0.125;
const L3 = 0.060;

export function Arm() {
  // Pull the latest joints every frame via refs so React doesn't
  // rerender on every WS message. This is the standard r3f pattern
  // for high-frequency state.
  const baseRef = useRef<Group>(null);
  const shoulderRef = useRef<Group>(null);
  const elbowRef = useRef<Group>(null);
  const wristVertRef = useRef<Group>(null);
  const wristRotRef = useRef<Group>(null);

  useFrame(() => {
    const js = useTelemetryStore.getState().joints;
    if (baseRef.current) {
      baseRef.current.rotation.y = braccioDegreesToUrdfRadians(
        "joint_base",
        js[0],
      );
    }
    if (shoulderRef.current) {
      shoulderRef.current.rotation.z = -braccioDegreesToUrdfRadians(
        "joint_shoulder",
        js[1],
      );
    }
    if (elbowRef.current) {
      elbowRef.current.rotation.z = braccioDegreesToUrdfRadians(
        "joint_elbow",
        js[2],
      );
    }
    if (wristVertRef.current) {
      wristVertRef.current.rotation.z = braccioDegreesToUrdfRadians(
        "joint_wrist_vert",
        js[3],
      );
    }
    if (wristRotRef.current) {
      wristRotRef.current.rotation.x = braccioDegreesToUrdfRadians(
        "joint_wrist_rot",
        js[4],
      );
    }
  });

  return (
    // We render with Y-up (r3f convention). The URDF uses Z-up, so
    // the top-level group rotates +90° around X to line up Y↔Z. After
    // this rotation, every child can use the URDF's (x, y, z) layout
    // directly while R3F sees Y as the vertical axis.
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {/* base_link — solid box sitting on the table */}
      <mesh position={[0, 0, 0.030]} castShadow receiveShadow>
        <boxGeometry args={[0.110, 0.110, 0.060]} />
        <meshStandardMaterial color="#3f3f3f" />
      </mesh>

      {/* joint_base: revolute Z at top of base */}
      <group ref={baseRef} position={[0, 0, 0.060]}>
        {/* shoulder_pillar_link */}
        <mesh position={[0, 0, 0.005]} castShadow>
          <cylinderGeometry args={[0.035, 0.035, 0.010, 24]} />
          <meshStandardMaterial color="#1a59c0" />
        </mesh>

        {/* joint_shoulder: revolute Y (rendered as Z-rotate after
            the top-level rotation). We nest a Y-rotation group and a
            cylinder visual of length L1 along the local +X. */}
        <group position={[0, 0, 0.010]} ref={shoulderRef}>
          {/* upper_arm_link */}
          <mesh
            position={[L1 / 2, 0, 0]}
            rotation={[0, 0, Math.PI / 2]}
            castShadow
          >
            <cylinderGeometry args={[0.018, 0.018, L1, 16]} />
            <meshStandardMaterial color="#d97320" />
          </mesh>

          {/* joint_elbow: revolute Y at the end of the upper arm */}
          <group position={[L1, 0, 0]} ref={elbowRef}>
            <mesh
              position={[L2 / 2, 0, 0]}
              rotation={[0, 0, Math.PI / 2]}
              castShadow
            >
              <cylinderGeometry args={[0.016, 0.016, L2, 16]} />
              <meshStandardMaterial color="#e68c26" />
            </mesh>

            {/* joint_wrist_vert at the end of the forearm */}
            <group position={[L2, 0, 0]} ref={wristVertRef}>
              <mesh
                position={[0.015, 0, 0]}
                rotation={[0, 0, Math.PI / 2]}
                castShadow
              >
                <cylinderGeometry args={[0.022, 0.022, 0.030, 16]} />
                <meshStandardMaterial color="#343439" />
              </mesh>

              {/* joint_wrist_rot — rotation around local +X */}
              <group position={[0.030, 0, 0]} ref={wristRotRef}>
                {/* wrist_roll_link — the "hand" that carries the 4 ToF sensors */}
                <mesh position={[0.010, 0, 0]} castShadow>
                  <boxGeometry args={[0.030, 0.040, 0.040]} />
                  <meshStandardMaterial color="#26262e" />
                </mesh>

                {/* gripper_link (fixed extension of L3 past the wrist) */}
                <mesh
                  position={[L3, 0, 0]}
                  rotation={[0, 0, Math.PI / 2]}
                  castShadow
                >
                  <cylinderGeometry args={[0.012, 0.012, 0.040, 16]} />
                  <meshStandardMaterial color="#b3b3bb" />
                </mesh>
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}
