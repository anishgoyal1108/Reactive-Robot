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

// Gripper servo range — Arduino Braccio convention: 10° = closed,
// 73° = wide open. We linearly interpolate each finger to a rotation
// inside [0, GRIPPER_MAX_ANGLE] so the two fingers scissor apart.
const GRIPPER_CLOSED_SERVO = 10;
const GRIPPER_OPEN_SERVO = 73;
const GRIPPER_MAX_ANGLE = Math.PI / 4;

export function Arm() {
  // Pull the latest joints every frame via refs so React doesn't
  // rerender on every WS message. This is the standard r3f pattern
  // for high-frequency state.
  const baseRef = useRef<Group>(null);
  const shoulderRef = useRef<Group>(null);
  const elbowRef = useRef<Group>(null);
  const wristVertRef = useRef<Group>(null);
  const wristRotRef = useRef<Group>(null);
  const leftFingerRef = useRef<Group>(null);
  const rightFingerRef = useRef<Group>(null);

  useFrame(() => {
    const js = useTelemetryStore.getState().joints;
    // URDF joint axes (see braccio.urdf):
    //   joint_base       → Z   (vertical "twist" of the whole arm)
    //   joint_shoulder   → Y
    //   joint_elbow      → Y
    //   joint_wrist_vert → Y
    //   joint_wrist_rot  → X
    // The outer <group rotation={[-π/2, 0, 0]}> below rotates the
    // whole URDF frame so URDF +Z → world +Y, but inside that group
    // every child still works in URDF-local coordinates. That means
    // the LOCAL rotation axes used here map directly to the URDF
    // joint axes — base spins around its own Z, shoulder/elbow/
    // wrist_vert all hinge around Y, and wrist_rot rolls around X.
    if (baseRef.current) {
      baseRef.current.rotation.z = braccioDegreesToUrdfRadians(
        "joint_base",
        js[0],
      );
    }
    if (shoulderRef.current) {
      shoulderRef.current.rotation.y = braccioDegreesToUrdfRadians(
        "joint_shoulder",
        js[1],
      );
    }
    if (elbowRef.current) {
      elbowRef.current.rotation.y = braccioDegreesToUrdfRadians(
        "joint_elbow",
        js[2],
      );
    }
    if (wristVertRef.current) {
      wristVertRef.current.rotation.y = braccioDegreesToUrdfRadians(
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
    // Gripper: linearly interpolate per-finger rotation from the
    // gripper servo value. The two fingers mirror each other around
    // the wrist axis so they "scissor" open as the servo opens.
    const gripperServo = js[5] ?? GRIPPER_CLOSED_SERVO;
    const t = Math.max(
      0,
      Math.min(
        1,
        (gripperServo - GRIPPER_CLOSED_SERVO) /
          (GRIPPER_OPEN_SERVO - GRIPPER_CLOSED_SERVO),
      ),
    );
    const fingerAngle = t * GRIPPER_MAX_ANGLE;
    if (leftFingerRef.current) {
      leftFingerRef.current.rotation.z = fingerAngle;
    }
    if (rightFingerRef.current) {
      rightFingerRef.current.rotation.z = -fingerAngle;
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
        {/* shoulder_pillar_link — cylinderGeometry is Y-axis by
            default; rotate +π/2 around X so its height aligns with
            the parent group's +Z (which maps to world +Y "up" via the
            top-level frame rotation). Without this the disc ends up
            lying horizontally instead of standing on the platform. */}
        <mesh
          position={[0, 0, 0.005]}
          rotation={[Math.PI / 2, 0, 0]}
          castShadow
        >
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

                {/* Gripper assembly — replaces the URDF's placeholder
                    cylinder with a palm box + two pivoting finger
                    boxes so the visual mirrors the real Braccio claw
                    and actually opens/closes from the gripper servo
                    angle. The palm fills the back ~⅔ of the URDF
                    L3 stub; the fingers extend from the palm front
                    out to L3 and pivot around their local +Z (URDF
                    up) so they scissor apart symmetrically. */}
                {(() => {
                  const PALM_LENGTH = L3 * 0.4; // 24 mm
                  const FINGER_LENGTH = L3 - PALM_LENGTH; // 36 mm
                  return (
                    <>
                      <mesh
                        position={[PALM_LENGTH / 2, 0, 0]}
                        castShadow
                      >
                        <boxGeometry args={[PALM_LENGTH, 0.036, 0.024]} />
                        <meshStandardMaterial color="#8d8d94" />
                      </mesh>

                      {/* left finger — hinges at the front-left
                          corner of the palm. */}
                      <group
                        position={[PALM_LENGTH, 0.014, 0]}
                        ref={leftFingerRef}
                      >
                        <mesh
                          position={[FINGER_LENGTH / 2, 0.004, 0]}
                          castShadow
                        >
                          <boxGeometry args={[FINGER_LENGTH, 0.008, 0.020]} />
                          <meshStandardMaterial color="#cfcfd4" />
                        </mesh>
                      </group>

                      {/* right finger — mirror of the left. */}
                      <group
                        position={[PALM_LENGTH, -0.014, 0]}
                        ref={rightFingerRef}
                      >
                        <mesh
                          position={[FINGER_LENGTH / 2, -0.004, 0]}
                          castShadow
                        >
                          <boxGeometry args={[FINGER_LENGTH, 0.008, 0.020]} />
                          <meshStandardMaterial color="#cfcfd4" />
                        </mesh>
                      </group>
                    </>
                  );
                })()}
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}
