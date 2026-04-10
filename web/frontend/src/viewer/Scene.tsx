// Scene.tsx — Top-level react-three-fiber <Canvas> with lights + camera.
//
// The scene is deliberately plain so the Arm/SensorRays/Obstacles
// children own their own styling. That makes them easy to re-arrange
// as the UI grows. OrbitControls come from @react-three/drei so the
// user can click-drag-zoom with no extra code.

import type { ReactNode } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

interface SceneProps {
  children?: ReactNode;
}

export function Scene({ children }: SceneProps) {
  return (
    <Canvas
      camera={{ position: [0.5, 0.5, 0.6], fov: 45, near: 0.01, far: 10 }}
      shadows
      dpr={[1, 2]}
    >
      {/* Tri-light rig: one keylight, one soft fill, one rim. */}
      <ambientLight intensity={0.35} />
      <directionalLight
        position={[2, 3, 2]}
        intensity={1.1}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <directionalLight position={[-2, 2, -1]} intensity={0.25} />

      {/* Ground plane in the URDF +X/+Y plane (Z is up in the URDF,
          but r3f expects Y-up; we rotate the whole world once at the
          root so X/Y stay as-is horizontally). */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[0, 0, 0]}>
        <planeGeometry args={[1, 1]} />
        <meshStandardMaterial color="#1b2029" />
      </mesh>

      {/* Grid lines for depth cues. */}
      <gridHelper args={[1, 20, "#2a3441", "#1e2631"]} />

      {children}

      <OrbitControls
        makeDefault
        enableDamping
        target={[0, 0.12, 0]}
        maxPolarAngle={Math.PI / 2}
      />
    </Canvas>
  );
}
