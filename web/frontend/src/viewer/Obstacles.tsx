// Obstacles.tsx — World-frame obstacle markers.
//
// For Phase 4 the frontend does not yet maintain its own obstacle
// list; the backend's SimBackend (behind a Phase 7 toggle) is the
// source of truth. This component accepts a ``spheres`` prop so
// parent code can feed it from whatever source it wants — the
// ObstacleWorld JSON, a Blockly DEFINE OBSTACLE block, or a
// hand-authored test list.

export interface ObstacleSphere {
  id: string;
  x: number;
  y: number;
  z: number;
  radius: number;
}

interface ObstaclesProps {
  spheres?: ObstacleSphere[];
}

export function Obstacles({ spheres = [] }: ObstaclesProps) {
  return (
    <group>
      {spheres.map((s) => (
        <mesh key={s.id} position={[s.x, s.z, -s.y]} castShadow>
          <sphereGeometry args={[s.radius, 16, 16]} />
          <meshStandardMaterial
            color="#f43f5e"
            transparent
            opacity={0.55}
          />
        </mesh>
      ))}
    </group>
  );
}
