// urdf.ts — Minimal URDF parser that extracts the Braccio arm kinematics.
//
// The Three.js viewer needs link origins and joint axes to place its
// meshes, and the production backend serves the hand-authored
// braccio.urdf at /urdf. We don't need a full URDF implementation —
// just a parser that can find every <joint>, read its origin xyz/rpy,
// axis, type, and parent/child link names. urdf-loaders is overkill
// for primitive-only geometry and pulls in a mesh loader we don't use.

/** A parsed, revolute-or-fixed URDF joint (Braccio has no prismatic). */
export interface UrdfJoint {
  name: string;
  type: "revolute" | "fixed" | "prismatic" | "continuous";
  parent: string;
  child: string;
  origin: {
    xyz: [number, number, number];
    rpy: [number, number, number];
  };
  axis: [number, number, number];
  limit?: {
    lower: number;
    upper: number;
  };
}

export interface UrdfRobot {
  name: string;
  joints: UrdfJoint[];
}

const WS_RE = /\s+/;

function parseXyz(raw: string | null): [number, number, number] {
  if (raw === null) return [0, 0, 0];
  const parts = raw.trim().split(WS_RE).map(Number);
  return [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0];
}

/**
 * Parse a braccio-style URDF string.
 *
 * The parser accepts the narrow subset used by the hand-authored
 * URDF (one <origin>, one <axis>, optional <limit>). It does not
 * resolve <include> directives, materials, or mesh references.
 */
export function parseBraccioUrdf(xmlText: string): UrdfRobot {
  const doc = new DOMParser().parseFromString(xmlText, "application/xml");
  const errorNode = doc.getElementsByTagName("parsererror")[0];
  if (errorNode) {
    throw new Error("urdf: xml parse failed");
  }
  const robot = doc.getElementsByTagName("robot")[0];
  if (!robot) {
    throw new Error("urdf: no <robot> element");
  }
  const jointEls = Array.from(robot.getElementsByTagName("joint"));
  const joints: UrdfJoint[] = [];
  for (const el of jointEls) {
    const name = el.getAttribute("name") ?? "";
    const typeRaw = (el.getAttribute("type") ?? "fixed") as UrdfJoint["type"];
    const parent =
      el.getElementsByTagName("parent")[0]?.getAttribute("link") ?? "";
    const child =
      el.getElementsByTagName("child")[0]?.getAttribute("link") ?? "";
    const originEl = el.getElementsByTagName("origin")[0];
    const origin = {
      xyz: parseXyz(originEl?.getAttribute("xyz") ?? null),
      rpy: parseXyz(originEl?.getAttribute("rpy") ?? null),
    };
    const axisEl = el.getElementsByTagName("axis")[0];
    const axis = axisEl
      ? parseXyz(axisEl.getAttribute("xyz"))
      : ([1, 0, 0] as [number, number, number]);
    const limitEl = el.getElementsByTagName("limit")[0];
    const limit = limitEl
      ? {
          lower: Number(limitEl.getAttribute("lower") ?? "0"),
          upper: Number(limitEl.getAttribute("upper") ?? "0"),
        }
      : undefined;
    joints.push({ name, type: typeRaw, parent, child, origin, axis, limit });
  }
  return {
    name: robot.getAttribute("name") ?? "robot",
    joints,
  };
}

/**
 * Map the Braccio servo angle (degrees, 0..180) into the URDF joint
 * angle (radians) for one of the 6 movable joints.
 *
 * The URDF is hand-authored to agree with the real arm's "home" pose
 * at Braccio 90° — same convention the Python sim uses. Keeping the
 * mapping in TypeScript means the viewer doesn't need to ask the
 * backend to rewrite every frame.
 */
export function braccioDegreesToUrdfRadians(
  jointName: string,
  servoDeg: number,
): number {
  const rad = (deg: number) => (deg * Math.PI) / 180;
  switch (jointName) {
    case "joint_base":
      // 0..180 -> -π/2..+π/2, centred at 90°
      return rad(servoDeg - 90);
    case "joint_shoulder":
      // 15..165 -> -2.8798..-0.2618 (URDF lower..upper).
      // At servo 90 the URDF joint is -π/2 (arm straight up).
      return rad(servoDeg - 90) - Math.PI / 2;
    case "joint_elbow":
      // 0..180 -> 0..π. URDF joint is zero when arm is in line.
      return rad(servoDeg);
    case "joint_wrist_vert":
      return rad(servoDeg - 90);
    case "joint_wrist_rot":
      return rad(servoDeg - 90);
    case "joint_gripper":
      // Prismatic/gripper is ignored by the viewer for Phase 4.
      return 0;
    default:
      return 0;
  }
}
