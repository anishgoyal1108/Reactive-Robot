// urdf.test.ts — Unit tests for the URDF parser + joint mapping.

import { describe, expect, it } from "vitest";
import { braccioDegreesToUrdfRadians, parseBraccioUrdf } from "./urdf";

const SAMPLE = `<?xml version="1.0"?>
<robot name="braccio_v2">
  <joint name="joint_base" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_pillar_link"/>
    <origin xyz="0 0 0.060" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5708" upper="1.5708" effort="5" velocity="3.14"/>
  </joint>
  <joint name="joint_elbow" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="forearm_link"/>
    <origin xyz="0.125 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="3.14159" effort="5" velocity="3.14"/>
  </joint>
  <joint name="mount_tof_up" type="fixed">
    <parent link="wrist_roll_link"/>
    <child link="tof_up_link"/>
    <origin xyz="0.010 0 0.020" rpy="0 -1.5708 0"/>
  </joint>
</robot>`;

describe("parseBraccioUrdf", () => {
  it("extracts all joints", () => {
    const robot = parseBraccioUrdf(SAMPLE);
    expect(robot.name).toBe("braccio_v2");
    expect(robot.joints).toHaveLength(3);
    const names = robot.joints.map((j) => j.name);
    expect(names).toContain("joint_base");
    expect(names).toContain("joint_elbow");
    expect(names).toContain("mount_tof_up");
  });

  it("parses origin xyz/rpy and axis", () => {
    const robot = parseBraccioUrdf(SAMPLE);
    const base = robot.joints.find((j) => j.name === "joint_base");
    expect(base).toBeDefined();
    expect(base!.origin.xyz).toEqual([0, 0, 0.06]);
    expect(base!.axis).toEqual([0, 0, 1]);
  });

  it("parses limits when present and omits them for fixed joints", () => {
    const robot = parseBraccioUrdf(SAMPLE);
    const base = robot.joints.find((j) => j.name === "joint_base");
    expect(base!.limit).toBeDefined();
    expect(base!.limit!.lower).toBeCloseTo(-1.5708, 3);
    expect(base!.limit!.upper).toBeCloseTo(1.5708, 3);
    const mount = robot.joints.find((j) => j.name === "mount_tof_up");
    expect(mount!.limit).toBeUndefined();
    expect(mount!.type).toBe("fixed");
  });

  it("throws on unparseable XML", () => {
    expect(() => parseBraccioUrdf("<robot name='x'")).toThrow();
  });
});

describe("braccioDegreesToUrdfRadians", () => {
  it("maps joint_base 90° to 0 rad", () => {
    expect(braccioDegreesToUrdfRadians("joint_base", 90)).toBeCloseTo(0, 6);
  });

  it("maps joint_base 0° to -π/2 rad", () => {
    expect(braccioDegreesToUrdfRadians("joint_base", 0)).toBeCloseTo(
      -Math.PI / 2,
      6,
    );
  });

  it("maps joint_shoulder 90° to -π/2 rad (arm vertical)", () => {
    expect(braccioDegreesToUrdfRadians("joint_shoulder", 90)).toBeCloseTo(
      -Math.PI / 2,
      6,
    );
  });

  it("maps joint_elbow 0° to 0 rad", () => {
    expect(braccioDegreesToUrdfRadians("joint_elbow", 0)).toBeCloseTo(0, 6);
  });

  it("maps joint_elbow 180° to π rad", () => {
    expect(braccioDegreesToUrdfRadians("joint_elbow", 180)).toBeCloseTo(
      Math.PI,
      6,
    );
  });

  it("returns 0 for unknown joint names", () => {
    expect(braccioDegreesToUrdfRadians("joint_nonexistent", 45)).toBe(0);
  });
});
