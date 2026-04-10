// SensorRays.test.ts — Pure colour-mapping tests.
//
// The visual mesh rendering lives under @react-three/fiber's
// <Canvas>, which requires a WebGL context. We test the pure
// threshold logic in isolation so Phase 4 has real coverage for the
// ToF/IR severity decisions without needing a GPU.

import { describe, expect, it } from "vitest";
import { _internal } from "./SensorRays";

const { tofColor, irColor, BLOCK_THRESHOLD_MM } = _internal;

describe("tofColor", () => {
  it("is green when the reading is well above threshold", () => {
    expect(tofColor(BLOCK_THRESHOLD_MM + 100)).toBe("#22c55e");
    expect(tofColor(9999)).toBe("#22c55e");
  });

  it("is yellow inside the warning band", () => {
    // Below BLOCK_THRESHOLD_MM but above half of it.
    expect(tofColor(BLOCK_THRESHOLD_MM - 10)).toBe("#facc15");
  });

  it("is red inside the danger band", () => {
    expect(tofColor(BLOCK_THRESHOLD_MM * 0.3)).toBe("#ef4444");
    expect(tofColor(10)).toBe("#ef4444");
  });
});

describe("irColor", () => {
  it("is green for severity 0 (clear)", () => {
    expect(irColor(0)).toBe("#22c55e");
  });

  it("is yellow for severity 1 (far)", () => {
    expect(irColor(1)).toBe("#facc15");
  });

  it("is orange for severity 2 (close)", () => {
    expect(irColor(2)).toBe("#f97316");
  });

  it("is red for severity 3 (danger)", () => {
    expect(irColor(3)).toBe("#ef4444");
  });
});
