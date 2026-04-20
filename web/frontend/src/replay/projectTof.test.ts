// projectTof.test.ts — sanity-check the client-side ToF projector.

import { describe, expect, it } from "vitest";
import {
  projectGridToWorldPoints,
  projectTickTofToWorld,
} from "./projectTof";
import type { JointVector } from "../state/types";

const HOME: JointVector = [90, 90, 90, 90, 90, 73] as unknown as JointVector;

describe("projectGridToWorldPoints", () => {
  it("returns no points when all cells are beyond threshold", () => {
    const grid = [
      [1000, 1000, 1000, 1000],
      [1000, 1000, 1000, 1000],
    ];
    expect(projectGridToWorldPoints(grid, 0, HOME, 250)).toHaveLength(0);
  });

  it("returns a point for each sub-threshold cell", () => {
    const grid = [
      [80, 1000, 1000, 1000],
      [1000, 1000, 1000, 1000],
    ];
    const pts = projectGridToWorldPoints(grid, 0, HOME, 250);
    expect(pts).toHaveLength(1);
    const [x, y, z] = pts[0];
    expect(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z))
      .toBe(true);
  });

  it("produces differently-oriented points for different channels", () => {
    // Single close cell at the centre of each channel's grid.
    const centreCell = [
      [50, 50],
      [50, 50],
    ];
    const up = projectGridToWorldPoints(centreCell, 0, HOME, 250);
    const right = projectGridToWorldPoints(centreCell, 1, HOME, 250);
    const left = projectGridToWorldPoints(centreCell, 2, HOME, 250);
    const down = projectGridToWorldPoints(centreCell, 3, HOME, 250);
    // Channels 0/1/2/3 point up/right/left/down respectively, so the
    // returned points must differ pairwise in at least one axis — if
    // the mount rotations were broken they'd collapse to the same xyz.
    expect(up).not.toEqual(right);
    expect(left).not.toEqual(right);
    expect(up).not.toEqual(down);
  });

  it("ignores invalid cells (NaN / zero / negative)", () => {
    const grid = [
      [NaN, 0, -5, 80],
    ];
    const pts = projectGridToWorldPoints(grid, 0, HOME, 250);
    expect(pts).toHaveLength(1);
  });
});

describe("projectTickTofToWorld", () => {
  it("skips channel 3 (ground) by default", () => {
    const grids = [
      [[1000]], [[1000]], [[1000]],
      [[10]],       // channel 3 very close
    ];
    const thresholds = [250, 250, 50, 50];
    expect(projectTickTofToWorld(HOME, grids, thresholds)).toHaveLength(0);
  });

  it("projects channel 3 when explicitly requested", () => {
    const grids = [
      [[1000]], [[1000]], [[1000]], [[40]],
    ];
    const thresholds = [250, 250, 50, 50];
    expect(
      projectTickTofToWorld(HOME, grids, thresholds, [3]),
    ).toHaveLength(1);
  });
});
