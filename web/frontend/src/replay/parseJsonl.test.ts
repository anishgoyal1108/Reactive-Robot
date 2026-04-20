// parseJsonl.test.ts — Round-trip a hand-written JSONL sample.

import { describe, expect, it } from "vitest";
import { parseSessionJsonl } from "./parseJsonl";

const HEADER = JSON.stringify({
  type: "header",
  t: 1776400000,
  hz: 10,
  format: "jsonl/1",
});

const TICK_NEW = JSON.stringify({
  t: 1776400001,
  t_rel: 0.1,
  joints: [92, 88, 85, 0, 90, 73],
  theta: 92.0,
  r: 152.0,
  z: 35.0,
  tof: {
    ch_min_mm: [700, 750, 620, 80],
    thresholds: [100, 100, 50, 50],
    active: [1, 1, 1, 1],
    hz: [0, 0, 0, 0],
    grids: [],
  },
  ir: { bits: 0, label: "DISABLED", action: "" },
  obstacle: { response: "clear", source: "", dist_mm: -1 },
  bt: {
    mode: "sweep",
    state: "sweep_running",
    emergency: false,
    last_strategy: "sweep_detour:pull_in",
    last_failure: null,
    polar_blocked: [[95, 115]],
    queue_length: 0,
  },
  world: { num_points: 2, oldest_age_s: 0, grid_cells_occ: 0, grid_cells_free: 0 },
  sweep: { running: true, direction: 1, target_theta: 120 },
  arm_io: { last_cmd: "", last_resp: "", last_error: "" },
  events: [],
  world_points: [[150.0, 60.0, 35.0], [155.0, 62.0, 32.0]],
  detour_path: [
    [94, 88, 85, 0, 90, 73],
    [96, 88, 85, 0, 90, 73],
  ],
  last_strategy_entry_tick: true,
});

const TICK_OLD = JSON.stringify({
  t: 1776400002,
  t_rel: 0.2,
  joints: [93, 88, 85, 0, 90, 73],
  theta: 93.0,
  r: 152.0,
  z: 35.0,
  tof: { ch_min_mm: [], thresholds: [], active: [] },
  obstacle: { response: "clear", source: "", dist_mm: -1 },
  bt: {
    mode: "sweep",
    state: "sweep_running",
    emergency: false,
    last_strategy: "simple_direct",
    last_failure: null,
    polar_blocked: [],
  },
  sweep: { running: true, direction: 1, target_theta: 95 },
});

describe("parseSessionJsonl", () => {
  it("parses new-format tick with geometry fields", () => {
    const ticks = parseSessionJsonl(`${HEADER}\n${TICK_NEW}\n`);
    expect(ticks).toHaveLength(1);
    expect(ticks[0].joints).toEqual([92, 88, 85, 0, 90, 73]);
    expect(ticks[0].world_points).toHaveLength(2);
    expect(ticks[0].detour_path).toHaveLength(2);
    expect(ticks[0].last_strategy_entry_tick).toBe(true);
    expect(ticks[0].bt.last_strategy).toBe("sweep_detour:pull_in");
  });

  it("accepts old-format tick with missing world_points/detour_path", () => {
    const ticks = parseSessionJsonl(TICK_OLD);
    expect(ticks).toHaveLength(1);
    expect(ticks[0].world_points).toEqual([]);
    expect(ticks[0].detour_path).toEqual([]);
    expect(ticks[0].last_strategy_entry_tick).toBe(false);
  });

  it("skips header lines and malformed JSON", () => {
    const blob = `${HEADER}\nnot-json-at-all\n${TICK_NEW}\n\n${TICK_OLD}\n`;
    const ticks = parseSessionJsonl(blob);
    expect(ticks).toHaveLength(2);
  });
});
