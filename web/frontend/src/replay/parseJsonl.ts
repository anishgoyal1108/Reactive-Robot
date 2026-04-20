// parseJsonl.ts — Session-log line parser for the Replay tab.
//
// The Python session logger writes one JSON object per line. Headers
// start with {"type":"header",...} and are skipped. Every other line is
// a tick record whose fields match session_logger.py::_emit_sample.
//
// Older logs may not carry world_points / detour_path /
// last_strategy_entry_tick — those are filled with empty defaults so
// downstream overlays don't have to null-check each tick.

import type { JointVector } from "../state/types";

export interface SessionTick {
  t: number;
  t_rel: number;
  /** 6-DOF servo degrees [B, S, E, WV, WR, G]. */
  joints: JointVector;
  theta: number;
  r: number;
  z: number;
  tof: {
    ch_min_mm: number[];
    thresholds: number[];
    active: number[];
    /** Per-channel N×N distance grids (mm); empty when the firmware
     *  didn't publish or the logger ran without a live Teensy. Needed
     *  for client-side ToF→world projection on old logs without
     *  `world_points`. */
    grids: number[][][];
  };
  obstacle: {
    response: string;
    source: string;
    dist_mm: number;
  };
  bt: {
    mode: string;
    state: string;
    emergency: boolean;
    last_strategy: string;
    last_failure: string | null;
    polar_blocked: Array<[number, number]>;
  };
  sweep: {
    direction: number;
    target_theta: number;
    running: boolean;
  };
  /** Optional — present when SESSION_LOG_INCLUDE_GEOMETRY=True. */
  world_points: Array<[number, number, number]>;
  detour_path: number[][];
  last_strategy_entry_tick: boolean;
}

function _fallbackTick(): SessionTick {
  return {
    t: 0,
    t_rel: 0,
    joints: [90, 90, 90, 90, 90, 73] as unknown as JointVector,
    theta: 90,
    r: 0,
    z: 0,
    tof: { ch_min_mm: [], thresholds: [], active: [], grids: [] },
    obstacle: { response: "clear", source: "", dist_mm: -1 },
    bt: {
      mode: "idle",
      state: "",
      emergency: false,
      last_strategy: "",
      last_failure: null,
      polar_blocked: [],
    },
    sweep: { direction: 1, target_theta: 90, running: false },
    world_points: [],
    detour_path: [],
    last_strategy_entry_tick: false,
  };
}

function _coerceJoints(input: unknown): JointVector {
  const arr = Array.isArray(input) ? input : [];
  const out: number[] = [];
  for (let i = 0; i < 6; i++) {
    const v = Number(arr[i]);
    out.push(Number.isFinite(v) ? v : 90);
  }
  return out as unknown as JointVector;
}

function _coerceNumberArray(input: unknown): number[] {
  if (!Array.isArray(input)) return [];
  return input.map((v) => (Number.isFinite(Number(v)) ? Number(v) : 0));
}

function _coerceGrids(input: unknown): number[][][] {
  if (!Array.isArray(input)) return [];
  const out: number[][][] = [];
  for (const ch of input) {
    if (!Array.isArray(ch)) continue;
    const rows: number[][] = [];
    for (const row of ch) {
      if (!Array.isArray(row)) continue;
      rows.push(row.map((v) => Number(v)));
    }
    out.push(rows);
  }
  return out;
}

function _coerceWorldPoints(input: unknown): Array<[number, number, number]> {
  if (!Array.isArray(input)) return [];
  const out: Array<[number, number, number]> = [];
  for (const p of input) {
    if (!Array.isArray(p) || p.length < 3) continue;
    out.push([Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0]);
  }
  return out;
}

function _coerceDetourPath(input: unknown): number[][] {
  if (!Array.isArray(input)) return [];
  const out: number[][] = [];
  for (const q of input) {
    if (!Array.isArray(q)) continue;
    out.push(q.map((v) => Math.round(Number(v) || 0)));
  }
  return out;
}

function _parseOne(obj: Record<string, unknown>): SessionTick | null {
  if (obj.type === "header") return null;
  if (!("joints" in obj)) return null;
  const base = _fallbackTick();
  const tof = (obj.tof as Record<string, unknown>) ?? {};
  const bt = (obj.bt as Record<string, unknown>) ?? {};
  const sweep = (obj.sweep as Record<string, unknown>) ?? {};
  const obstacle = (obj.obstacle as Record<string, unknown>) ?? {};
  return {
    t: Number(obj.t) || 0,
    t_rel: Number(obj.t_rel) || 0,
    joints: _coerceJoints(obj.joints),
    theta: Number(obj.theta) || 0,
    r: Number(obj.r) || 0,
    z: Number(obj.z) || 0,
    tof: {
      ch_min_mm: _coerceNumberArray(tof.ch_min_mm),
      thresholds: _coerceNumberArray(tof.thresholds),
      active: _coerceNumberArray(tof.active),
      grids: _coerceGrids(tof.grids),
    },
    obstacle: {
      response: String(obstacle.response ?? "clear"),
      source: String(obstacle.source ?? ""),
      dist_mm: Number(obstacle.dist_mm ?? -1),
    },
    bt: {
      mode: String(bt.mode ?? "idle"),
      state: String(bt.state ?? ""),
      emergency: Boolean(bt.emergency),
      last_strategy: String(bt.last_strategy ?? ""),
      last_failure:
        bt.last_failure == null ? null : String(bt.last_failure),
      polar_blocked: Array.isArray(bt.polar_blocked)
        ? (bt.polar_blocked as Array<[number, number]>)
        : base.bt.polar_blocked,
    },
    sweep: {
      direction: Number(sweep.direction) || 1,
      target_theta: Number(sweep.target_theta) || 0,
      running: Boolean(sweep.running),
    },
    world_points: _coerceWorldPoints(obj.world_points),
    detour_path: _coerceDetourPath(obj.detour_path),
    last_strategy_entry_tick: Boolean(obj.last_strategy_entry_tick),
  };
}

/** Parse a whole JSONL file blob into an ordered array of ticks.
 *  Malformed / header lines are silently skipped. */
export function parseSessionJsonl(text: string): SessionTick[] {
  const out: SessionTick[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.length === 0) continue;
    let obj: Record<string, unknown>;
    try {
      obj = JSON.parse(line);
    } catch {
      continue;
    }
    const tick = _parseOne(obj);
    if (tick) out.push(tick);
  }
  return out;
}
