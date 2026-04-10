// types.ts — Shared TypeScript types for frontend ↔ backend messages.
//
// These mirror the Pydantic schemas in web/backend/models.py. Keep
// them in sync; the Python side is the source of truth, but these
// types let the TS code enforce the same shape at compile time.

export type JointVector = readonly [
  number, number, number, number, number, number,
];

export interface StateEntry {
  name: string;
  joints: number[];
  theta: number;
  r: number;
  z: number;
  wrist_offset: number;
  wrist_rot: number;
  gripper: number;
}

export interface RunStatus {
  running: boolean;
  line: number;
  kind: string;
  depth: number;
  pass_current: number;
  pass_total: number;
  errors: string[];
}

/**
 * A single frame streamed over /ws/telemetry.
 *
 * Note: ToF keys come back as strings because JSON objects cannot
 * have integer keys. The frontend store normalises to Map<number,
 * number> at ingest time so UI code can use channel indices.
 */
export interface TelemetryFrame {
  type: "telemetry";
  joints: number[];
  tof: Record<string, number>;
  ir: number;
  status: RunStatus | null;
}

export interface StatusEvent {
  type: "status";
  line: number;
  kind: string;
  depth: number;
  pass_current: number;
  pass_total: number;
}

export type WsFrame = TelemetryFrame | StatusEvent;

// Canonical Braccio joint tokens (matches braccio_ctrl.constants).
export const JOINT_TOKENS = ["B", "S", "E", "WV", "WR", "G"] as const;
export type JointToken = (typeof JOINT_TOKENS)[number];

export const HOME_JOINTS: JointVector = [90, 90, 90, 90, 90, 73];
