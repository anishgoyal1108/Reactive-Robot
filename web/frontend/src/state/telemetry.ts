// telemetry.ts — WebSocket client + zustand store for live joint/sensor data.
//
// The Phase 3 backend broadcasts two frame types:
//   - type: "telemetry"  — full snapshot every ~33 ms
//   - type: "status"     — fires once per DSL statement
//
// This module owns the WebSocket connection, parses incoming frames,
// and exposes a zustand store that React components subscribe to via
// selectors. Keeping the connection outside React avoids the common
// StrictMode double-mount pitfall.

import { create } from "zustand";
import {
  HOME_JOINTS,
  type JointVector,
  type RunStatus,
  type TelemetryFrame,
  type WsFrame,
} from "./types";

// ── Store shape ──────────────────────────────────────────────────────

export interface TelemetryState {
  connected: boolean;
  /** Latest 6-DOF joint vector, in servo degrees. */
  joints: JointVector;
  /** Per-channel ToF distances in mm; missing channels report 9999. */
  tof: Record<number, number>;
  /** IR severity 0..3. */
  ir: number;
  /** Latest run status broadcast by the interpreter, if any. */
  status: RunStatus | null;
  /** Most recent WS error message, or null if the stream is healthy. */
  lastError: string | null;
  /** Count of frames received since the connection opened. */
  frameCount: number;

  // setters — invoked by the WS client below.
  _setConnected: (v: boolean) => void;
  _applyTelemetry: (frame: TelemetryFrame) => void;
  _setStatus: (status: RunStatus) => void;
  _setError: (msg: string | null) => void;
}

const DEFAULT_TOF: Record<number, number> = { 0: 9999, 1: 9999, 2: 9999, 3: 9999 };

export const useTelemetryStore = create<TelemetryState>((set) => ({
  connected: false,
  joints: HOME_JOINTS,
  tof: { ...DEFAULT_TOF },
  ir: 0,
  status: null,
  lastError: null,
  frameCount: 0,

  _setConnected: (v) => set({ connected: v }),

  _applyTelemetry: (frame) => {
    const joints = coerceJoints(frame.joints);
    const tof = { ...DEFAULT_TOF };
    for (const [k, v] of Object.entries(frame.tof)) {
      const idx = Number(k);
      if (Number.isFinite(idx)) {
        tof[idx] = Number(v);
      }
    }
    set((prev) => ({
      joints,
      tof,
      ir: Number(frame.ir),
      status: frame.status ?? prev.status,
      frameCount: prev.frameCount + 1,
    }));
  },

  _setStatus: (status) => set({ status }),
  _setError: (msg) => set({ lastError: msg }),
}));

// ── WS client ────────────────────────────────────────────────────────

export interface TelemetryClientOptions {
  url?: string;
  /** WebSocket constructor — overridable for tests. */
  wsImpl?: typeof WebSocket;
  /** Milliseconds to wait before reconnecting after a drop. */
  reconnectMs?: number;
}

/**
 * Start a live telemetry subscription. Returns a disposer that closes
 * the socket and detaches reconnect timers. Safe to call multiple
 * times (each returns an independent subscription); ``App.tsx`` only
 * starts one.
 */
export function startTelemetry(options: TelemetryClientOptions = {}): () => void {
  const url = options.url ?? defaultWsUrl();
  const WsCtor = options.wsImpl ?? (globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket;
  const reconnectMs = options.reconnectMs ?? 1000;

  let closed = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const store = useTelemetryStore.getState();

  function connect(): void {
    if (closed) return;
    try {
      socket = new WsCtor(url);
    } catch (err) {
      store._setError(String(err));
      scheduleReconnect();
      return;
    }

    socket.onopen = () => {
      store._setConnected(true);
      store._setError(null);
    };
    socket.onmessage = (ev: MessageEvent) => {
      try {
        const frame = JSON.parse(String(ev.data)) as WsFrame;
        handleFrame(frame);
      } catch (err) {
        store._setError(`bad frame: ${String(err)}`);
      }
    };
    socket.onerror = () => {
      store._setError("websocket error");
    };
    socket.onclose = () => {
      store._setConnected(false);
      scheduleReconnect();
    };
  }

  function scheduleReconnect(): void {
    if (closed) return;
    if (reconnectTimer !== null) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, reconnectMs);
  }

  connect();

  return () => {
    closed = true;
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (socket !== null) {
      try {
        socket.close();
      } catch {
        /* ignore */
      }
      socket = null;
    }
  };
}

// ── Frame handling ───────────────────────────────────────────────────

/**
 * Exported for tests: route one parsed WS payload into the store.
 * The production code path never calls this directly; it uses the
 * WebSocket ``onmessage`` handler above.
 */
export function handleFrame(frame: WsFrame): void {
  const store = useTelemetryStore.getState();
  if (frame.type === "telemetry") {
    store._applyTelemetry(frame);
  } else if (frame.type === "status") {
    store._setStatus({
      running: true,
      line: frame.line,
      kind: frame.kind,
      depth: frame.depth,
      pass_current: frame.pass_current,
      pass_total: frame.pass_total,
      errors: [],
    });
  }
}

/** Normalise a joints array into a fixed-length 6-tuple. */
export function coerceJoints(input: readonly number[]): JointVector {
  const out: number[] = [];
  for (let i = 0; i < 6; i++) {
    const v = input[i];
    out.push(Number.isFinite(v) ? Number(v) : HOME_JOINTS[i]);
  }
  return out as unknown as JointVector;
}

// ── URL helpers ──────────────────────────────────────────────────────

/** Derive the WS URL from the current page origin. */
export function defaultWsUrl(): string {
  if (typeof window === "undefined" || !window.location) {
    return "ws://localhost:8000/ws/telemetry";
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/telemetry`;
}

/** Reset the store to its initial values. Useful for tests. */
export function resetTelemetryStore(): void {
  useTelemetryStore.setState({
    connected: false,
    joints: HOME_JOINTS,
    tof: { ...DEFAULT_TOF },
    ir: 0,
    status: null,
    lastError: null,
    frameCount: 0,
  });
}
