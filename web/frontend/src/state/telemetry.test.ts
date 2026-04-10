// telemetry.test.ts — Unit tests for the WS client + zustand store.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  coerceJoints,
  handleFrame,
  resetTelemetryStore,
  startTelemetry,
  useTelemetryStore,
} from "./telemetry";
import type { TelemetryFrame, StatusEvent } from "./types";

// ── Mock WebSocket ───────────────────────────────────────────────────

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  readyState = 0;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  close(): void {
    this.readyState = 3;
    this.onclose?.(new CloseEvent("close"));
  }

  // Test helpers.
  fakeOpen(): void {
    this.readyState = 1;
    this.onopen?.(new Event("open"));
  }

  fakeMessage(data: unknown): void {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  fakeError(): void {
    this.onerror?.(new Event("error"));
  }
}

beforeEach(() => {
  MockWebSocket.instances = [];
  resetTelemetryStore();
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
});

// ── Store + frame routing ────────────────────────────────────────────

describe("handleFrame", () => {
  it("applies a telemetry frame to joints/tof/ir", () => {
    const frame: TelemetryFrame = {
      type: "telemetry",
      joints: [10, 20, 30, 40, 50, 60],
      tof: { "0": 150.0, "1": 9999.0, "2": 9999.0, "3": 9999.0 },
      ir: 2,
      status: null,
    };
    handleFrame(frame);
    const s = useTelemetryStore.getState();
    expect(s.joints).toEqual([10, 20, 30, 40, 50, 60]);
    expect(s.tof[0]).toBe(150.0);
    expect(s.ir).toBe(2);
    expect(s.frameCount).toBe(1);
  });

  it("fills missing joint entries from HOME_JOINTS", () => {
    handleFrame({
      type: "telemetry",
      joints: [10, 20], // truncated
      tof: {},
      ir: 0,
      status: null,
    } as TelemetryFrame);
    const s = useTelemetryStore.getState();
    expect(s.joints[0]).toBe(10);
    expect(s.joints[1]).toBe(20);
    // Remaining joints fall back to home defaults.
    expect(s.joints[2]).toBe(90);
    expect(s.joints[5]).toBe(73);
  });

  it("applies a status frame without touching joints", () => {
    // Seed joints first.
    handleFrame({
      type: "telemetry",
      joints: [1, 2, 3, 4, 5, 6],
      tof: {},
      ir: 0,
      status: null,
    });
    const before = useTelemetryStore.getState().joints;
    const status: StatusEvent = {
      type: "status",
      line: 5,
      kind: "IfBlock",
      depth: 1,
      pass_current: 1,
      pass_total: 3,
    };
    handleFrame(status);
    const s = useTelemetryStore.getState();
    expect(s.joints).toEqual(before);
    expect(s.status?.line).toBe(5);
    expect(s.status?.kind).toBe("IfBlock");
    expect(s.status?.pass_total).toBe(3);
  });
});

describe("coerceJoints", () => {
  it("pads short arrays with HOME defaults", () => {
    const out = coerceJoints([10]);
    expect(out[0]).toBe(10);
    expect(out[1]).toBe(90); // HOME[1]
    expect(out.length).toBe(6);
  });

  it("truncates long arrays to 6 elements", () => {
    const out = coerceJoints([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(out.length).toBe(6);
    expect(out[5]).toBe(6);
  });

  it("replaces NaN with the home value", () => {
    const out = coerceJoints([Number.NaN, 20, 30, 40, 50, 60]);
    expect(out[0]).toBe(90); // HOME[0]
    expect(out[1]).toBe(20);
  });
});

// ── WS client lifecycle ──────────────────────────────────────────────

describe("startTelemetry", () => {
  it("sets connected when the socket opens", () => {
    const dispose = startTelemetry({
      url: "ws://mock/ws",
      wsImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    expect(MockWebSocket.instances).toHaveLength(1);
    MockWebSocket.instances[0].fakeOpen();
    expect(useTelemetryStore.getState().connected).toBe(true);
    dispose();
  });

  it("routes incoming frames through the store", () => {
    const dispose = startTelemetry({
      url: "ws://mock/ws",
      wsImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    const sock = MockWebSocket.instances[0];
    sock.fakeOpen();
    sock.fakeMessage({
      type: "telemetry",
      joints: [45, 45, 45, 45, 45, 45],
      tof: { "0": 200.0 },
      ir: 1,
      status: null,
    });
    const s = useTelemetryStore.getState();
    expect(s.joints[0]).toBe(45);
    expect(s.ir).toBe(1);
    expect(s.frameCount).toBe(1);
    dispose();
  });

  it("schedules a reconnect when the socket closes", () => {
    vi.useFakeTimers();
    const dispose = startTelemetry({
      url: "ws://mock/ws",
      wsImpl: MockWebSocket as unknown as typeof WebSocket,
      reconnectMs: 50,
    });
    expect(MockWebSocket.instances).toHaveLength(1);
    MockWebSocket.instances[0].fakeOpen();
    MockWebSocket.instances[0].close();
    expect(useTelemetryStore.getState().connected).toBe(false);
    vi.advanceTimersByTime(60);
    expect(MockWebSocket.instances.length).toBeGreaterThanOrEqual(2);
    dispose();
  });

  it("dispose() stops further reconnects", () => {
    vi.useFakeTimers();
    const dispose = startTelemetry({
      url: "ws://mock/ws",
      wsImpl: MockWebSocket as unknown as typeof WebSocket,
      reconnectMs: 20,
    });
    MockWebSocket.instances[0].fakeOpen();
    dispose();
    MockWebSocket.instances[0].close();
    vi.advanceTimersByTime(200);
    // Dispose happened before the reconnect timer fired, so no new
    // socket should be created.
    expect(MockWebSocket.instances).toHaveLength(1);
  });

  it("captures an error on bad JSON and keeps the connection", () => {
    const dispose = startTelemetry({
      url: "ws://mock/ws",
      wsImpl: MockWebSocket as unknown as typeof WebSocket,
    });
    const sock = MockWebSocket.instances[0];
    sock.fakeOpen();
    sock.onmessage?.({ data: "{not: json" } as MessageEvent);
    const s = useTelemetryStore.getState();
    expect(s.lastError).toContain("bad frame");
    dispose();
  });
});
