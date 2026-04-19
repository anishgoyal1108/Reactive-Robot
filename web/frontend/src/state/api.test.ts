// api.test.ts — Unit tests for the REST client.
//
// The tests stub fetch so they don't depend on a running backend.
// Each test exercises one method and verifies both the URL/body
// produced and the unwrapping of the response.

import { describe, expect, it, vi } from "vitest";
import { ApiClient } from "./api";

interface Call {
  url: string;
  init?: RequestInit;
}

function makeStub(responses: Record<string, unknown>): {
  calls: Call[];
  fetchImpl: typeof fetch;
} {
  const calls: Call[] = [];
  const fetchImpl = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
    const s = String(url);
    calls.push({ url: s, init });
    if (!(s in responses)) {
      return new Response("not found", { status: 404 });
    }
    const payload = responses[s];
    if (typeof payload === "string") {
      return new Response(payload, {
        status: 200,
        headers: { "content-type": "application/xml" },
      });
    }
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as unknown as typeof fetch;
  return { calls, fetchImpl };
}

describe("ApiClient.health", () => {
  it("GETs /health and returns the body", async () => {
    const { calls, fetchImpl } = makeStub({
      "/health": { ok: true, backend_open: true, running: false },
    });
    const api = new ApiClient({ fetchImpl });
    const body = await api.health();
    expect(body.ok).toBe(true);
    expect(calls[0].url).toBe("/health");
    expect(calls[0].init).toBeUndefined();
  });
});

describe("ApiClient.urdf", () => {
  it("returns raw XML text", async () => {
    const xml = "<robot name='braccio_v2'/>";
    const { fetchImpl } = makeStub({ "/urdf": xml });
    const api = new ApiClient({ fetchImpl });
    expect(await api.urdf()).toBe(xml);
  });

  it("throws when the backend 404s", async () => {
    const { fetchImpl } = makeStub({});
    const api = new ApiClient({ fetchImpl });
    await expect(api.urdf()).rejects.toThrow(/404/);
  });
});

describe("ApiClient.listStates", () => {
  it("unwraps the states array", async () => {
    const { fetchImpl } = makeStub({
      "/states": {
        states: [
          {
            name: "HOME_REST",
            joints: [90, 90, 90, 90, 90, 73],
            theta: 0,
            r: 0,
            z: 0,
            wrist_offset: 0,
            wrist_rot: 0,
            gripper: 73,
          },
        ],
      },
    });
    const api = new ApiClient({ fetchImpl });
    const states = await api.listStates();
    expect(states).toHaveLength(1);
    expect(states[0].name).toBe("HOME_REST");
  });
});

describe("ApiClient.saveState", () => {
  it("POSTs JSON to /states/{name}", async () => {
    const returned = {
      name: "MY",
      joints: [10, 20, 30, 40, 50, 60],
      theta: 0,
      r: 0,
      z: 0,
      wrist_offset: 0,
      wrist_rot: 0,
      gripper: 60,
    };
    const { calls, fetchImpl } = makeStub({ "/states/MY": returned });
    const api = new ApiClient({ fetchImpl });
    const body = await api.saveState("MY", { joints: [10, 20, 30, 40, 50, 60] });
    expect(body).toEqual(returned);
    expect(calls[0].url).toBe("/states/MY");
    expect(calls[0].init?.method).toBe("POST");
    const sent = JSON.parse(String(calls[0].init?.body));
    expect(sent.joints).toEqual([10, 20, 30, 40, 50, 60]);
  });
});

describe("ApiClient sequences", () => {
  it("listSequences unwraps the sequences array", async () => {
    const { fetchImpl } = makeStub({ "/sequences": { sequences: ["a", "b"] } });
    const api = new ApiClient({ fetchImpl });
    expect(await api.listSequences()).toEqual(["a", "b"]);
  });

  it("getSequence unwraps the text field", async () => {
    const { fetchImpl } = makeStub({
      "/sequences/demo": { name: "demo", text: "HOME_REST 50\n" },
    });
    const api = new ApiClient({ fetchImpl });
    expect(await api.getSequence("demo")).toBe("HOME_REST 50\n");
  });

  it("saveSequence POSTs the text and returns ok/errors", async () => {
    const { calls, fetchImpl } = makeStub({
      "/sequences/demo": { ok: true, errors: [] },
    });
    const api = new ApiClient({ fetchImpl });
    const r = await api.saveSequence("demo", "HOME_REST 10\n");
    expect(r.ok).toBe(true);
    const sent = JSON.parse(String(calls[0].init?.body));
    expect(sent.text).toBe("HOME_REST 10\n");
  });

  it("deleteSequence returns the deleted flag", async () => {
    const { calls, fetchImpl } = makeStub({
      "/sequences/demo": { deleted: true },
    });
    const api = new ApiClient({ fetchImpl });
    expect(await api.deleteSequence("demo")).toBe(true);
    expect(calls[0].init?.method).toBe("DELETE");
  });
});

describe("ApiClient DSL run control", () => {
  it("validateDsl returns ok + errors", async () => {
    const { fetchImpl } = makeStub({
      "/dsl/validate": { ok: false, errors: ["boom"] },
    });
    const api = new ApiClient({ fetchImpl });
    const r = await api.validateDsl("bad");
    expect(r.ok).toBe(false);
    expect(r.errors).toEqual(["boom"]);
  });

  it("runDsl returns started flag", async () => {
    const { fetchImpl } = makeStub({
      "/dsl/run": { started: true, errors: [] },
    });
    const api = new ApiClient({ fetchImpl });
    const r = await api.runDsl("MOVE HOME_REST WAIT 10");
    expect(r.started).toBe(true);
  });

  it("stopDsl returns the stopped flag", async () => {
    const { fetchImpl } = makeStub({ "/dsl/stop": { stopped: true } });
    const api = new ApiClient({ fetchImpl });
    expect(await api.stopDsl()).toBe(true);
  });

  it("runStatus returns the full status object", async () => {
    const { fetchImpl } = makeStub({
      "/dsl/status": {
        running: false,
        line: 2,
        kind: "MoveStmt",
        depth: 0,
        pass_current: 1,
        pass_total: 1,
        errors: [],
      },
    });
    const api = new ApiClient({ fetchImpl });
    const status = await api.runStatus();
    expect(status.line).toBe(2);
    expect(status.kind).toBe("MoveStmt");
  });
});

describe("ApiClient telemetry pokes", () => {
  it("pokeTof POSTs channel + distance_mm", async () => {
    const { calls, fetchImpl } = makeStub({ "/telemetry/tof": { ok: true } });
    const api = new ApiClient({ fetchImpl });
    await api.pokeTof(2, 180.5);
    const sent = JSON.parse(String(calls[0].init?.body));
    expect(sent).toEqual({ channel: 2, distance_mm: 180.5 });
  });

  it("pokeIr POSTs severity", async () => {
    const { calls, fetchImpl } = makeStub({ "/telemetry/ir": { ok: true } });
    const api = new ApiClient({ fetchImpl });
    await api.pokeIr(3);
    const sent = JSON.parse(String(calls[0].init?.body));
    expect(sent).toEqual({ severity: 3 });
  });
});

describe("ApiClient error surfacing", () => {
  it("throws with the backend detail on 4xx", async () => {
    const fetchImpl = vi.fn(
      async () =>
        new Response(JSON.stringify({ detail: "unknown state" }), {
          status: 404,
          headers: { "content-type": "application/json" },
        }),
    ) as unknown as typeof fetch;
    const api = new ApiClient({ fetchImpl });
    await expect(api.getState("MISSING")).rejects.toThrow(/unknown state/);
  });
});

// ── Phase 7: deploy mode ──────────────────────────────────────────────

describe("ApiClient.getMode / setMode", () => {
  it("getMode returns the current mode string", async () => {
    const { calls, fetchImpl } = makeStub({ "/mode": { mode: "sim" } });
    const api = new ApiClient({ fetchImpl });
    expect(await api.getMode()).toBe("sim");
    expect(calls[0].url).toBe("/mode");
    expect(calls[0].init).toBeUndefined();
  });

  it("setMode POSTs the target mode and returns the applied value", async () => {
    const { calls, fetchImpl } = makeStub({
      "/mode": { mode: "hardware" },
    });
    const api = new ApiClient({ fetchImpl });
    const applied = await api.setMode("hardware");
    expect(applied).toBe("hardware");
    expect(calls[0].url).toBe("/mode");
    expect(calls[0].init?.method).toBe("POST");
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({
      mode: "hardware",
    });
  });

  it("setMode surfaces backend 400 errors", async () => {
    const fetchImpl = vi.fn(
      async () =>
        new Response(JSON.stringify({ detail: "invalid mode 'moon'" }), {
          status: 400,
          headers: { "content-type": "application/json" },
        }),
    ) as unknown as typeof fetch;
    const api = new ApiClient({ fetchImpl });
    await expect(
      api.setMode("moon" as unknown as "sim"),
    ).rejects.toThrow(/invalid mode/);
  });
});
