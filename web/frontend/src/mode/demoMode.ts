// demoMode.ts — Static-demo shim for GitHub Pages.
//
// When the bundle is built with VITE_DEMO_MODE=true there is no live
// FastAPI backend, so a mock fetch interceptor stands in for every
// REST call the ApiClient makes. State library and sequence entries
// round-trip through localStorage so a visitor can experiment with
// named poses and DSL drafts across page reloads.
//
// Routes that would move a real arm (``/dsl/run``, ``/dsl/stop``,
// ``/telemetry/tof``, ``/telemetry/ir``, ``/mode`` switches to
// hardware) return 503 with an ``X-Demo-Reason`` body so the UI can
// surface a clear "demo mode — no controller attached" message
// instead of silently appearing to run.
//
// The URDF endpoint is rewritten to ``/braccio.urdf``, a static file
// the CI build copies from braccio_twin/urdf/ into the SPA's public
// directory.

import type { StateEntry } from "../state/types";

const STATES_KEY = "reactive-robot:demo:states";
const SEQUENCES_KEY = "reactive-robot:demo:sequences";

type StateRecord = Record<string, StateEntry>;
type SequenceRecord = Record<string, string>;

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = typeof localStorage !== "undefined" ? localStorage.getItem(key) : null;
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown): void {
  try {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* quota exceeded or blocked — the UI keeps working from memory */
  }
}

function seedStates(): StateRecord {
  const existing = readJson<StateRecord>(STATES_KEY, {});
  if (Object.keys(existing).length > 0) return existing;
  const home: StateEntry = {
    name: "HOME",
    joints: [90, 90, 90, 90, 90, 73],
    theta: 90,
    r: 250,
    z: 60,
    wrist_offset: 0,
    wrist_rot: 90,
    gripper: 73,
  };
  const rest: StateEntry = {
    name: "HOME_REST",
    joints: [90, 150, 90, 90, 90, 73],
    theta: 90,
    r: 125,
    z: 0,
    wrist_offset: 0,
    wrist_rot: 90,
    gripper: 73,
  };
  const seeded: StateRecord = { HOME: home, HOME_REST: rest };
  writeJson(STATES_KEY, seeded);
  return seeded;
}

function loadStates(): StateRecord {
  return seedStates();
}

function loadSequences(): SequenceRecord {
  return readJson<SequenceRecord>(SEQUENCES_KEY, {});
}

/** True when the bundle was built with VITE_DEMO_MODE=true. */
export function isDemoMode(): boolean {
  const env = (import.meta as { env?: Record<string, string | undefined> }).env;
  return env?.VITE_DEMO_MODE === "true";
}

function json(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init.headers ?? {}),
    },
  });
}

function disabled(reason: string): Response {
  return new Response(JSON.stringify({ detail: reason }), {
    status: 503,
    headers: {
      "content-type": "application/json",
      "x-demo-reason": reason,
    },
  });
}

function resolveUrl(input: RequestInfo | URL): { path: string; search: string } {
  const raw =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
  const parsed = new URL(raw, "http://demo.local");
  return { path: parsed.pathname, search: parsed.search };
}

async function readBody(init?: RequestInit): Promise<unknown> {
  if (!init || init.body == null) return null;
  if (typeof init.body === "string") {
    try {
      return JSON.parse(init.body);
    } catch {
      return init.body;
    }
  }
  return init.body;
}

async function handle(
  path: string,
  method: string,
  body: unknown,
): Promise<Response | null> {
  // Health — always ok, always sim, never running.
  if (path === "/health" && method === "GET") {
    return json({ ok: true, backend_open: false, running: false, mode: "sim" });
  }

  // Deploy mode — demo mode is effectively sim and cannot be changed.
  if (path === "/mode" && method === "GET") {
    return json({ mode: "sim" });
  }
  if (path === "/mode" && method === "POST") {
    return disabled("Mode switching is disabled in demo mode");
  }

  // URDF — redirect to the static copy in the SPA public dir.
  if (path === "/urdf" && method === "GET") {
    const base =
      (import.meta as { env?: Record<string, string | undefined> }).env?.BASE_URL ??
      "/";
    const resp = await fetch(`${base}braccio.urdf`);
    if (!resp.ok) {
      return disabled("braccio.urdf is not bundled in this build");
    }
    return new Response(await resp.text(), {
      status: 200,
      headers: { "content-type": "application/xml" },
    });
  }

  // State library.
  if (path === "/states" && method === "GET") {
    const states = loadStates();
    return json({ states: Object.values(states) });
  }
  const stateMatch = path.match(/^\/states\/(.+)$/);
  if (stateMatch) {
    const name = decodeURIComponent(stateMatch[1]);
    const states = loadStates();
    if (method === "GET") {
      if (!(name in states)) {
        return new Response(JSON.stringify({ detail: "not found" }), {
          status: 404,
          headers: { "content-type": "application/json" },
        });
      }
      return json(states[name]);
    }
    if (method === "POST") {
      const payload = (body ?? {}) as Partial<StateEntry>;
      const merged: StateEntry = {
        name,
        joints: payload.joints ?? [90, 90, 90, 90, 90, 73],
        theta: payload.theta ?? 0,
        r: payload.r ?? 0,
        z: payload.z ?? 0,
        wrist_offset: payload.wrist_offset ?? 0,
        wrist_rot: payload.wrist_rot ?? 90,
        gripper: payload.gripper ?? 73,
      };
      states[name] = merged;
      writeJson(STATES_KEY, states);
      return json(merged);
    }
    if (method === "DELETE") {
      const existed = name in states;
      delete states[name];
      writeJson(STATES_KEY, states);
      return json({ deleted: existed });
    }
  }

  // Sequences.
  if (path === "/sequences" && method === "GET") {
    const sequences = loadSequences();
    return json({ sequences: Object.keys(sequences) });
  }
  const seqMatch = path.match(/^\/sequences\/(.+)$/);
  if (seqMatch) {
    const name = decodeURIComponent(seqMatch[1]);
    const sequences = loadSequences();
    if (method === "GET") {
      if (!(name in sequences)) {
        return new Response(JSON.stringify({ detail: "not found" }), {
          status: 404,
          headers: { "content-type": "application/json" },
        });
      }
      return json({ name, text: sequences[name] });
    }
    if (method === "POST") {
      const text = ((body ?? {}) as { text?: string }).text ?? "";
      sequences[name] = text;
      writeJson(SEQUENCES_KEY, sequences);
      return json({ ok: true, errors: [] });
    }
    if (method === "DELETE") {
      const existed = name in sequences;
      delete sequences[name];
      writeJson(SEQUENCES_KEY, sequences);
      return json({ deleted: existed });
    }
  }

  // DSL validate — Blockly emits grammar-correct text by construction,
  // so a blanket OK is good enough for the demo. Hardware-side validate
  // still runs on a real backend before any sequence reaches the arm.
  if (path === "/dsl/validate" && method === "POST") {
    return json({ ok: true, errors: [] });
  }

  // DSL run / stop — blocked with a clear reason.
  if (path === "/dsl/run" && method === "POST") {
    return disabled("Demo mode cannot run sequences; copy the DSL to a local controller");
  }
  if (path === "/dsl/stop" && method === "POST") {
    return disabled("Demo mode has nothing to stop");
  }
  if (path === "/dsl/status" && method === "GET") {
    return json({
      running: false,
      line: 0,
      kind: "idle",
      depth: 0,
      pass_current: 0,
      pass_total: 0,
      errors: [],
    });
  }

  // Telemetry — static snapshot plus disabled pokes.
  if (path === "/telemetry" && method === "GET") {
    return json({
      type: "telemetry",
      joints: [90, 90, 90, 90, 90, 73],
      tof: { "0": 9999, "1": 9999, "2": 9999, "3": 9999 },
      ir: 0,
      status: { running: false },
    });
  }
  if (path === "/telemetry/tof" && method === "POST") {
    return disabled("Telemetry injection is disabled in demo mode");
  }
  if (path === "/telemetry/ir" && method === "POST") {
    return disabled("Telemetry injection is disabled in demo mode");
  }

  return null;
}

/**
 * Drop-in replacement for ``fetch`` that intercepts every backend
 * route. Non-backend URLs (e.g. third-party CDN fetches) fall through
 * to the real ``fetch`` so the viewer still works.
 */
export async function demoFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const { path } = resolveUrl(input);
  const method = (init?.method ?? "GET").toUpperCase();
  const body = await readBody(init);

  const match = await handle(path, method, body);
  if (match !== null) return match;

  // Anything the mock doesn't recognise — pass through. The only
  // expected pass-through is the static ``braccio.urdf`` asset, which
  // ``handle`` already reroutes.
  return fetch(input, init);
}

/** Thrown when a caller explicitly guards against demo-mode actions. */
export class DemoModeDisabledError extends Error {
  constructor(action: string) {
    super(`${action} is disabled in demo mode`);
    this.name = "DemoModeDisabledError";
  }
}
