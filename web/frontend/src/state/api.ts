// api.ts — Thin REST wrapper around the Phase 3 FastAPI backend.
//
// Every endpoint has a matching helper so components can call
// ``api.runDsl(text)`` without reaching for fetch() directly. Errors
// bubble up as plain Error objects — the caller decides whether to
// show a toast or crash the workspace.

import type { RunStatus, StateEntry, TelemetryFrame } from "./types";

const DEFAULT_BASE = "";

export interface ValidateResponse {
  ok: boolean;
  errors: string[];
}

export interface RunResponse {
  started: boolean;
  errors: string[];
}

export interface SaveSequenceResponse {
  ok: boolean;
  errors: string[];
}

/** Phase 7 — backend deploy mode. */
export type DeployMode = "sim" | "hardware";

export interface ModeResponse {
  mode: DeployMode;
}

export interface ApiClientOptions {
  /** Base URL prefix. Defaults to same-origin ("") for the Vite proxy. */
  baseUrl?: string;
  /** Override fetch for tests. */
  fetchImpl?: typeof fetch;
}

/**
 * Typed, dependency-free REST client for the FastAPI backend.
 *
 * The client is intentionally boring — no retries, no caching, no
 * global state. Components that want caching can wrap it in
 * react-query; the Phase 4 viewer doesn't, it just re-runs queries
 * on demand.
 */
export class ApiClient {
  readonly baseUrl: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE;
    this.fetchImpl = options.fetchImpl ?? fetch.bind(globalThis);
  }

  // ── Health + URDF ──────────────────────────────────────────────

  async health(): Promise<{
    ok: boolean;
    backend_open: boolean;
    running: boolean;
    mode: DeployMode;
  }> {
    return this.getJson("/health");
  }

  async urdf(): Promise<string> {
    const resp = await this.fetchImpl(`${this.baseUrl}/urdf`);
    if (!resp.ok) {
      throw new Error(`GET /urdf failed: ${resp.status}`);
    }
    return resp.text();
  }

  // ── Deploy mode (Phase 7) ──────────────────────────────────────

  async getMode(): Promise<DeployMode> {
    const body = await this.getJson<ModeResponse>("/mode");
    return body.mode;
  }

  async setMode(mode: DeployMode): Promise<DeployMode> {
    const body = await this.postJson<ModeResponse>("/mode", { mode });
    return body.mode;
  }

  // ── State library ──────────────────────────────────────────────

  async listStates(): Promise<StateEntry[]> {
    const body = await this.getJson<{ states: StateEntry[] }>("/states");
    return body.states;
  }

  async getState(name: string): Promise<StateEntry> {
    return this.getJson<StateEntry>(`/states/${encodeURIComponent(name)}`);
  }

  async saveState(name: string, entry: Partial<StateEntry>): Promise<StateEntry> {
    return this.postJson<StateEntry>(
      `/states/${encodeURIComponent(name)}`,
      entry,
    );
  }

  async deleteState(name: string): Promise<boolean> {
    const body = await this.deleteJson<{ deleted: boolean }>(
      `/states/${encodeURIComponent(name)}`,
    );
    return body.deleted;
  }

  // ── Sequences ──────────────────────────────────────────────────

  async listSequences(): Promise<string[]> {
    const body = await this.getJson<{ sequences: string[] }>("/sequences");
    return body.sequences;
  }

  async getSequence(name: string): Promise<string> {
    const body = await this.getJson<{ name: string; text: string }>(
      `/sequences/${encodeURIComponent(name)}`,
    );
    return body.text;
  }

  async saveSequence(name: string, text: string): Promise<SaveSequenceResponse> {
    return this.postJson<SaveSequenceResponse>(
      `/sequences/${encodeURIComponent(name)}`,
      { text },
    );
  }

  async deleteSequence(name: string): Promise<boolean> {
    const body = await this.deleteJson<{ deleted: boolean }>(
      `/sequences/${encodeURIComponent(name)}`,
    );
    return body.deleted;
  }

  // ── DSL run control ────────────────────────────────────────────

  async validateDsl(text: string): Promise<ValidateResponse> {
    return this.postJson<ValidateResponse>("/dsl/validate", { text });
  }

  async runDsl(text: string): Promise<RunResponse> {
    return this.postJson<RunResponse>("/dsl/run", { text });
  }

  async stopDsl(): Promise<boolean> {
    const body = await this.postJson<{ stopped: boolean }>("/dsl/stop", {});
    return body.stopped;
  }

  async runStatus(): Promise<RunStatus> {
    return this.getJson<RunStatus>("/dsl/status");
  }

  // ── Telemetry ──────────────────────────────────────────────────

  async telemetry(): Promise<TelemetryFrame> {
    return this.getJson<TelemetryFrame>("/telemetry");
  }

  async pokeTof(channel: number, distanceMm: number): Promise<void> {
    await this.postJson<{ ok: boolean }>("/telemetry/tof", {
      channel,
      distance_mm: distanceMm,
    });
  }

  async pokeIr(severity: number): Promise<void> {
    await this.postJson<{ ok: boolean }>("/telemetry/ir", { severity });
  }

  // ── Internals ──────────────────────────────────────────────────

  private async getJson<T>(path: string): Promise<T> {
    const resp = await this.fetchImpl(`${this.baseUrl}${path}`);
    return this.unwrap<T>(resp, "GET", path);
  }

  private async postJson<T>(path: string, body: unknown): Promise<T> {
    const resp = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    return this.unwrap<T>(resp, "POST", path);
  }

  private async deleteJson<T>(path: string): Promise<T> {
    const resp = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method: "DELETE",
    });
    return this.unwrap<T>(resp, "DELETE", path);
  }

  private async unwrap<T>(resp: Response, method: string, path: string): Promise<T> {
    if (!resp.ok) {
      let detail = "";
      try {
        const j = (await resp.json()) as { detail?: string };
        detail = j.detail ?? "";
      } catch {
        /* ignore */
      }
      throw new Error(
        `${method} ${path} failed: ${resp.status}${detail ? " — " + detail : ""}`,
      );
    }
    return (await resp.json()) as T;
  }
}

/** Default client that talks to the same origin (via Vite proxy in dev). */
export const api = new ApiClient();
