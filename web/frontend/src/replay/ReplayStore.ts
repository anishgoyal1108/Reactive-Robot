// ReplayStore.ts — Zustand slice that drives the Replay tab.
//
// Holds the parsed session ticks, a cursor, and a play/pause/speed
// toggle. When `active === true`, Arm.tsx reads joints from this store
// instead of the live telemetry store (see Arm.tsx::useFrame source
// multiplex). Time advancement runs through a requestAnimationFrame
// loop that respects the playback speed and uses each tick's t_rel
// field for wall-clock-accurate pacing.

import { create } from "zustand";
import type { JointVector } from "../state/types";
import { parseSessionJsonl, type SessionTick } from "./parseJsonl";

export interface ReplayState {
  /** Whether the Replay tab is currently the active source. Arm.tsx
   *  reads this to decide between live telemetry and replay joints. */
  active: boolean;
  ticks: SessionTick[];
  cursor: number;
  playing: boolean;
  /** Playback speed multiplier (1× = wall-clock, 0.25× = quarter
   *  speed, etc.). */
  speed: number;
  /** Name of the file or sample that populated `ticks`; displayed in
   *  the scrubber UI. Empty string when nothing's loaded. */
  loadedName: string;
  /** Parse or fetch error surfaced to the UI, cleared on next load. */
  error: string | null;
  /** When true, old logs without `world_points` fall back to client-side
   *  projection of `tof.grids` through the URDF chain. Auto-enabled if
   *  the first tick of the loaded log has no world_points but does have
   *  grids. The user can toggle it off in the UI. */
  projectTofFallback: boolean;

  setActive: (v: boolean) => void;
  setProjectTofFallback: (v: boolean) => void;
  loadJsonl: (text: string, name?: string) => void;
  loadFromUrl: (url: string, name?: string) => Promise<void>;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  seek: (idx: number) => void;
  step: (delta: number) => void;
  setSpeed: (x: number) => void;
  reset: () => void;
}

function _clampCursor(idx: number, total: number): number {
  if (total === 0) return 0;
  if (idx < 0) return 0;
  if (idx >= total) return total - 1;
  return idx;
}

export const useReplayStore = create<ReplayState>((set, get) => ({
  active: false,
  ticks: [],
  cursor: 0,
  playing: false,
  speed: 1.0,
  loadedName: "",
  error: null,
  projectTofFallback: false,

  setActive: (v) => set({ active: v, playing: v ? get().playing : false }),
  setProjectTofFallback: (v) => set({ projectTofFallback: v }),

  loadJsonl: (text, name) => {
    try {
      const ticks = parseSessionJsonl(text);
      if (ticks.length === 0) {
        set({ error: "no tick records found in file", ticks: [],
              cursor: 0, playing: false });
        return;
      }
      // Auto-enable ToF fallback when the log is old-style: no
      // world_points anywhere in the first 60 ticks but grids are
      // present. That's the fingerprint of a session captured before
      // SESSION_LOG_INCLUDE_GEOMETRY shipped.
      const sniff = ticks.slice(0, 60);
      const anyWorldPoints = sniff.some((t) => t.world_points.length > 0);
      const anyGrids = sniff.some((t) => t.tof.grids.length > 0);
      const needsFallback = !anyWorldPoints && anyGrids;
      set({
        ticks,
        cursor: 0,
        playing: false,
        loadedName: name ?? "uploaded log",
        error: null,
        projectTofFallback: needsFallback,
      });
    } catch (err) {
      set({ error: String(err), ticks: [], cursor: 0, playing: false });
    }
  },

  loadFromUrl: async (url, name) => {
    try {
      const resp = await fetch(url);
      if (!resp.ok) {
        set({ error: `fetch failed: ${resp.status}`, ticks: [] });
        return;
      }
      const text = await resp.text();
      get().loadJsonl(text, name ?? url);
    } catch (err) {
      set({ error: String(err), ticks: [], cursor: 0, playing: false });
    }
  },

  play: () => {
    const { ticks, cursor } = get();
    if (ticks.length === 0) return;
    if (cursor >= ticks.length - 1) {
      set({ cursor: 0, playing: true });
      return;
    }
    set({ playing: true });
  },
  pause: () => set({ playing: false }),
  togglePlay: () => {
    if (get().playing) get().pause();
    else get().play();
  },

  seek: (idx) =>
    set(({ ticks }) => ({ cursor: _clampCursor(idx, ticks.length) })),
  step: (delta) =>
    set(({ ticks, cursor }) => ({
      cursor: _clampCursor(cursor + delta, ticks.length),
    })),
  setSpeed: (x) => set({ speed: Math.max(0.1, Math.min(8, x)) }),

  reset: () =>
    set({
      ticks: [],
      cursor: 0,
      playing: false,
      loadedName: "",
      error: null,
    }),
}));

/**
 * Start a requestAnimationFrame playback loop that advances the cursor
 * according to each tick's t_rel timestamp and the current speed.
 * Returns a disposer that cancels the loop. Safe to start/stop any
 * number of times — used by ReplayTab on mount/unmount.
 */
export function startReplayLoop(): () => void {
  let cancelled = false;
  let rafId: number | null = null;
  let anchorWallMs = 0;
  let anchorTRel = 0;
  let prevSpeed = 0;
  let prevCursor = -1;

  const tick = (nowMs: number) => {
    if (cancelled) return;
    const s = useReplayStore.getState();
    if (!s.playing || s.ticks.length === 0) {
      rafId = requestAnimationFrame(tick);
      return;
    }

    // Re-anchor whenever the user scrubs or speed changes so playback
    // stays wall-clock-correct from the new anchor forward.
    if (
      prevCursor !== s.cursor ||
      prevSpeed !== s.speed ||
      anchorWallMs === 0
    ) {
      anchorWallMs = nowMs;
      anchorTRel = s.ticks[s.cursor].t_rel;
      prevSpeed = s.speed;
    }

    const elapsedMs = (nowMs - anchorWallMs) * s.speed;
    const targetTRel = anchorTRel + elapsedMs / 1000;

    // Advance cursor forward through any ticks whose t_rel <= target.
    let idx = s.cursor;
    while (idx + 1 < s.ticks.length && s.ticks[idx + 1].t_rel <= targetTRel) {
      idx += 1;
    }
    if (idx !== s.cursor) {
      useReplayStore.setState({ cursor: idx });
    }
    prevCursor = idx;

    // End-of-log: pause and hold on the final tick.
    if (idx >= s.ticks.length - 1) {
      useReplayStore.setState({ playing: false });
    }

    rafId = requestAnimationFrame(tick);
  };

  rafId = requestAnimationFrame(tick);
  return () => {
    cancelled = true;
    if (rafId !== null) cancelAnimationFrame(rafId);
  };
}

/** Resolve the current joint vector from the replay cursor. Returns
 *  null when no tick is loaded (Arm.tsx falls back to live telemetry). */
export function getReplayJoints(): JointVector | null {
  const { ticks, cursor } = useReplayStore.getState();
  if (ticks.length === 0) return null;
  return ticks[cursor].joints;
}
