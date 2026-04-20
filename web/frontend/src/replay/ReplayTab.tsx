// ReplayTab.tsx — Left-panel UI for the Session Replay tab.
//
// Wraps file-upload + sample-load + playback controls + a tick table
// so the user can scrub through a recorded run and see the arm, the
// detour geometry, and the detected obstacle points in the 3D scene.

import { useCallback, useEffect, useMemo, useRef, type DragEvent } from "react";
import {
  useReplayStore,
  startReplayLoop,
} from "./ReplayStore";

const SAMPLE_URL = "samples/sweep_with_detour.jsonl";

export function ReplayTab() {
  const ticks = useReplayStore((s) => s.ticks);
  const cursor = useReplayStore((s) => s.cursor);
  const playing = useReplayStore((s) => s.playing);
  const speed = useReplayStore((s) => s.speed);
  const loadedName = useReplayStore((s) => s.loadedName);
  const error = useReplayStore((s) => s.error);
  const projectTofFallback = useReplayStore((s) => s.projectTofFallback);

  const loadJsonl = useReplayStore((s) => s.loadJsonl);
  const loadFromUrl = useReplayStore((s) => s.loadFromUrl);
  const togglePlay = useReplayStore((s) => s.togglePlay);
  const seek = useReplayStore((s) => s.seek);
  const step = useReplayStore((s) => s.step);
  const setSpeed = useReplayStore((s) => s.setSpeed);
  const setProjectTofFallback = useReplayStore(
    (s) => s.setProjectTofFallback,
  );

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Spin up the playback loop once while this tab is mounted.
  useEffect(() => {
    const stop = startReplayLoop();
    return stop;
  }, []);

  const currentTick = ticks[cursor];
  const totalTicks = ticks.length;
  const hasGrids = (currentTick?.tof.grids.length ?? 0) > 0;
  const hasLoggedPoints = (currentTick?.world_points.length ?? 0) > 0;

  const onFilePicked = useCallback(
    async (file: File | null) => {
      if (!file) return;
      const text = await file.text();
      loadJsonl(text, file.name);
    },
    [loadJsonl],
  );

  const onDrop = useCallback(
    async (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      if (file) await onFilePicked(file);
    },
    [onFilePicked],
  );

  const scrubberMax = Math.max(0, totalTicks - 1);
  const tRel = currentTick?.t_rel.toFixed(2) ?? "—";
  const strategy = currentTick?.bt.last_strategy || "—";
  const theta = currentTick ? currentTick.theta.toFixed(1) : "—";
  const numPoints = currentTick?.world_points.length ?? 0;
  const inDetour = (currentTick?.detour_path.length ?? 0) > 0;

  const stats = useMemo(() => {
    if (totalTicks === 0) return null;
    const first = ticks[0].t_rel;
    const last = ticks[totalTicks - 1].t_rel;
    return {
      ticks: totalTicks,
      duration: (last - first).toFixed(1),
    };
  }, [ticks, totalTicks]);

  return (
    <div className="replay-tab">
      <div className="panel-heading">Session Replay</div>

      <div
        className="replay-tab__dropzone"
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        role="button"
        tabIndex={0}
        onClick={() => fileInputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
      >
        Drop a session_logs/*.jsonl here
        <div className="replay-tab__dropzone-hint">or click to browse</div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".jsonl,.json,.txt"
          style={{ display: "none" }}
          onChange={(e) => void onFilePicked(e.target.files?.[0] ?? null)}
        />
      </div>

      <div className="replay-tab__row">
        <button
          type="button"
          onClick={() => void loadFromUrl(SAMPLE_URL, "sample: sweep_with_detour")}
        >
          Load sample run
        </button>
        {loadedName && (
          <span className="replay-tab__name" title={loadedName}>
            {loadedName}
          </span>
        )}
      </div>

      {error && <div className="replay-tab__error">⚠ {error}</div>}

      {totalTicks > 0 && (
        <>
          <div className="replay-tab__row">
            <button type="button" onClick={() => step(-1)}>◀ step</button>
            <button type="button" onClick={togglePlay}>
              {playing ? "⏸ pause" : "▶ play"}
            </button>
            <button type="button" onClick={() => step(+1)}>step ▶</button>
          </div>

          <div className="replay-tab__scrubber">
            <input
              type="range"
              min={0}
              max={scrubberMax}
              value={cursor}
              onChange={(e) => seek(Number(e.target.value))}
              aria-label="Replay scrubber"
            />
            <div className="replay-tab__scrubber-meta">
              tick {cursor + 1}/{totalTicks} · t={tRel}s
            </div>
          </div>

          <div className="replay-tab__row">
            <label style={{ fontSize: 12, opacity: 0.8 }}>
              speed
              <select
                value={speed}
                onChange={(e) => setSpeed(Number(e.target.value))}
                style={{ marginLeft: 6 }}
              >
                <option value={0.25}>0.25×</option>
                <option value={0.5}>0.5×</option>
                <option value={1}>1×</option>
                <option value={2}>2×</option>
                <option value={4}>4×</option>
              </select>
            </label>
          </div>

          {hasGrids && !hasLoggedPoints && (
            <label
              className="replay-tab__row"
              style={{ fontSize: 12, opacity: 0.85 }}
              title={
                "Old log has no world_points — project the ToF grids " +
                "client-side through the URDF chain instead."
              }
            >
              <input
                type="checkbox"
                checked={projectTofFallback}
                onChange={(e) =>
                  setProjectTofFallback(e.target.checked)
                }
              />
              Project ToF grids (legacy log)
            </label>
          )}

          <div className="panel-heading">Tick detail</div>
          <div className="joint-row">
            <span>strategy</span>
            <span
              style={{
                color: inDetour
                  ? "#f59e0b"
                  : strategy.startsWith("simple_reverse")
                    ? "#ef4444"
                    : undefined,
              }}
            >
              {strategy}
            </span>
          </div>
          <div className="joint-row">
            <span>θ (base)</span>
            <span>{theta}°</span>
          </div>
          <div className="joint-row">
            <span>obstacle points</span>
            <span>{numPoints}</span>
          </div>
          <div className="joint-row">
            <span>detour active</span>
            <span>{inDetour ? "yes" : "no"}</span>
          </div>
          {stats && (
            <>
              <div className="joint-row">
                <span>run length</span>
                <span>{stats.duration}s ({stats.ticks} ticks)</span>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
