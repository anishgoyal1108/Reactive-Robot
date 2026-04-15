// App.tsx — Phase 5+7 top-level layout.
//
// Three columns:
//   left    — Run controls + Blockly workspace (Phase 5)
//   center  — Three.js viewer (Phase 4)
//   right   — live telemetry readouts + Manual Joint Panel (Phase 5)
//             + Recorder panel (Phase 6)
//
// Phase 7 adds a deploy-mode banner in the topbar. Clicking it pops a
// confirmation modal — flipping to hardware is a blast-radius action
// (moving a real arm), so the frontend deliberately makes it two
// clicks and shows a safety warning about MotionGuard not being in
// the web backend path.
//
// The WebSocket subscription lives outside React so StrictMode's
// double-mount cannot open two sockets. We kick it off from a
// top-level effect and tear it down on unmount.

import { useCallback, useEffect, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Scene } from "./viewer/Scene";
import { Arm } from "./viewer/Arm";
import { SensorRays } from "./viewer/SensorRays";
import { Obstacles } from "./viewer/Obstacles";
import { startTelemetry, useTelemetryStore } from "./state/telemetry";
import { BlocklyEditor } from "./editor/BlocklyEditor";
import { RunControls } from "./editor/RunControls";
import { ManualJointPanel } from "./editor/ManualJointPanel";
import { RecorderPanel } from "./recorder/RecorderPanel";
import { ModeBanner } from "./mode/ModeBanner";
import { api, type DeployMode } from "./state/api";

export function App() {
  useEffect(() => startTelemetry(), []);

  const connected = useTelemetryStore((s) => s.connected);
  const joints = useTelemetryStore((s) => s.joints);
  const tof = useTelemetryStore((s) => s.tof);
  const ir = useTelemetryStore((s) => s.ir);
  const status = useTelemetryStore((s) => s.status);

  const [code, setCode] = useState<string>("");
  const [savedStates, setSavedStates] = useState<string[]>(["HOME"]);
  const [mode, setMode] = useState<DeployMode>("sim");

  // Left-sidebar layout. The width is adjustable via a drag handle on
  // the sidebar's right edge; the whole panel can also be collapsed to
  // zero width via the topbar toggle. Both pieces of state live here
  // so the topbar button can toggle the sidebar regardless of whether
  // the sidebar itself is rendered.
  const [leftWidth, setLeftWidth] = useState<number>(480);
  const [leftCollapsed, setLeftCollapsed] = useState<boolean>(false);

  const onLeftResizeStart = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      // Guard against a no-op drag while the panel is collapsed.
      if (leftCollapsed) return;
      e.preventDefault();
      const startX = e.clientX;
      const startW = leftWidth;
      const onMove = (ev: PointerEvent) => {
        const next = startW + (ev.clientX - startX);
        // Clamp to a sensible range: wide enough to show the block
        // toolbox, narrow enough to leave room for the viewer.
        setLeftWidth(Math.max(240, Math.min(900, next)));
      };
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      document.body.style.cursor = "ew-resize";
      document.body.style.userSelect = "none";
    },
    [leftCollapsed, leftWidth],
  );

  const shellStyle = {
    gridTemplateColumns: `${leftCollapsed ? 0 : leftWidth}px 1fr 280px`,
  } as const;

  const refreshSavedStates = useCallback(async () => {
    try {
      const entries = await api.listStates();
      const names = entries.map((e) => e.name);
      setSavedStates(names.length > 0 ? names : ["HOME"]);
    } catch {
      // Leave the existing list alone on error — the backend may not
      // be running (e.g. under `npm run test:watch`).
    }
  }, []);

  const refreshMode = useCallback(async () => {
    try {
      setMode(await api.getMode());
    } catch {
      // Backend down — keep the last known value. Tests rely on this
      // not blowing up the render tree.
    }
  }, []);

  useEffect(() => {
    void refreshSavedStates();
    void refreshMode();
  }, [refreshSavedStates, refreshMode]);

  return (
    <div className="app-shell" style={shellStyle}>
      <header className="app-shell__topbar">
        <div className="app-shell__topbar-left">
          <button
            type="button"
            className="topbar-toggle"
            aria-label={leftCollapsed ? "Show blocks panel" : "Hide blocks panel"}
            title={leftCollapsed ? "Show blocks panel" : "Hide blocks panel"}
            data-testid="sidebar-toggle"
            onClick={() => setLeftCollapsed((c) => !c)}
          >
            {leftCollapsed ? "›" : "‹"}
          </button>
          <span
            className={`status-dot ${
              connected ? "status-dot--ok" : "status-dot--bad"
            }`}
          />
          Braccio Digital Twin
        </div>
        <ModeBanner mode={mode} onModeChange={setMode} />
        <div>
          {status?.running
            ? `running: line ${status.line} (${status.kind})`
            : "idle"}
        </div>
      </header>

      <aside
        className={`app-shell__left${
          leftCollapsed ? " app-shell__left--collapsed" : ""
        }`}
      >
        <div className="app-shell__left-header">
          <div className="panel-heading">Blocks</div>
          <RunControls code={code} />
        </div>
        <div className="app-shell__left-workspace">
          <BlocklyEditor
            savedStates={savedStates}
            onCodeChange={setCode}
            mode={mode}
          />
        </div>
        <div
          className="app-shell__resize-handle"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize blocks panel"
          data-testid="sidebar-resize-handle"
          onPointerDown={onLeftResizeStart}
        />
      </aside>

      <main className="app-shell__viewer">
        <Scene>
          <Arm />
          <SensorRays />
          <Obstacles />
        </Scene>
      </main>

      <aside className="app-shell__right">
        <div className="panel-heading">Joints</div>
        {(["B", "S", "E", "WV", "WR", "G"] as const).map((tok, i) => (
          <div className="joint-row" key={tok}>
            <span>{tok}</span>
            <span>{joints[i]}°</span>
          </div>
        ))}

        <div className="panel-heading">ToF (mm)</div>
        {[0, 1, 2, 3].map((ch) => (
          <div className="joint-row" key={ch}>
            <span>CH{ch}</span>
            <span>{Math.round(tof[ch] ?? 9999)}</span>
          </div>
        ))}

        <div className="panel-heading">IR</div>
        <div className="joint-row">
          <span>severity</span>
          <span>{ir}</span>
        </div>

        <ManualJointPanel onStateSaved={() => void refreshSavedStates()} />

        <RecorderPanel filenamePrefix="braccio_demo" />
      </aside>
    </div>
  );
}
