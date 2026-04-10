// App.tsx — Phase 4 top-level layout.
//
// Three columns:
//   left    — state library panel (placeholder, filled in Phase 5)
//   center  — Three.js viewer with Scene + Arm + SensorRays + Obstacles
//   right   — live telemetry readouts (joints, ToF, IR, status)
//
// The WebSocket subscription lives outside React so StrictMode's
// double-mount cannot open two sockets. We kick it off from a
// top-level effect and tear it down on unmount.

import { useEffect } from "react";
import { Scene } from "./viewer/Scene";
import { Arm } from "./viewer/Arm";
import { SensorRays } from "./viewer/SensorRays";
import { Obstacles } from "./viewer/Obstacles";
import { startTelemetry, useTelemetryStore } from "./state/telemetry";

export function App() {
  useEffect(() => startTelemetry(), []);

  const connected = useTelemetryStore((s) => s.connected);
  const joints = useTelemetryStore((s) => s.joints);
  const tof = useTelemetryStore((s) => s.tof);
  const ir = useTelemetryStore((s) => s.ir);
  const status = useTelemetryStore((s) => s.status);

  return (
    <div className="app-shell">
      <header className="app-shell__topbar">
        <div>
          <span
            className={`status-dot ${
              connected ? "status-dot--ok" : "status-dot--bad"
            }`}
          />
          Braccio Digital Twin
        </div>
        <div>
          {status?.running
            ? `running: line ${status.line} (${status.kind})`
            : "idle"}
        </div>
      </header>

      <aside className="app-shell__left">
        <div className="panel-heading">Saved States</div>
        <div style={{ color: "#9ca3af", fontSize: 13 }}>
          Phase 5 will add a live list + "load into workspace" buttons.
        </div>
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
      </aside>
    </div>
  );
}
