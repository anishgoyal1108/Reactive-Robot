// RecorderPanel.tsx — Record / Stop / Download UI for the Three.js
// viewer.
//
// The panel takes an injectable CanvasRecorder + canvas getter so
// tests never have to mount a real <Canvas>. In production the
// canvas is located via a DOM selector (the r3f renderer appends
// its <canvas> to the viewer column at mount time, so a simple
// querySelector finds it without any r3f plumbing).

import { useCallback, useState } from "react";
import {
  CanvasRecorder,
  defaultRecordingFilename,
} from "./CanvasRecorder";

export interface RecorderPanelProps {
  /** Override the default recorder for tests. */
  recorder?: CanvasRecorder;
  /** Override the canvas source for tests. */
  canvasProvider?: () => HTMLCanvasElement | null;
  /**
   * CSS selector for the live viewer canvas when the default
   * provider is used. Defaults to `.app-shell__viewer canvas`,
   * which matches the r3f canvas mounted by `<Scene />` inside the
   * center column.
   */
  canvasSelector?: string;
  /** Filename prefix — defaults to "braccio_demo". */
  filenamePrefix?: string;
}

/** Lazy-constructed singleton so production doesn't allocate per-render. */
let _defaultRecorder: CanvasRecorder | null = null;

function getDefaultRecorder(): CanvasRecorder {
  if (!_defaultRecorder) _defaultRecorder = new CanvasRecorder();
  return _defaultRecorder;
}

/** Test-only reset hook. */
export function _resetDefaultRecorder(): void {
  _defaultRecorder = null;
}

/** Human-readable size for the status chip — always non-zero for non-empty blobs. */
export function formatBlobSize(bytes: number): string {
  if (bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function RecorderPanel(props: RecorderPanelProps): JSX.Element {
  const recorder = props.recorder ?? getDefaultRecorder();
  const selector = props.canvasSelector ?? ".app-shell__viewer canvas";
  const canvasProvider =
    props.canvasProvider ??
    (() =>
      typeof document !== "undefined"
        ? (document.querySelector(selector) as HTMLCanvasElement | null)
        : null);
  const prefix = props.filenamePrefix ?? "braccio_demo";

  const [state, setState] = useState<
    "idle" | "recording" | "stopping"
  >("idle");
  const [error, setError] = useState<string | null>(null);
  const [lastBlob, setLastBlob] = useState<Blob | null>(null);

  const handleRecord = useCallback(() => {
    setError(null);
    setLastBlob(null);
    const canvas = canvasProvider();
    if (!canvas) {
      setError("viewer canvas not ready");
      return;
    }
    try {
      recorder.start(canvas);
      setState("recording");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [canvasProvider, recorder]);

  const handleStop = useCallback(async () => {
    setState("stopping");
    try {
      const blob = await recorder.stop();
      setLastBlob(blob);
      setState("idle");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("idle");
    }
  }, [recorder]);

  const handleDownload = useCallback(() => {
    if (!lastBlob) return;
    recorder.download(defaultRecordingFilename(prefix), lastBlob);
  }, [lastBlob, prefix, recorder]);

  return (
    <div className="recorder-panel" data-testid="recorder-panel">
      <div className="panel-heading">Record</div>
      <div className="recorder-panel__row">
        <button
          type="button"
          onClick={handleRecord}
          disabled={state !== "idle"}
          data-testid="recorder-panel-record"
        >
          ● Record
        </button>
        <button
          type="button"
          onClick={handleStop}
          disabled={state !== "recording"}
          data-testid="recorder-panel-stop"
        >
          ■ Stop
        </button>
        <button
          type="button"
          onClick={handleDownload}
          disabled={!lastBlob}
          data-testid="recorder-panel-download"
        >
          ⬇ Save
        </button>
      </div>

      <div
        className={`recorder-panel__status status-dot ${
          state === "recording"
            ? "status-dot--bad"
            : lastBlob
              ? "status-dot--ok"
              : "status-dot--warn"
        }`}
        data-testid="recorder-panel-status"
      >
        {state === "recording"
          ? "recording…"
          : state === "stopping"
            ? "finishing…"
            : lastBlob
              ? `captured ${formatBlobSize(lastBlob.size)}`
              : "idle"}
      </div>

      {error && (
        <div
          className="recorder-panel__error"
          data-testid="recorder-panel-error"
        >
          {error}
        </div>
      )}
    </div>
  );
}
