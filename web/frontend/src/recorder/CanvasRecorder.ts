// CanvasRecorder.ts — Live Three.js canvas → WebM capture wrapper.
//
// The recorder is a pure class with no React dependencies: tests mount
// a MockMediaRecorder + MockCanvasCaptureStream and drive the lifecycle
// directly. This keeps Phase 6 testable without ever starting a real
// WebGL context.
//
// Usage::
//
//     const recorder = new CanvasRecorder({ fps: 30 });
//     recorder.start(canvasEl);           // begins capture immediately
//     // … time passes, blocks run, arm moves …
//     const blob = await recorder.stop(); // finalised WebM blob
//     recorder.download("demo.webm", blob);
//
// Matches the Phase 6 plan: "the recorder captures the live Three.js
// canvas as the sequence executes", so there is no pre-rendered file
// being played back — the WebM IS the live render.

/** Options accepted by the CanvasRecorder constructor. */
export interface CanvasRecorderOptions {
  /** Target frames-per-second for captureStream. Defaults to 30. */
  fps?: number;
  /** Preferred mimeType. Defaults to WebM VP9, falls back to VP8. */
  mimeType?: string;
  /** Injection point for tests — defaults to globalThis.MediaRecorder. */
  mediaRecorderCtor?: typeof MediaRecorder;
  /** Injection point for URL.createObjectURL — tests may stub. */
  createObjectUrl?: (blob: Blob) => string;
  /** Injection point for URL.revokeObjectURL. */
  revokeObjectUrl?: (url: string) => void;
  /**
   * Injection point for triggering a browser download. Defaults to
   * creating an invisible <a download> and clicking it.
   */
  triggerDownload?: (url: string, filename: string) => void;
}

/** Readable recorder lifecycle. */
export type RecorderState = "idle" | "recording" | "stopping";

/**
 * Pick the best-supported WebM mime type on this browser. VP9 beats
 * VP8 for size; if both are unsupported we fall back to the generic
 * "video/webm" which Chrome/Firefox always accept.
 */
function chooseMimeType(
  preferred: string | undefined,
  isTypeSupported: (t: string) => boolean = (t) =>
    typeof MediaRecorder !== "undefined" &&
    typeof MediaRecorder.isTypeSupported === "function" &&
    MediaRecorder.isTypeSupported(t),
): string {
  const candidates = [
    preferred,
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ].filter((s): s is string => typeof s === "string" && s.length > 0);
  for (const mime of candidates) {
    try {
      if (isTypeSupported(mime)) return mime;
    } catch {
      /* ignore, try next candidate */
    }
  }
  return "video/webm";
}

/**
 * Produce a timestamped default filename. Matches the plan's example
 * format (META_FULL_DEMO_2026-04-10_15-22-08.mp4) but uses .webm since
 * that's what MediaRecorder emits natively.
 */
export function defaultRecordingFilename(
  prefix = "braccio_demo",
  now: Date = new Date(),
): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const ymd = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(
    now.getDate(),
  )}`;
  const hms = `${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(
    now.getSeconds(),
  )}`;
  return `${prefix}_${ymd}_${hms}.webm`;
}

/** Pure recorder — no React, no DOM side-effects outside download(). */
export class CanvasRecorder {
  readonly fps: number;
  readonly mimeType: string;
  private readonly mediaRecorderCtor: typeof MediaRecorder | undefined;
  private readonly createObjectUrl: (blob: Blob) => string;
  private readonly revokeObjectUrl: (url: string) => void;
  private readonly triggerDownload: (url: string, filename: string) => void;

  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private _state: RecorderState = "idle";
  private pending: Promise<Blob> | null = null;
  private resolvePending: ((blob: Blob) => void) | null = null;
  private rejectPending: ((err: Error) => void) | null = null;

  constructor(options: CanvasRecorderOptions = {}) {
    this.fps = options.fps ?? 30;
    this.mediaRecorderCtor =
      options.mediaRecorderCtor ??
      (typeof MediaRecorder !== "undefined" ? MediaRecorder : undefined);
    this.mimeType = chooseMimeType(
      options.mimeType,
      this.mediaRecorderCtor &&
        typeof (this.mediaRecorderCtor as unknown as {
          isTypeSupported?: (t: string) => boolean;
        }).isTypeSupported === "function"
        ? (t) =>
            (this.mediaRecorderCtor as unknown as {
              isTypeSupported: (t: string) => boolean;
            }).isTypeSupported(t)
        : () => true,
    );
    this.createObjectUrl =
      options.createObjectUrl ??
      ((blob: Blob) =>
        typeof URL !== "undefined" ? URL.createObjectURL(blob) : "");
    this.revokeObjectUrl =
      options.revokeObjectUrl ??
      ((url: string) => {
        if (typeof URL !== "undefined") URL.revokeObjectURL(url);
      });
    this.triggerDownload =
      options.triggerDownload ??
      ((url: string, filename: string) => {
        if (typeof document === "undefined") return;
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      });
  }

  get state(): RecorderState {
    return this._state;
  }

  /** True if the browser can actually record. */
  get supported(): boolean {
    return typeof this.mediaRecorderCtor === "function";
  }

  /**
   * Begin capturing from the given canvas. Throws synchronously if the
   * browser lacks MediaRecorder OR if a capture is already in flight.
   */
  start(canvas: HTMLCanvasElement): void {
    if (this._state !== "idle") {
      throw new Error(`CanvasRecorder.start: already ${this._state}`);
    }
    if (!this.supported) {
      throw new Error("CanvasRecorder.start: MediaRecorder not supported");
    }
    const grab = (
      canvas as unknown as {
        captureStream?: (fps: number) => MediaStream;
      }
    ).captureStream;
    if (typeof grab !== "function") {
      throw new Error(
        "CanvasRecorder.start: canvas.captureStream() unavailable",
      );
    }
    const stream = grab.call(canvas, this.fps) as MediaStream;
    const MR = this.mediaRecorderCtor as typeof MediaRecorder;
    const rec = new MR(stream, { mimeType: this.mimeType });
    this.chunks = [];

    this.pending = new Promise<Blob>((resolve, reject) => {
      this.resolvePending = resolve;
      this.rejectPending = reject;
    });

    rec.ondataavailable = (ev: BlobEvent) => {
      if (ev.data && ev.data.size > 0) {
        this.chunks.push(ev.data);
      }
    };
    rec.onstop = () => {
      const blob = new Blob(this.chunks, { type: this.mimeType });
      this._state = "idle";
      this.recorder = null;
      this.resolvePending?.(blob);
      this.resolvePending = null;
      this.rejectPending = null;
    };
    rec.onerror = (ev: Event) => {
      this._state = "idle";
      this.recorder = null;
      const err =
        (ev as unknown as { error?: Error }).error ??
        new Error("MediaRecorder error");
      this.rejectPending?.(err);
      this.resolvePending = null;
      this.rejectPending = null;
    };

    rec.start();
    this.recorder = rec;
    this._state = "recording";
  }

  /**
   * Stop the in-flight capture and resolve with the finalised blob.
   * Calling stop() while idle resolves immediately with an empty blob
   * so UI code can be trigger-happy.
   */
  stop(): Promise<Blob> {
    if (this._state === "idle") {
      return Promise.resolve(new Blob([], { type: this.mimeType }));
    }
    if (!this.recorder || !this.pending) {
      return Promise.resolve(new Blob([], { type: this.mimeType }));
    }
    this._state = "stopping";
    try {
      this.recorder.stop();
    } catch (err) {
      this._state = "idle";
      return Promise.reject(err instanceof Error ? err : new Error(String(err)));
    }
    return this.pending;
  }

  /** Force-dispose the recorder without awaiting a blob. */
  dispose(): void {
    if (this.recorder) {
      try {
        this.recorder.stop();
      } catch {
        /* ignore — recorder may already be idle */
      }
    }
    this.recorder = null;
    this._state = "idle";
    this.chunks = [];
    this.pending = null;
    this.resolvePending = null;
    this.rejectPending = null;
  }

  /** Kick off a browser download for the given blob. */
  download(filename: string, blob: Blob): void {
    const url = this.createObjectUrl(blob);
    this.triggerDownload(url, filename);
    // Revoke on next tick so the download has time to start.
    setTimeout(() => this.revokeObjectUrl(url), 1000);
  }
}

/** Test-only exports. */
export const _internal = { chooseMimeType };
