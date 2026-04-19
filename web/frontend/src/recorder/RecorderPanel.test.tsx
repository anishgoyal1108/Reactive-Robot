// RecorderPanel.test.tsx — click-driven tests for Record / Stop / Save.
//
// We inject a stub CanvasRecorder (not the real one) plus a stub
// canvasProvider so the panel never touches a real <canvas> and jsdom
// stays happy. Each test exercises one transition in the lifecycle.

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { RecorderPanel, formatBlobSize } from "./RecorderPanel";
import type { CanvasRecorder } from "./CanvasRecorder";

interface StubState {
  startCalls: HTMLCanvasElement[];
  stopCalls: number;
  downloadCalls: Array<{ filename: string; blob: Blob }>;
  nextStartError: Error | null;
  nextStopBlob: Blob;
}

function makeStubRecorder(overrides: Partial<StubState> = {}): {
  recorder: CanvasRecorder;
  calls: StubState;
} {
  const calls: StubState = {
    startCalls: [],
    stopCalls: 0,
    downloadCalls: [],
    nextStartError: null,
    nextStopBlob: new Blob(["video-bytes"], { type: "video/webm" }),
    ...overrides,
  };
  const recorder = {
    start: vi.fn((canvas: HTMLCanvasElement) => {
      if (calls.nextStartError) throw calls.nextStartError;
      calls.startCalls.push(canvas);
    }),
    stop: vi.fn(async () => {
      calls.stopCalls += 1;
      return calls.nextStopBlob;
    }),
    download: vi.fn((filename: string, blob: Blob) => {
      calls.downloadCalls.push({ filename, blob });
    }),
    dispose: vi.fn(),
    get state() {
      return "idle" as const;
    },
    get supported() {
      return true;
    },
  } as unknown as CanvasRecorder;
  return { recorder, calls };
}

function makeCanvas(): HTMLCanvasElement {
  return document.createElement("canvas");
}

describe("formatBlobSize", () => {
  it("returns '0 B' for empty/negative blobs", () => {
    expect(formatBlobSize(0)).toBe("0 B");
    expect(formatBlobSize(-5)).toBe("0 B");
  });

  it("prints bytes for tiny blobs (no rounding to 0 KB)", () => {
    expect(formatBlobSize(1)).toBe("1 B");
    expect(formatBlobSize(110)).toBe("110 B");
    expect(formatBlobSize(1023)).toBe("1023 B");
  });

  it("prints KB for blobs >= 1 KB with one decimal place", () => {
    expect(formatBlobSize(1024)).toBe("1.0 KB");
    expect(formatBlobSize(1536)).toBe("1.5 KB");
    expect(formatBlobSize(1024 * 1024 - 1)).toMatch(/KB$/);
  });

  it("prints MB for blobs >= 1 MB", () => {
    expect(formatBlobSize(1024 * 1024)).toBe("1.0 MB");
    expect(formatBlobSize(5 * 1024 * 1024 + 512 * 1024)).toBe("5.5 MB");
  });
});

describe("RecorderPanel", () => {
  it("renders the three buttons with initial idle state", () => {
    const { recorder } = makeStubRecorder();
    const canvas = makeCanvas();
    render(
      <RecorderPanel recorder={recorder} canvasProvider={() => canvas} />,
    );
    expect(screen.getByTestId("recorder-panel-record")).not.toBeDisabled();
    expect(screen.getByTestId("recorder-panel-stop")).toBeDisabled();
    expect(screen.getByTestId("recorder-panel-download")).toBeDisabled();
    expect(screen.getByTestId("recorder-panel-status")).toHaveTextContent(
      "idle",
    );
  });

  it("shows an error when the canvas is not ready", () => {
    const { recorder, calls } = makeStubRecorder();
    render(<RecorderPanel recorder={recorder} canvasProvider={() => null} />);

    fireEvent.click(screen.getByTestId("recorder-panel-record"));

    expect(calls.startCalls).toHaveLength(0);
    expect(screen.getByTestId("recorder-panel-error")).toHaveTextContent(
      /viewer canvas not ready/i,
    );
  });

  it("starts recording and toggles button states", () => {
    const { recorder, calls } = makeStubRecorder();
    const canvas = makeCanvas();
    render(
      <RecorderPanel recorder={recorder} canvasProvider={() => canvas} />,
    );

    fireEvent.click(screen.getByTestId("recorder-panel-record"));

    expect(calls.startCalls).toEqual([canvas]);
    expect(screen.getByTestId("recorder-panel-record")).toBeDisabled();
    expect(screen.getByTestId("recorder-panel-stop")).not.toBeDisabled();
    expect(screen.getByTestId("recorder-panel-status")).toHaveTextContent(
      /recording/i,
    );
  });

  it("surfaces recorder.start() failures as an error banner", () => {
    const { recorder } = makeStubRecorder({
      nextStartError: new Error("captureStream unavailable"),
    });
    const canvas = makeCanvas();
    render(
      <RecorderPanel recorder={recorder} canvasProvider={() => canvas} />,
    );

    fireEvent.click(screen.getByTestId("recorder-panel-record"));

    expect(screen.getByTestId("recorder-panel-error")).toHaveTextContent(
      /captureStream unavailable/,
    );
    // Still idle since start threw.
    expect(screen.getByTestId("recorder-panel-record")).not.toBeDisabled();
    expect(screen.getByTestId("recorder-panel-stop")).toBeDisabled();
  });

  it("Record → Stop → Save flow produces a download with the expected blob", async () => {
    const blob = new Blob(["webm-bytes-xyz"], { type: "video/webm" });
    const { recorder, calls } = makeStubRecorder({ nextStopBlob: blob });
    const canvas = makeCanvas();
    render(
      <RecorderPanel
        recorder={recorder}
        canvasProvider={() => canvas}
        filenamePrefix="unit_test"
      />,
    );

    fireEvent.click(screen.getByTestId("recorder-panel-record"));
    fireEvent.click(screen.getByTestId("recorder-panel-stop"));

    await waitFor(() => {
      expect(calls.stopCalls).toBe(1);
      expect(screen.getByTestId("recorder-panel-download")).not.toBeDisabled();
    });

    // Status reports the captured size in a human-readable format.
    expect(screen.getByTestId("recorder-panel-status")).toHaveTextContent(
      /captured \d+(?:\.\d+)? (?:B|KB|MB)/i,
    );

    fireEvent.click(screen.getByTestId("recorder-panel-download"));

    expect(calls.downloadCalls).toHaveLength(1);
    expect(calls.downloadCalls[0].blob).toBe(blob);
    expect(calls.downloadCalls[0].filename).toMatch(/^unit_test_.*\.webm$/);
  });

  it("Download button is a no-op when no blob has been captured", () => {
    const { recorder, calls } = makeStubRecorder();
    const canvas = makeCanvas();
    render(
      <RecorderPanel recorder={recorder} canvasProvider={() => canvas} />,
    );
    // Never recorded — download button should be disabled, and clicking
    // it (in case disabled styling is bypassed) must not invoke download.
    const btn = screen.getByTestId("recorder-panel-download");
    expect(btn).toBeDisabled();
    fireEvent.click(btn);
    expect(calls.downloadCalls).toHaveLength(0);
  });

  it("surfaces recorder.stop() rejections as an error banner", async () => {
    const { recorder } = makeStubRecorder();
    (recorder.stop as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      async () => {
        throw new Error("flush failed");
      },
    );
    const canvas = makeCanvas();
    render(
      <RecorderPanel recorder={recorder} canvasProvider={() => canvas} />,
    );

    fireEvent.click(screen.getByTestId("recorder-panel-record"));
    fireEvent.click(screen.getByTestId("recorder-panel-stop"));

    await waitFor(() => {
      expect(screen.getByTestId("recorder-panel-error")).toHaveTextContent(
        /flush failed/,
      );
    });
    // Returns to idle after the error.
    expect(screen.getByTestId("recorder-panel-record")).not.toBeDisabled();
  });
});
