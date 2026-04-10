// RunControls.test.tsx — click-driven tests for Run / Stop / Validate.
//
// We inject a stub ApiClient via `makeStubApi()` so there are no
// network calls and tests run in jsdom without a backend.

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { RunControls } from "./RunControls";
import type { ApiClient } from "../state/api";

interface StubCalls {
  validate: string[];
  run: string[];
  stop: number;
}

function makeStubApi(overrides: Partial<ApiClient> = {}): {
  api: ApiClient;
  calls: StubCalls;
} {
  const calls: StubCalls = { validate: [], run: [], stop: 0 };
  const api = {
    validateDsl: vi.fn(async (text: string) => {
      calls.validate.push(text);
      return { ok: true, errors: [] };
    }),
    runDsl: vi.fn(async (text: string) => {
      calls.run.push(text);
      return { started: true, errors: [] };
    }),
    stopDsl: vi.fn(async () => {
      calls.stop += 1;
      return true;
    }),
    ...overrides,
  } as unknown as ApiClient;
  return { api, calls };
}

describe("RunControls", () => {
  it("renders three buttons", () => {
    const { api } = makeStubApi();
    render(<RunControls code={"HOME\n"} api={api} />);
    expect(screen.getByTestId("run-controls-validate")).toBeInTheDocument();
    expect(screen.getByTestId("run-controls-run")).toBeInTheDocument();
    expect(screen.getByTestId("run-controls-stop")).toBeInTheDocument();
  });

  it("disables Run and Validate when code is empty", () => {
    const { api } = makeStubApi();
    render(<RunControls code="" api={api} />);
    expect(screen.getByTestId("run-controls-run")).toBeDisabled();
    expect(screen.getByTestId("run-controls-validate")).toBeDisabled();
    // Stop is always enabled so the user can cancel a run started
    // from another panel.
    expect(screen.getByTestId("run-controls-stop")).not.toBeDisabled();
  });

  it("Validate calls api.validateDsl with the current code", async () => {
    const { api, calls } = makeStubApi();
    render(<RunControls code={"MOVE HOME\n"} api={api} />);
    fireEvent.click(screen.getByTestId("run-controls-validate"));

    await waitFor(() => {
      expect(calls.validate).toEqual(["MOVE HOME\n"]);
    });
    expect(screen.getByTestId("run-controls-validation")).toHaveTextContent(
      /OK/i,
    );
  });

  it("Run calls api.runDsl and fires onRunStarted", async () => {
    const { api, calls } = makeStubApi();
    const onRunStarted = vi.fn();
    render(
      <RunControls
        code={"MOVE HOME\n"}
        api={api}
        onRunStarted={onRunStarted}
      />,
    );
    fireEvent.click(screen.getByTestId("run-controls-run"));

    await waitFor(() => {
      expect(calls.run).toEqual(["MOVE HOME\n"]);
      expect(onRunStarted).toHaveBeenCalledTimes(1);
    });
  });

  it("Stop calls api.stopDsl and fires onRunStopped", async () => {
    const { api, calls } = makeStubApi();
    const onRunStopped = vi.fn();
    render(
      <RunControls
        code={"MOVE HOME\n"}
        api={api}
        onRunStopped={onRunStopped}
      />,
    );
    fireEvent.click(screen.getByTestId("run-controls-stop"));

    await waitFor(() => {
      expect(calls.stop).toBe(1);
      expect(onRunStopped).toHaveBeenCalledTimes(1);
    });
  });

  it("surfaces validation errors from the backend", async () => {
    const api = {
      validateDsl: vi.fn(async () => ({
        ok: false,
        errors: ["line 1: expected MOVE"],
      })),
      runDsl: vi.fn(),
      stopDsl: vi.fn(),
    } as unknown as ApiClient;

    render(<RunControls code={"OOPS\n"} api={api} />);
    fireEvent.click(screen.getByTestId("run-controls-validate"));

    await waitFor(() => {
      expect(screen.getByTestId("run-controls-validation")).toHaveTextContent(
        "line 1: expected MOVE",
      );
    });
  });

  it("surfaces network errors from Run", async () => {
    const api = {
      validateDsl: vi.fn(),
      runDsl: vi.fn(async () => {
        throw new Error("ECONNREFUSED");
      }),
      stopDsl: vi.fn(),
    } as unknown as ApiClient;

    render(<RunControls code={"HOME\n"} api={api} />);
    fireEvent.click(screen.getByTestId("run-controls-run"));

    await waitFor(() => {
      expect(screen.getByTestId("run-controls-error")).toHaveTextContent(
        "ECONNREFUSED",
      );
    });
  });
});
