// ManualJointPanel.test.tsx — slider + save flow tests.

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ManualJointPanel } from "./ManualJointPanel";
import type { ApiClient } from "../state/api";
import type { StateEntry } from "../state/types";

interface SaveCall {
  name: string;
  entry: Partial<StateEntry>;
}

function makeStubApi(throwOnSave = false): {
  api: ApiClient;
  calls: SaveCall[];
} {
  const calls: SaveCall[] = [];
  const api = {
    saveState: vi.fn(async (name: string, entry: Partial<StateEntry>) => {
      if (throwOnSave) throw new Error("bad request");
      calls.push({ name, entry });
      return { name, joints: entry.joints ?? [] } as StateEntry;
    }),
  } as unknown as ApiClient;
  return { api, calls };
}

describe("ManualJointPanel", () => {
  it("renders six sliders plus a save form", () => {
    const { api } = makeStubApi();
    render(<ManualJointPanel api={api} />);
    for (const tok of ["B", "S", "E", "WV", "WR", "G"]) {
      expect(
        screen.getByTestId(`manual-joint-slider-${tok}`),
      ).toBeInTheDocument();
    }
    expect(screen.getByTestId("manual-joint-name")).toBeInTheDocument();
    expect(screen.getByTestId("manual-joint-save")).toBeInTheDocument();
  });

  it("starts with HOME joints by default", () => {
    const { api } = makeStubApi();
    render(<ManualJointPanel api={api} />);
    const b = screen.getByTestId("manual-joint-slider-B") as HTMLInputElement;
    expect(b.value).toBe("90");
    const g = screen.getByTestId("manual-joint-slider-G") as HTMLInputElement;
    expect(g.value).toBe("73");
  });

  it("updates joint value on slider change", () => {
    const { api } = makeStubApi();
    render(<ManualJointPanel api={api} />);
    const slider = screen.getByTestId(
      "manual-joint-slider-B",
    ) as HTMLInputElement;
    fireEvent.change(slider, { target: { value: "45" } });
    expect(slider.value).toBe("45");
  });

  it("uppercases the state name as the user types", () => {
    const { api } = makeStubApi();
    render(<ManualJointPanel api={api} />);
    const input = screen.getByTestId(
      "manual-joint-name",
    ) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "lean_left" } });
    expect(input.value).toBe("LEAN_LEFT");
  });

  it("calls api.saveState with the current joints", async () => {
    const { api, calls } = makeStubApi();
    const onSaved = vi.fn();
    render(<ManualJointPanel api={api} onStateSaved={onSaved} />);

    fireEvent.change(screen.getByTestId("manual-joint-slider-S"), {
      target: { value: "45" },
    });
    fireEvent.change(screen.getByTestId("manual-joint-name"), {
      target: { value: "TEST_STATE" },
    });
    fireEvent.click(screen.getByTestId("manual-joint-save"));

    await waitFor(() => {
      expect(calls).toHaveLength(1);
    });
    expect(calls[0].name).toBe("TEST_STATE");
    expect(calls[0].entry.joints).toEqual([90, 45, 90, 90, 90, 73]);
    expect(onSaved).toHaveBeenCalledWith("TEST_STATE");

    await waitFor(() => {
      expect(screen.getByTestId("manual-joint-message")).toHaveTextContent(
        "Saved TEST_STATE",
      );
    });
  });

  it("surfaces errors from api.saveState", async () => {
    const { api } = makeStubApi(true);
    render(<ManualJointPanel api={api} />);

    fireEvent.click(screen.getByTestId("manual-joint-save"));
    await waitFor(() => {
      expect(screen.getByTestId("manual-joint-error")).toHaveTextContent(
        "bad request",
      );
    });
  });

  it("rejects blank state names with a local error", async () => {
    const { api, calls } = makeStubApi();
    render(<ManualJointPanel api={api} />);
    fireEvent.change(screen.getByTestId("manual-joint-name"), {
      target: { value: "   " },
    });
    fireEvent.click(screen.getByTestId("manual-joint-save"));

    await waitFor(() => {
      expect(screen.getByTestId("manual-joint-error")).toHaveTextContent(
        /required/i,
      );
    });
    expect(calls).toHaveLength(0);
  });
});
