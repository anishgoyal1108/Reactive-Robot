// ModeBanner.test.tsx — click-driven tests for the Phase 7 topbar
// banner + confirmation modal.
//
// We inject ``setModeFn`` so the test never touches fetch. Each
// transition (open modal, cancel, confirm, error, concurrent busy
// state) gets its own narrow test.

import { describe, expect, it, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  act,
} from "@testing-library/react";
import { ModeBanner } from "./ModeBanner";
import type { DeployMode } from "../state/api";

function renderBanner(opts: {
  mode: DeployMode;
  onModeChange?: (m: DeployMode) => void;
  setModeFn?: (m: DeployMode) => Promise<DeployMode>;
}) {
  const onModeChange = opts.onModeChange ?? vi.fn();
  const setModeFn =
    opts.setModeFn ?? vi.fn(async (m: DeployMode) => m);
  render(
    <ModeBanner
      mode={opts.mode}
      onModeChange={onModeChange}
      setModeFn={setModeFn}
    />,
  );
  return { onModeChange, setModeFn };
}

describe("ModeBanner", () => {
  it("renders the sim-mode chip when mode is sim", () => {
    renderBanner({ mode: "sim" });
    const banner = screen.getByTestId("mode-banner");
    expect(banner).toHaveTextContent(/sim mode/i);
    expect(banner.className).toContain("mode-banner--sim");
  });

  it("renders the hardware-mode chip when mode is hardware", () => {
    renderBanner({ mode: "hardware" });
    const banner = screen.getByTestId("mode-banner");
    expect(banner).toHaveTextContent(/hardware mode/i);
    expect(banner.className).toContain("mode-banner--hardware");
  });

  it("does not show the modal on mount", () => {
    renderBanner({ mode: "sim" });
    expect(screen.queryByTestId("mode-modal")).toBeNull();
  });

  it("clicking the banner opens the modal", () => {
    renderBanner({ mode: "sim" });
    fireEvent.click(screen.getByTestId("mode-banner"));
    expect(screen.getByTestId("mode-modal")).toBeInTheDocument();
  });

  it("sim → hardware modal shows the MotionGuard safety warning", () => {
    renderBanner({ mode: "sim" });
    fireEvent.click(screen.getByTestId("mode-banner"));
    const warn = screen.getByTestId("mode-modal-warn");
    expect(warn).toHaveTextContent(/MotionGuard/);
    // Switch button is styled as danger (destructive) when going to
    // hardware.
    const confirmBtn = screen.getByTestId("mode-modal-confirm");
    expect(confirmBtn.className).toContain("modal__confirm--danger");
    expect(confirmBtn).toHaveTextContent(/switch to hardware/i);
  });

  it("hardware → sim modal does not show the danger warning", () => {
    renderBanner({ mode: "hardware" });
    fireEvent.click(screen.getByTestId("mode-banner"));
    // No MotionGuard warning on the reverse swap.
    expect(screen.queryByTestId("mode-modal-warn")).toBeNull();
    const confirmBtn = screen.getByTestId("mode-modal-confirm");
    expect(confirmBtn.className).not.toContain("modal__confirm--danger");
    expect(confirmBtn).toHaveTextContent(/switch to sim/i);
  });

  it("Cancel closes the modal without calling setModeFn", () => {
    const setModeFn = vi.fn(async (m: DeployMode) => m);
    renderBanner({ mode: "sim", setModeFn });
    fireEvent.click(screen.getByTestId("mode-banner"));
    fireEvent.click(screen.getByTestId("mode-modal-cancel"));
    expect(screen.queryByTestId("mode-modal")).toBeNull();
    expect(setModeFn).not.toHaveBeenCalled();
  });

  it("clicking the backdrop closes the modal", () => {
    renderBanner({ mode: "sim" });
    fireEvent.click(screen.getByTestId("mode-banner"));
    fireEvent.click(screen.getByTestId("mode-modal-backdrop"));
    expect(screen.queryByTestId("mode-modal")).toBeNull();
  });

  it("clicking inside the modal itself does NOT close it", () => {
    renderBanner({ mode: "sim" });
    fireEvent.click(screen.getByTestId("mode-banner"));
    fireEvent.click(screen.getByTestId("mode-modal"));
    expect(screen.getByTestId("mode-modal")).toBeInTheDocument();
  });

  it("confirm invokes setModeFn and onModeChange on success", async () => {
    const onModeChange = vi.fn();
    const setModeFn = vi.fn(async (m: DeployMode) => m);
    renderBanner({ mode: "sim", onModeChange, setModeFn });

    fireEvent.click(screen.getByTestId("mode-banner"));
    fireEvent.click(screen.getByTestId("mode-modal-confirm"));

    await waitFor(() => {
      expect(setModeFn).toHaveBeenCalledWith("hardware");
      expect(onModeChange).toHaveBeenCalledWith("hardware");
      expect(screen.queryByTestId("mode-modal")).toBeNull();
    });
  });

  it("setModeFn rejection surfaces an error banner and keeps modal open", async () => {
    const setModeFn = vi.fn(async () => {
      throw new Error("no serial port attached");
    });
    const onModeChange = vi.fn();
    renderBanner({ mode: "sim", onModeChange, setModeFn });

    fireEvent.click(screen.getByTestId("mode-banner"));
    fireEvent.click(screen.getByTestId("mode-modal-confirm"));

    await waitFor(() => {
      expect(screen.getByTestId("mode-modal-error")).toHaveTextContent(
        /no serial port attached/,
      );
    });
    // Modal still open, onModeChange never called.
    expect(screen.getByTestId("mode-modal")).toBeInTheDocument();
    expect(onModeChange).not.toHaveBeenCalled();
  });

  it("while the swap is in flight the confirm button is disabled", async () => {
    let resolveSwap: (m: DeployMode) => void = () => {};
    const pending = new Promise<DeployMode>((r) => {
      resolveSwap = r;
    });
    const setModeFn = vi.fn(() => pending);
    renderBanner({ mode: "sim", setModeFn });

    fireEvent.click(screen.getByTestId("mode-banner"));
    fireEvent.click(screen.getByTestId("mode-modal-confirm"));

    // Mid-flight: button is disabled and shows "Switching…".
    await waitFor(() => {
      expect(screen.getByTestId("mode-modal-confirm")).toBeDisabled();
      expect(screen.getByTestId("mode-modal-confirm")).toHaveTextContent(
        /switching/i,
      );
    });
    // Cancel is also disabled so the user can't bail mid-swap.
    expect(screen.getByTestId("mode-modal-cancel")).toBeDisabled();

    // Let the swap resolve and the modal closes.
    await act(async () => {
      resolveSwap("hardware");
      await pending;
    });
    await waitFor(() => {
      expect(screen.queryByTestId("mode-modal")).toBeNull();
    });
  });
});
