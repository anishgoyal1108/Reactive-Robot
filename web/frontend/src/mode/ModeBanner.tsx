// ModeBanner.tsx — Topbar widget that shows the active deploy mode
// and pops a confirmation modal before flipping sim ↔ hardware.
//
// Phase 7. The banner lives in the <header> strip between the
// connection indicator and the running-sequence status. Clicking it
// opens an inline modal — we deliberately avoid portals / dialogs
// that would require a global container, so jsdom tests can mount
// the banner in isolation without a document body provider.
//
// The modal warns the user that:
//   * hardware mode actually drives /dev/ttyACM0 (not the sim);
//   * the web backend does NOT go through MotionGuard — reactive
//     synchronous obstacle gating only exists in the curses
//     controller. This is the caveat called out in
//     feedback_synchronous_safety.md and bridge.WebBridge's docstring.
//
// The component is kept small and explicit. All network I/O is
// injected via the optional ``setModeFn`` prop so tests never touch
// fetch.

import { useCallback, useState } from "react";
import { api, type DeployMode } from "../state/api";

export interface ModeBannerProps {
  /** Current mode — the parent owns the value so /health can refresh it. */
  mode: DeployMode;
  /** Called with the new mode after a successful swap. */
  onModeChange: (next: DeployMode) => void;
  /** Injectable network call. Defaults to ``api.setMode``. */
  setModeFn?: (mode: DeployMode) => Promise<DeployMode>;
}

export function ModeBanner(props: ModeBannerProps): JSX.Element {
  const { mode, onModeChange } = props;
  const setModeFn = props.setModeFn ?? ((m: DeployMode) => api.setMode(m));

  const [modalTarget, setModalTarget] = useState<DeployMode | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const other: DeployMode = mode === "sim" ? "hardware" : "sim";

  const openModal = useCallback(() => {
    setError(null);
    setModalTarget(other);
  }, [other]);

  const closeModal = useCallback(() => {
    if (busy) return;
    setModalTarget(null);
    setError(null);
  }, [busy]);

  const confirm = useCallback(async () => {
    if (modalTarget === null) return;
    setBusy(true);
    setError(null);
    try {
      const applied = await setModeFn(modalTarget);
      onModeChange(applied);
      setModalTarget(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [modalTarget, onModeChange, setModeFn]);

  const chipLabel =
    mode === "hardware"
      ? "🔧 Hardware mode — /dev/ttyACM0"
      : "🧪 Sim mode — virtual arm";

  return (
    <>
      <button
        type="button"
        className={`mode-banner mode-banner--${mode}`}
        data-testid="mode-banner"
        onClick={openModal}
      >
        <span className="mode-banner__label">{chipLabel}</span>
        <span className="mode-banner__switch">switch</span>
      </button>

      {modalTarget !== null && (
        <div
          className="modal-backdrop"
          data-testid="mode-modal-backdrop"
          onClick={closeModal}
        >
          <div
            className="modal"
            data-testid="mode-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal__title">
              Switch to{" "}
              <strong>
                {modalTarget === "hardware" ? "Hardware" : "Simulation"}
              </strong>
              ?
            </div>
            <div className="modal__body">
              {modalTarget === "hardware" ? (
                <>
                  <p>
                    The next Run command will move the <strong>real</strong>{" "}
                    Braccio arm connected to <code>/dev/ttyACM0</code>.
                  </p>
                  <p className="modal__warn" data-testid="mode-modal-warn">
                    ⚠ Reactive obstacle gating (<code>MotionGuard</code>) is
                    only enforced by the curses controller. Sequences run from
                    this web UI do <strong>not</strong> re-check ToF before
                    each command. Keep the work area clear and be ready to
                    hit Stop.
                  </p>
                </>
              ) : (
                <p>
                  Commands will go to the virtual arm instead of{" "}
                  <code>/dev/ttyACM0</code>. Obstacle definitions come back
                  into scope.
                </p>
              )}
              {error && (
                <div className="modal__error" data-testid="mode-modal-error">
                  {error}
                </div>
              )}
            </div>
            <div className="modal__actions">
              <button
                type="button"
                onClick={closeModal}
                disabled={busy}
                data-testid="mode-modal-cancel"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void confirm()}
                disabled={busy}
                data-testid="mode-modal-confirm"
                className={
                  modalTarget === "hardware"
                    ? "modal__confirm modal__confirm--danger"
                    : "modal__confirm"
                }
              >
                {busy
                  ? "Switching…"
                  : modalTarget === "hardware"
                    ? "Switch to hardware"
                    : "Switch to sim"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
