// RunControls.tsx — Run / Stop / Validate buttons above the editor.
//
// The component is pure presentation + API wiring — it does NOT own
// the DSL text itself. The BlocklyEditor streams text up to App.tsx,
// App.tsx passes it back down via the `code` prop, and pressing Run
// posts it to /dsl/run via the injected ApiClient.
//
// Injecting the ApiClient (instead of importing the default singleton)
// keeps this component fully unit-testable in jsdom.

import { useCallback, useState } from "react";
import { ApiClient, api as defaultApi } from "../state/api";
import { isDemoMode } from "../mode/demoMode";

export interface RunControlsProps {
  /** Generated DSL text from the BlocklyEditor. */
  code: string;
  /** Override the default same-origin client in tests. */
  api?: ApiClient;
  /** Called when the user successfully kicks off a run. */
  onRunStarted?: () => void;
  /** Called when Stop is pressed. */
  onRunStopped?: () => void;
  /** Force the Copy-DSL affordance regardless of build-time flag. */
  demoMode?: boolean;
}

interface ControlState {
  busy: boolean;
  error: string | null;
  lastValidation: { ok: boolean; errors: string[] } | null;
}

const INITIAL: ControlState = {
  busy: false,
  error: null,
  lastValidation: null,
};

export function RunControls(props: RunControlsProps): JSX.Element {
  const { code, onRunStarted, onRunStopped } = props;
  const api = props.api ?? defaultApi;
  const demo = props.demoMode ?? isDemoMode();
  const [state, setState] = useState<ControlState>(INITIAL);
  const [copied, setCopied] = useState(false);

  const runCopy = useCallback(async () => {
    try {
      if (typeof navigator !== "undefined" && navigator.clipboard) {
        await navigator.clipboard.writeText(code);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      setState({
        busy: false,
        error: err instanceof Error ? err.message : String(err),
        lastValidation: null,
      });
    }
  }, [code]);

  const runValidate = useCallback(async () => {
    setState((s) => ({ ...s, busy: true, error: null }));
    try {
      const resp = await api.validateDsl(code);
      setState({ busy: false, error: null, lastValidation: resp });
    } catch (err) {
      setState({
        busy: false,
        error: err instanceof Error ? err.message : String(err),
        lastValidation: null,
      });
    }
  }, [api, code]);

  const runRun = useCallback(async () => {
    setState((s) => ({ ...s, busy: true, error: null }));
    try {
      const resp = await api.runDsl(code);
      setState({
        busy: false,
        error: null,
        lastValidation: {
          ok: resp.started,
          errors: resp.errors,
        },
      });
      if (resp.started) onRunStarted?.();
    } catch (err) {
      setState({
        busy: false,
        error: err instanceof Error ? err.message : String(err),
        lastValidation: null,
      });
    }
  }, [api, code, onRunStarted]);

  const runStop = useCallback(async () => {
    setState((s) => ({ ...s, busy: true, error: null }));
    try {
      await api.stopDsl();
      setState({ busy: false, error: null, lastValidation: null });
      onRunStopped?.();
    } catch (err) {
      setState({
        busy: false,
        error: err instanceof Error ? err.message : String(err),
        lastValidation: null,
      });
    }
  }, [api, onRunStopped]);

  const canRun = code.trim().length > 0 && !state.busy;
  const canCopy = code.trim().length > 0;

  return (
    <div className="run-controls" data-testid="run-controls">
      <button
        type="button"
        onClick={runValidate}
        disabled={!canRun}
        data-testid="run-controls-validate"
      >
        Validate
      </button>
      {demo ? (
        <button
          type="button"
          onClick={() => void runCopy()}
          disabled={!canCopy}
          data-testid="run-controls-copy"
          className="run-controls__copy"
        >
          {copied ? "Copied!" : "Copy DSL"}
        </button>
      ) : (
        <>
          <button
            type="button"
            onClick={runRun}
            disabled={!canRun}
            data-testid="run-controls-run"
          >
            Run
          </button>
          <button
            type="button"
            onClick={runStop}
            disabled={state.busy}
            data-testid="run-controls-stop"
          >
            Stop
          </button>
        </>
      )}
      {state.error && (
        <span className="run-controls__error" data-testid="run-controls-error">
          {state.error}
        </span>
      )}
      {state.lastValidation && (
        <span
          className={`run-controls__validation ${
            state.lastValidation.ok ? "ok" : "bad"
          }`}
          data-testid="run-controls-validation"
        >
          {state.lastValidation.ok
            ? "OK"
            : state.lastValidation.errors.join("; ")}
        </span>
      )}
    </div>
  );
}
