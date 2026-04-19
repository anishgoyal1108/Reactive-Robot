// ManualJointPanel.tsx — six sliders + "Save as state" form.
//
// Each slider is bound to a local React state vector; changing a slider
// updates the preview value immediately. "Save as state" POSTs to
// /states/{name} via the ApiClient, creating a new entry in states.json
// that the BlocklyEditor's "move to state" dropdown can then target.
//
// The panel does NOT yet live-drive the arm (Phase 0 backend adapter
// would make that trivial; wiring it in Phase 5 is out of scope).

import { FormEvent, useCallback, useState } from "react";
import { ApiClient, api as defaultApi } from "../state/api";
import { HOME_JOINTS, JOINT_TOKENS } from "../state/types";
import type { JointToken, JointVector } from "../state/types";

export interface ManualJointPanelProps {
  /** Override the default same-origin client in tests. */
  api?: ApiClient;
  /** Initial joint vector — defaults to HOME. */
  initial?: JointVector;
  /** Called after a successful save, with the saved state name. */
  onStateSaved?: (name: string) => void;
}

function cloneJoints(v: JointVector): [number, number, number, number, number, number] {
  return [v[0], v[1], v[2], v[3], v[4], v[5]];
}

export function ManualJointPanel(props: ManualJointPanelProps): JSX.Element {
  const { initial = HOME_JOINTS, onStateSaved } = props;
  const api = props.api ?? defaultApi;

  const [joints, setJoints] = useState<
    [number, number, number, number, number, number]
  >(() => cloneJoints(initial));
  const [name, setName] = useState("NEW_STATE");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const updateJoint = useCallback(
    (idx: number, value: number) => {
      setJoints((prev) => {
        const next = [...prev] as [
          number,
          number,
          number,
          number,
          number,
          number,
        ];
        next[idx] = value;
        return next;
      });
    },
    [],
  );

  const handleSave = useCallback(
    async (ev: FormEvent<HTMLFormElement>) => {
      ev.preventDefault();
      const trimmed = name.trim();
      if (!trimmed) {
        setError("State name is required");
        return;
      }
      setBusy(true);
      setError(null);
      setMessage(null);
      try {
        await api.saveState(trimmed, { joints });
        setBusy(false);
        setMessage(`Saved ${trimmed}`);
        onStateSaved?.(trimmed);
      } catch (err) {
        setBusy(false);
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [api, joints, name, onStateSaved],
  );

  return (
    <form
      className="manual-joint-panel"
      data-testid="manual-joint-panel"
      onSubmit={handleSave}
    >
      <div className="panel-heading">Manual Joints</div>
      {(JOINT_TOKENS as readonly JointToken[]).map((tok, i) => (
        <label key={tok} className="manual-joint-row">
          <span>{tok}</span>
          <input
            type="range"
            min={0}
            max={180}
            step={1}
            value={joints[i]}
            onChange={(e) => updateJoint(i, Number(e.target.value))}
            data-testid={`manual-joint-slider-${tok}`}
          />
          <span>{joints[i]}°</span>
        </label>
      ))}

      <label className="manual-joint-name">
        <span>Save as</span>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value.toUpperCase())}
          data-testid="manual-joint-name"
        />
      </label>

      <button
        type="submit"
        disabled={busy}
        data-testid="manual-joint-save"
      >
        {busy ? "Saving…" : "Save as state"}
      </button>

      {error && (
        <div
          className="manual-joint-error"
          data-testid="manual-joint-error"
        >
          {error}
        </div>
      )}
      {message && (
        <div
          className="manual-joint-message"
          data-testid="manual-joint-message"
        >
          {message}
        </div>
      )}
    </form>
  );
}
