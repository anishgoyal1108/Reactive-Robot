// BlocklyEditor.tsx — React wrapper around a full Blockly workspace.
//
// The component mounts a <div> target, calls Blockly.inject() once,
// wires a CHANGE listener that regenerates DSL text on every edit,
// and exposes the generated text via an `onCodeChange` callback.
//
// Rendering-free parts (block definitions, toolbox, DSL generator) are
// all in sibling files so they're unit-testable in jsdom without
// needing a real WebGL canvas.

import { useEffect, useRef } from "react";
import * as Blockly from "blockly/core";
import "blockly/blocks"; // built-in math/text/variables blocks
import { defineBraccioBlocks, setSavedStates } from "./blocks";
import { dslGenerator, workspaceToDsl } from "./dslGenerator";
import { buildToolbox } from "./toolbox";
import type { DeployMode } from "../state/api";

export interface BlocklyEditorProps {
  /** Called every time the generated DSL text changes. */
  onCodeChange?: (dsl: string) => void;
  /** Dropdown values for the "move to state" block. */
  savedStates?: readonly string[];
  /**
   * Active deploy mode. Controls whether the obstacle-definition
   * block is shown in the Definitions category (hidden in hardware
   * mode since the backend no-ops it).
   */
  mode?: DeployMode;
  /** Initial workspace serialization (JSON) — optional. */
  initialState?: Blockly.serialization.ISerializer | null;
}

/**
 * Register block definitions exactly once per browser session. Blockly
 * keeps a module-level `Blockly.Blocks` map so re-registering is safe
 * but wasteful — gating on this flag avoids spamming console warnings
 * during hot-reload.
 */
let _blocksDefined = false;

function ensureBlocksDefined(): void {
  if (_blocksDefined) return;
  defineBraccioBlocks();
  _blocksDefined = true;
}

export function BlocklyEditor(props: BlocklyEditorProps): JSX.Element {
  const { onCodeChange, savedStates, mode } = props;
  const targetRef = useRef<HTMLDivElement | null>(null);
  const workspaceRef = useRef<Blockly.WorkspaceSvg | null>(null);

  // Keep the saved-state dropdown in sync with props. Defined outside
  // the main effect so prop updates don't tear down the workspace.
  useEffect(() => {
    if (savedStates) setSavedStates(savedStates);
  }, [savedStates]);

  // When the deploy mode changes, rebuild the toolbox and hand it
  // back to the existing workspace. Running this in a separate effect
  // (instead of adding ``mode`` to the mount effect's deps) avoids
  // tearing down the Blockly workspace every time the user flips the
  // deploy-mode banner, which would lose block state.
  useEffect(() => {
    const workspace = workspaceRef.current;
    if (!workspace) return;
    workspace.updateToolbox(buildToolbox(mode ?? "sim"));
  }, [mode]);

  // Inject Blockly once on mount, dispose on unmount.
  useEffect(() => {
    const target = targetRef.current;
    if (!target) return;

    ensureBlocksDefined();

    const workspace = Blockly.inject(target, {
      toolbox: buildToolbox(mode ?? "sim"),
      renderer: "zelos",
      theme: Blockly.Themes.Classic,
      trashcan: true,
      zoom: {
        controls: true,
        wheel: true,
        startScale: 1.0,
        maxScale: 2.0,
        minScale: 0.5,
        scaleSpeed: 1.2,
      },
      move: {
        scrollbars: true,
        drag: true,
        wheel: false,
      },
    });
    workspaceRef.current = workspace;

    const emit = () => {
      try {
        const code = workspaceToDsl(workspace);
        onCodeChange?.(code);
      } catch (err) {
        // Block edits can briefly leave the workspace in a half-
        // parented state. Swallow the error so the UI keeps working;
        // the next CHANGE event will re-run with valid state.
        console.warn("[BlocklyEditor] code generation error:", err);
      }
    };

    workspace.addChangeListener(emit);
    // Emit once after mount so consumers see the (empty) initial code.
    emit();

    const handleResize = () => Blockly.svgResize(workspace);
    window.addEventListener("resize", handleResize);

    // Also watch the container itself — the left sidebar can be
    // dragged/collapsed without triggering a window resize, and
    // Blockly's SVG won't re-fit on its own. ResizeObserver covers
    // any layout change that moves the target's bounding box.
    const resizeObserver =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => Blockly.svgResize(workspace))
        : null;
    resizeObserver?.observe(target);

    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener("resize", handleResize);
      workspace.removeChangeListener(emit);
      workspace.dispose();
      workspaceRef.current = null;
    };
    // onCodeChange intentionally omitted — we capture it via closure
    // and re-emit on every workspace change; re-running this effect
    // would tear down the workspace on every parent render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={targetRef}
      className="blockly-workspace"
      style={{
        width: "100%",
        height: "100%",
        minHeight: 320,
      }}
    />
  );
}

/** Test-only handle so integration tests can poke the generator. */
export const _internal = { dslGenerator };
