// toolbox.ts — Blockly toolbox (left-hand block palette).
//
// Split into five categories mapping to the DSL:
//   Movement   — MOVE / SET JOINT / WAIT / HOME
//   Control    — REPEAT / IF / IF/ELSE / WHILE
//   Sensing    — condition blocks (TOF / IR / JOINT)
//   Definitions— STATE / OBSTACLE
//
// The Blockly JSON toolbox format is stable across v10/v11 and is
// much easier to read than the legacy XML flavour. Blockly v11 does
// not re-export ``ToolboxDefinition`` from its public entry point
// (blockly/core), so we locally define just the shape we emit.
//
// Phase 7 note: in hardware mode the ``braccio_define_obstacle``
// block is hidden because defining a virtual obstacle has no effect
// when the backend is driving a real arm — obstacles come from real
// ToF readings in that world. ``buildToolbox(mode)`` returns a fresh
// definition so Blockly's workspace.updateToolbox() call can swap
// palettes without restarting the workspace.

import type { DeployMode } from "../state/api";

export interface BraccioToolboxBlock {
  kind: "block";
  type: string;
}

export interface BraccioToolboxCategory {
  kind: "category";
  name: string;
  colour: string;
  contents: BraccioToolboxBlock[];
}

export interface BraccioToolboxDefinition {
  kind: "categoryToolbox";
  contents: BraccioToolboxCategory[];
}

/**
 * Build a toolbox definition for the given mode.
 *
 * In hardware mode the Definitions category drops
 * ``braccio_define_obstacle`` — the block is harmless (the backend
 * no-ops it), but presenting it suggests it has an effect when it
 * doesn't, which is exactly the sort of thing that confuses kids.
 */
export function buildToolbox(mode: DeployMode = "sim"): BraccioToolboxDefinition {
  const definitionBlocks: BraccioToolboxBlock[] = [
    { kind: "block", type: "braccio_define_state" },
  ];
  if (mode === "sim") {
    definitionBlocks.push({ kind: "block", type: "braccio_define_obstacle" });
  }

  return {
    kind: "categoryToolbox",
    contents: [
      {
        kind: "category",
        name: "Movement",
        colour: "160",
        contents: [
          { kind: "block", type: "braccio_move" },
          { kind: "block", type: "braccio_set_joint" },
          { kind: "block", type: "braccio_wait" },
          { kind: "block", type: "braccio_home" },
        ],
      },
      {
        kind: "category",
        name: "Control",
        colour: "20",
        contents: [
          { kind: "block", type: "braccio_repeat" },
          { kind: "block", type: "braccio_if" },
          { kind: "block", type: "braccio_if_else" },
          { kind: "block", type: "braccio_while" },
        ],
      },
      {
        kind: "category",
        name: "Sensing",
        colour: "260",
        contents: [
          { kind: "block", type: "braccio_cond_tof" },
          { kind: "block", type: "braccio_cond_ir" },
          { kind: "block", type: "braccio_cond_joint" },
        ],
      },
      {
        kind: "category",
        name: "Definitions",
        colour: "290",
        contents: definitionBlocks,
      },
    ],
  };
}

/** Back-compat default: sim-mode toolbox with every block visible. */
export const toolbox: BraccioToolboxDefinition = buildToolbox("sim");
