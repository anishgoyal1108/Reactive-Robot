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

export const toolbox: BraccioToolboxDefinition = {
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
      contents: [
        { kind: "block", type: "braccio_define_state" },
        { kind: "block", type: "braccio_define_obstacle" },
      ],
    },
  ],
};
