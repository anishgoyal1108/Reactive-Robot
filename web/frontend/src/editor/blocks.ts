// blocks.ts — Braccio custom Blockly block definitions.
//
// These blocks map 1:1 onto the Phase 2 DSL grammar in
// braccio_main_runner/braccio_ctrl/dsl/grammar.lark. The code generator
// in dslGenerator.ts walks the workspace and emits DSL text that the
// FastAPI backend ingests via POST /dsl/run.
//
// Block naming convention: all types are prefixed "braccio_" so they
// cannot collide with Blockly's built-in blocks if the toolbox ever
// grows to include math/variables for on-the-fly calculations.
//
// The call `defineBraccioBlocks()` is idempotent — defining the same
// block name twice with Blockly.common.defineBlocks overwrites the
// first, so it is safe to call from BlocklyEditor.tsx on every mount
// and from unit tests at module load time.

import * as Blockly from "blockly/core";

/** Joint tokens accepted by the DSL — canonical order (B/S/E/WV/WR/G). */
export const JOINT_CHOICES: ReadonlyArray<[string, string]> = [
  ["base", "B"],
  ["shoulder", "S"],
  ["elbow", "E"],
  ["wrist vert", "WV"],
  ["wrist rot", "WR"],
  ["gripper", "G"],
];

/** Comparison operators — must match the CMP terminal in grammar.lark. */
export const CMP_CHOICES: ReadonlyArray<[string, string]> = [
  ["<", "<"],
  ["<=", "<="],
  ["=", "=="],
  ["!=", "!="],
  [">=", ">="],
  [">", ">"],
];

/** ToF channel choices — CH3 is the floor sensor, left out on purpose. */
export const TOF_CHANNEL_CHOICES: ReadonlyArray<[string, string]> = [
  ["CH0 (front)", "0"],
  ["CH1 (back)", "1"],
  ["CH2 (top)", "2"],
  ["CH3 (floor)", "3"],
];

/** Obstacle shape kinds — must match SHAPE_KIND terminal. */
export const SHAPE_CHOICES: ReadonlyArray<[string, string]> = [
  ["box", "BOX"],
  ["sphere", "SPHERE"],
  ["cylinder", "CYLINDER"],
];

/**
 * Saved state names exposed via the "move to state" dropdown. The
 * BlocklyEditor React wrapper refreshes this from ApiClient.listStates()
 * on mount; tests can override it directly. An empty list falls back
 * to ["HOME"] so the block always has at least one legal value.
 */
let _savedStates: string[] = ["HOME"];

export function setSavedStates(names: readonly string[]): void {
  _savedStates = names.length > 0 ? [...names] : ["HOME"];
}

export function getSavedStates(): readonly string[] {
  return _savedStates;
}

/**
 * Register all custom Braccio blocks with Blockly. Safe to call many
 * times — each call overwrites the previous definitions.
 */
export function defineBraccioBlocks(): void {
  Blockly.defineBlocksWithJsonArray([
    // ── Movement ────────────────────────────────────────────────
    {
      type: "braccio_move",
      message0: "move to state %1 wait %2 ms",
      args0: [
        {
          type: "field_dropdown",
          name: "STATE_NAME",
          options: () =>
            _savedStates.map((n) => [n, n] as [string, string]),
        },
        {
          type: "field_number",
          name: "WAIT_MS",
          value: 500,
          min: 0,
          precision: 1,
        },
      ],
      previousStatement: null,
      nextStatement: null,
      colour: 160,
      tooltip:
        "MOVE to a saved state from the library, then pause for N ms.",
    },
    {
      type: "braccio_set_joint",
      message0: "set joint %1 to %2 ° wait %3 ms",
      args0: [
        {
          type: "field_dropdown",
          name: "JOINT",
          options: JOINT_CHOICES.map(([label, val]) => [label, val]),
        },
        {
          type: "field_number",
          name: "ANGLE",
          value: 90,
          min: 0,
          max: 180,
          precision: 1,
        },
        {
          type: "field_number",
          name: "WAIT_MS",
          value: 200,
          min: 0,
          precision: 1,
        },
      ],
      previousStatement: null,
      nextStatement: null,
      colour: 160,
      tooltip:
        "SET JOINT a single servo to a specific angle, then pause.",
    },
    {
      type: "braccio_wait",
      message0: "wait %1 ms",
      args0: [
        {
          type: "field_number",
          name: "WAIT_MS",
          value: 500,
          min: 0,
          precision: 1,
        },
      ],
      previousStatement: null,
      nextStatement: null,
      colour: 160,
      tooltip: "WAIT for N milliseconds without moving.",
    },
    {
      type: "braccio_home",
      message0: "go home",
      previousStatement: null,
      nextStatement: null,
      colour: 160,
      tooltip: "HOME — return the arm to its home pose.",
    },

    // ── Control flow ────────────────────────────────────────────
    {
      type: "braccio_repeat",
      message0: "repeat %1 times",
      args0: [
        {
          type: "field_number",
          name: "COUNT",
          value: 3,
          min: 1,
          precision: 1,
        },
      ],
      message1: "do %1",
      args1: [{ type: "input_statement", name: "BODY" }],
      previousStatement: null,
      nextStatement: null,
      colour: 20,
      tooltip: "REPEAT the inner blocks N times.",
    },
    {
      type: "braccio_if",
      message0: "if %1",
      args0: [
        {
          type: "input_value",
          name: "COND",
          check: "Condition",
        },
      ],
      message1: "then %1",
      args1: [{ type: "input_statement", name: "THEN" }],
      previousStatement: null,
      nextStatement: null,
      colour: 20,
      tooltip:
        "IF the sensor condition is true, run the inner blocks once.",
    },
    {
      type: "braccio_if_else",
      message0: "if %1",
      args0: [
        {
          type: "input_value",
          name: "COND",
          check: "Condition",
        },
      ],
      message1: "then %1",
      args1: [{ type: "input_statement", name: "THEN" }],
      message2: "else %1",
      args2: [{ type: "input_statement", name: "ELSE" }],
      previousStatement: null,
      nextStatement: null,
      colour: 20,
      tooltip:
        "IF/ELSE — pick a branch based on a sensor condition.",
    },
    {
      type: "braccio_while",
      message0: "while %1",
      args0: [
        {
          type: "input_value",
          name: "COND",
          check: "Condition",
        },
      ],
      message1: "do %1",
      args1: [{ type: "input_statement", name: "BODY" }],
      previousStatement: null,
      nextStatement: null,
      colour: 20,
      tooltip: "WHILE the sensor condition is true, keep running the body.",
    },

    // ── Condition value blocks (produce the `Condition` type) ──
    {
      type: "braccio_cond_tof",
      message0: "ToF %1 %2 %3 mm",
      args0: [
        {
          type: "field_dropdown",
          name: "CH",
          options: TOF_CHANNEL_CHOICES.map(([l, v]) => [l, v]),
        },
        {
          type: "field_dropdown",
          name: "CMP",
          options: CMP_CHOICES.map(([l, v]) => [l, v]),
        },
        {
          type: "field_number",
          name: "VALUE",
          value: 250,
          min: 0,
          precision: 1,
        },
      ],
      output: "Condition",
      colour: 260,
      tooltip: "Compare a ToF channel reading to a distance in mm.",
    },
    {
      type: "braccio_cond_ir",
      message0: "IR %1 %2",
      args0: [
        {
          type: "field_dropdown",
          name: "CMP",
          options: CMP_CHOICES.map(([l, v]) => [l, v]),
        },
        {
          type: "field_number",
          name: "VALUE",
          value: 1,
          min: 0,
          max: 3,
          precision: 1,
        },
      ],
      output: "Condition",
      colour: 260,
      tooltip: "Compare the IR severity (0..3) to a number.",
    },
    {
      type: "braccio_cond_joint",
      message0: "joint %1 %2 %3 °",
      args0: [
        {
          type: "field_dropdown",
          name: "JOINT",
          options: JOINT_CHOICES.map(([l, v]) => [l, v]),
        },
        {
          type: "field_dropdown",
          name: "CMP",
          options: CMP_CHOICES.map(([l, v]) => [l, v]),
        },
        {
          type: "field_number",
          name: "VALUE",
          value: 90,
          min: 0,
          max: 180,
          precision: 1,
        },
      ],
      output: "Condition",
      colour: 260,
      tooltip:
        "Compare a joint's current angle to a target (°).",
    },

    // ── Definition blocks (top-level declarations) ──────────────
    {
      type: "braccio_define_state",
      message0: "define state %1",
      args0: [
        {
          type: "field_input",
          name: "NAME",
          text: "MY_STATE",
        },
      ],
      message1: "B %1 S %2 E %3 WV %4 WR %5 G %6",
      args1: [
        { type: "field_number", name: "B", value: 90, min: 0, max: 180 },
        { type: "field_number", name: "S", value: 90, min: 0, max: 180 },
        { type: "field_number", name: "E", value: 90, min: 0, max: 180 },
        { type: "field_number", name: "WV", value: 90, min: 0, max: 180 },
        { type: "field_number", name: "WR", value: 90, min: 0, max: 180 },
        { type: "field_number", name: "G", value: 73, min: 0, max: 180 },
      ],
      previousStatement: null,
      nextStatement: null,
      colour: 290,
      tooltip:
        "Declare a named pose the MOVE block can target.",
    },
    {
      type: "braccio_define_obstacle",
      message0: "define obstacle %1",
      args0: [
        { type: "field_input", name: "NAME", text: "BLOCK_A" },
      ],
      message1: "position x %1 y %2 z %3 (mm)",
      args1: [
        {
          type: "field_number",
          name: "X",
          value: 0,
          precision: 0.1,
        },
        {
          type: "field_number",
          name: "Y",
          value: 0,
          precision: 0.1,
        },
        {
          type: "field_number",
          name: "Z",
          value: 0,
          precision: 0.1,
        },
      ],
      message2: "radius %1 mm shape %2",
      args2: [
        {
          type: "field_number",
          name: "RADIUS",
          value: 40,
          min: 1,
          precision: 0.1,
        },
        {
          type: "field_dropdown",
          name: "SHAPE",
          options: SHAPE_CHOICES.map(([l, v]) => [l, v]),
        },
      ],
      previousStatement: null,
      nextStatement: null,
      colour: 290,
      tooltip:
        "Declare a virtual obstacle for the sim backend to raycast against.",
    },
  ]);
}

/**
 * Internal escape hatch for tests that want to observe what Blockly
 * thinks the registered block types are. Not used by runtime code.
 */
export function _getRegisteredBlockTypes(): string[] {
  return Object.keys(Blockly.Blocks);
}
