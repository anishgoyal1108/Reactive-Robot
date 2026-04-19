// dslGenerator.ts — Blockly → Phase 2 DSL code generator.
//
// We subclass Blockly.CodeGenerator and attach one `forBlock[name]`
// handler per custom block type from blocks.ts. The output MUST be
// parseable by braccio_main_runner/braccio_ctrl/dsl/parser.py — this
// file is the sole source of truth for that mapping.
//
// The generator is testable headlessly: create a `new Blockly.Workspace()`
// in jsdom, call `workspace.newBlock("braccio_move")` + `setFieldValue`
// + `initSvg()` is NOT needed (no SVG in tests), then call
// `dslGenerator.workspaceToCode(workspace)`. See dslGenerator.test.ts.

import * as Blockly from "blockly/core";

/**
 * Concrete generator instance — there's only one. Create it lazily so
 * unit tests that import from this module don't pay the cost when they
 * only want the helper functions.
 */
export const dslGenerator = new Blockly.CodeGenerator("BraccioDSL");

// ── Statement order constants ────────────────────────────────────────
// The DSL has no operator precedence; condition value blocks emit a
// single string like "TOF[0] < 250" that IF/WHILE wrap directly.
const ORDER_ATOMIC = 0;

/**
 * Indent every non-empty line by two spaces. Used for REPEAT/IF/WHILE
 * block bodies so the generated DSL reads legibly.
 */
function indent(text: string): string {
  if (!text) return "";
  return text
    .split("\n")
    .map((line) => (line.length > 0 ? "  " + line : line))
    .join("\n");
}

/**
 * Format a number the Lark grammar's `INT` / `SIGNED_NUMBER` accepts.
 * Integer-valued numbers lose their trailing ".0" so the DSL stays
 * readable; everything else keeps up to two decimal places.
 */
function num(value: number | string): string {
  const n = typeof value === "string" ? Number(value) : value;
  if (!Number.isFinite(n)) return "0";
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(2).replace(/\.?0+$/, "");
}

// ── Custom block handlers ────────────────────────────────────────────

dslGenerator.forBlock["braccio_move"] = (block) => {
  const name = String(block.getFieldValue("STATE_NAME") ?? "HOME");
  const wait = Number(block.getFieldValue("WAIT_MS") ?? 0);
  const tail = wait > 0 ? ` WAIT ${num(wait)}` : "";
  return `MOVE ${name}${tail}\n`;
};

dslGenerator.forBlock["braccio_set_joint"] = (block) => {
  const joint = String(block.getFieldValue("JOINT") ?? "B");
  const angle = Number(block.getFieldValue("ANGLE") ?? 90);
  const wait = Number(block.getFieldValue("WAIT_MS") ?? 0);
  const tail = wait > 0 ? ` WAIT ${num(wait)}` : "";
  return `SET ${joint} ${num(angle)}${tail}\n`;
};

dslGenerator.forBlock["braccio_wait"] = (block) => {
  const wait = Number(block.getFieldValue("WAIT_MS") ?? 0);
  return `WAIT ${num(wait)}\n`;
};

dslGenerator.forBlock["braccio_home"] = () => "HOME\n";

// Note: `statementToCode()` already prepends the generator's INDENT
// (two spaces) to every line of the nested body, so we must NOT re-
// indent here. Double-indenting was the original bug — nested blocks
// grew by 2 spaces per level instead of staying aligned.

dslGenerator.forBlock["braccio_repeat"] = (block, gen) => {
  const count = Number(block.getFieldValue("COUNT") ?? 1);
  const body = gen.statementToCode(block, "BODY");
  return `REPEAT ${num(count)} {\n${body.trimEnd()}\n}\n`;
};

dslGenerator.forBlock["braccio_if"] = (block, gen) => {
  const cond = gen.valueToCode(block, "COND", ORDER_ATOMIC) || "IR >= 1";
  const then = gen.statementToCode(block, "THEN");
  return `IF ${cond} {\n${then.trimEnd()}\n}\n`;
};

dslGenerator.forBlock["braccio_if_else"] = (block, gen) => {
  const cond = gen.valueToCode(block, "COND", ORDER_ATOMIC) || "IR >= 1";
  const then = gen.statementToCode(block, "THEN");
  const otherwise = gen.statementToCode(block, "ELSE");
  return (
    `IF ${cond} {\n${then.trimEnd()}\n}` +
    ` ELSE {\n${otherwise.trimEnd()}\n}\n`
  );
};

dslGenerator.forBlock["braccio_while"] = (block, gen) => {
  const cond = gen.valueToCode(block, "COND", ORDER_ATOMIC) || "IR >= 1";
  const body = gen.statementToCode(block, "BODY");
  return `WHILE ${cond} {\n${body.trimEnd()}\n}\n`;
};

// Condition value blocks — return [code, order] tuples.
dslGenerator.forBlock["braccio_cond_tof"] = (block): [string, number] => {
  const ch = Number(block.getFieldValue("CH") ?? 0);
  const cmp = String(block.getFieldValue("CMP") ?? "<");
  const val = Number(block.getFieldValue("VALUE") ?? 0);
  return [`TOF[${num(ch)}] ${cmp} ${num(val)}`, ORDER_ATOMIC];
};

dslGenerator.forBlock["braccio_cond_ir"] = (block): [string, number] => {
  const cmp = String(block.getFieldValue("CMP") ?? ">=");
  const val = Number(block.getFieldValue("VALUE") ?? 0);
  return [`IR ${cmp} ${num(val)}`, ORDER_ATOMIC];
};

dslGenerator.forBlock["braccio_cond_joint"] = (block): [string, number] => {
  const joint = String(block.getFieldValue("JOINT") ?? "B");
  const cmp = String(block.getFieldValue("CMP") ?? "==");
  const val = Number(block.getFieldValue("VALUE") ?? 90);
  return [`JOINT[${joint}] ${cmp} ${num(val)}`, ORDER_ATOMIC];
};

dslGenerator.forBlock["braccio_define_state"] = (block) => {
  const name = String(block.getFieldValue("NAME") ?? "MY_STATE").trim();
  const b = Number(block.getFieldValue("B") ?? 90);
  const s = Number(block.getFieldValue("S") ?? 90);
  const e = Number(block.getFieldValue("E") ?? 90);
  const wv = Number(block.getFieldValue("WV") ?? 90);
  const wr = Number(block.getFieldValue("WR") ?? 90);
  const g = Number(block.getFieldValue("G") ?? 73);
  return (
    `STATE ${name} {\n` +
    `  JOINTS ${num(b)} ${num(s)} ${num(e)} ${num(wv)} ${num(wr)} ${num(g)}\n` +
    `}\n`
  );
};

dslGenerator.forBlock["braccio_define_obstacle"] = (block) => {
  const name = String(block.getFieldValue("NAME") ?? "OBS").trim();
  const x = Number(block.getFieldValue("X") ?? 0);
  const y = Number(block.getFieldValue("Y") ?? 0);
  const z = Number(block.getFieldValue("Z") ?? 0);
  const radius = Number(block.getFieldValue("RADIUS") ?? 40);
  const shape = String(block.getFieldValue("SHAPE") ?? "BOX");
  return (
    `OBSTACLE ${name} {\n` +
    `  POS ${num(x)} ${num(y)} ${num(z)}\n` +
    `  RADIUS ${num(radius)}\n` +
    `  SHAPE ${shape}\n` +
    `}\n`
  );
};

// ── Scrub: glue statements together without blank separators ─────────
//
// The base implementation inserts nothing between blocks, which is
// exactly what we want — each handler already emits a trailing newline.
// We override `scrub_` only to chain `nextConnection` blocks.
(dslGenerator as unknown as {
  scrub_: (block: Blockly.Block, code: string, thisOnly?: boolean) => string;
}).scrub_ = function (block, code, thisOnly) {
  const nextBlock =
    block.nextConnection && block.nextConnection.targetBlock();
  if (nextBlock && !thisOnly) {
    return code + dslGenerator.blockToCode(nextBlock);
  }
  return code;
};

/**
 * Compile a workspace's top-level blocks into DSL text.
 *
 * Returns the generated code with leading/trailing whitespace trimmed
 * and a single trailing newline added — the parser accepts both but
 * this keeps snapshot tests stable.
 */
export function workspaceToDsl(workspace: Blockly.Workspace): string {
  const raw = dslGenerator.workspaceToCode(workspace);
  const trimmed = raw.replace(/\s+$/u, "");
  return trimmed ? trimmed + "\n" : "";
}

/** Test-only re-export of the helpers above. */
export const _internal = { indent, num };
