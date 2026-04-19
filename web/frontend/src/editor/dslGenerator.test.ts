// dslGenerator.test.ts — Headless Blockly → DSL text round-trips.
//
// Strategy: create a `new Blockly.Workspace()` (NOT WorkspaceSvg), use
// `workspace.newBlock("braccio_move")`, `initModel()` for fields,
// `setFieldValue`, and `block.getInput(...).connection.connect(...)`
// to assemble programs programmatically. We never touch the SVG
// layer, so no WebGL / canvas is required.
//
// The expected DSL strings are the contract with Phase 2's Lark parser
// in braccio_main_runner/braccio_ctrl/dsl/grammar.lark. Breaking these
// tests means the backend parser would reject the output too.

import { beforeAll, describe, expect, it } from "vitest";
import * as Blockly from "blockly/core";
import { defineBraccioBlocks, setSavedStates } from "./blocks";
import { workspaceToDsl, _internal } from "./dslGenerator";

beforeAll(() => {
  defineBraccioBlocks();
  setSavedStates(["HOME", "STOW_COMPACT", "LIFT_HIGH_CARRY"]);
});

function freshWorkspace(): Blockly.Workspace {
  return new Blockly.Workspace();
}

/**
 * Helper: create a top-level statement block with fields populated.
 * `initModel()` wires the NEW value → dropdown text lookup the
 * generator relies on, without touching the SVG layer.
 */
function makeBlock(
  workspace: Blockly.Workspace,
  type: string,
  fields: Record<string, string | number> = {},
): Blockly.Block {
  const block = workspace.newBlock(type);
  block.initModel();
  for (const [name, value] of Object.entries(fields)) {
    block.setFieldValue(value as never, name);
  }
  return block;
}

function connectNext(
  top: Blockly.Block,
  nextBlock: Blockly.Block,
): void {
  // Walk to the last block in the chain and attach `nextBlock` there.
  let tail: Blockly.Block = top;
  while (tail.nextConnection?.targetBlock()) {
    tail = tail.nextConnection.targetBlock() as Blockly.Block;
  }
  tail.nextConnection!.connect(nextBlock.previousConnection!);
}

function connectStatement(
  parent: Blockly.Block,
  inputName: string,
  childBlock: Blockly.Block,
): void {
  const input = parent.getInput(inputName);
  if (!input?.connection) throw new Error(`no statement input ${inputName}`);
  input.connection.connect(childBlock.previousConnection!);
}

function connectValue(
  parent: Blockly.Block,
  inputName: string,
  childBlock: Blockly.Block,
): void {
  const input = parent.getInput(inputName);
  if (!input?.connection) throw new Error(`no value input ${inputName}`);
  input.connection.connect(childBlock.outputConnection!);
}

describe("dslGenerator — pure helpers", () => {
  it("indent prefixes each non-empty line with two spaces", () => {
    expect(_internal.indent("a\nb\n")).toBe("  a\n  b\n");
  });

  it("indent leaves empty input alone", () => {
    expect(_internal.indent("")).toBe("");
  });

  it("num drops trailing zeros on integer values", () => {
    expect(_internal.num(42)).toBe("42");
    expect(_internal.num(42.0)).toBe("42");
  });

  it("num keeps decimals on non-integer values", () => {
    expect(_internal.num(3.25)).toBe("3.25");
  });

  it("num coerces numeric strings and returns 0 for non-finite", () => {
    expect(_internal.num("7")).toBe("7");
    expect(_internal.num(NaN)).toBe("0");
  });
});

describe("dslGenerator — single blocks", () => {
  it("braccio_move emits MOVE <NAME>", () => {
    const ws = freshWorkspace();
    makeBlock(ws, "braccio_move", {
      STATE_NAME: "HOME",
      WAIT_MS: 0,
    });
    expect(workspaceToDsl(ws)).toBe("MOVE HOME\n");
  });

  it("braccio_move adds WAIT when nonzero", () => {
    const ws = freshWorkspace();
    makeBlock(ws, "braccio_move", {
      STATE_NAME: "LIFT_HIGH_CARRY",
      WAIT_MS: 250,
    });
    expect(workspaceToDsl(ws)).toBe("MOVE LIFT_HIGH_CARRY WAIT 250\n");
  });

  it("braccio_set_joint emits SET <JOINT> <ANGLE>", () => {
    const ws = freshWorkspace();
    makeBlock(ws, "braccio_set_joint", {
      JOINT: "B",
      ANGLE: 45,
      WAIT_MS: 0,
    });
    expect(workspaceToDsl(ws)).toBe("SET B 45\n");
  });

  it("braccio_set_joint adds WAIT tail", () => {
    const ws = freshWorkspace();
    makeBlock(ws, "braccio_set_joint", {
      JOINT: "WV",
      ANGLE: 120,
      WAIT_MS: 400,
    });
    expect(workspaceToDsl(ws)).toBe("SET WV 120 WAIT 400\n");
  });

  it("braccio_wait emits WAIT n", () => {
    const ws = freshWorkspace();
    makeBlock(ws, "braccio_wait", { WAIT_MS: 1000 });
    expect(workspaceToDsl(ws)).toBe("WAIT 1000\n");
  });

  it("braccio_home emits HOME", () => {
    const ws = freshWorkspace();
    makeBlock(ws, "braccio_home");
    expect(workspaceToDsl(ws)).toBe("HOME\n");
  });
});

describe("dslGenerator — condition value blocks", () => {
  it("braccio_cond_tof emits TOF[ch] cmp value", () => {
    const ws = freshWorkspace();
    const ifBlk = makeBlock(ws, "braccio_if");
    const cond = makeBlock(ws, "braccio_cond_tof", {
      CH: "0",
      CMP: "<",
      VALUE: 200,
    });
    const body = makeBlock(ws, "braccio_home");
    connectValue(ifBlk, "COND", cond);
    connectStatement(ifBlk, "THEN", body);

    expect(workspaceToDsl(ws)).toBe(
      "IF TOF[0] < 200 {\n  HOME\n}\n",
    );
  });

  it("braccio_cond_ir emits IR cmp value", () => {
    const ws = freshWorkspace();
    const whileBlk = makeBlock(ws, "braccio_while");
    const cond = makeBlock(ws, "braccio_cond_ir", {
      CMP: ">=",
      VALUE: 2,
    });
    const body = makeBlock(ws, "braccio_wait", { WAIT_MS: 100 });
    connectValue(whileBlk, "COND", cond);
    connectStatement(whileBlk, "BODY", body);

    expect(workspaceToDsl(ws)).toBe(
      "WHILE IR >= 2 {\n  WAIT 100\n}\n",
    );
  });

  it("braccio_cond_joint emits JOINT[tok] cmp value", () => {
    const ws = freshWorkspace();
    const ifBlk = makeBlock(ws, "braccio_if");
    const cond = makeBlock(ws, "braccio_cond_joint", {
      JOINT: "S",
      CMP: "==",
      VALUE: 90,
    });
    const body = makeBlock(ws, "braccio_home");
    connectValue(ifBlk, "COND", cond);
    connectStatement(ifBlk, "THEN", body);

    expect(workspaceToDsl(ws)).toBe(
      "IF JOINT[S] == 90 {\n  HOME\n}\n",
    );
  });
});

describe("dslGenerator — control flow", () => {
  it("REPEAT wraps its body and indents", () => {
    const ws = freshWorkspace();
    const rep = makeBlock(ws, "braccio_repeat", { COUNT: 3 });
    const body1 = makeBlock(ws, "braccio_home");
    const body2 = makeBlock(ws, "braccio_wait", { WAIT_MS: 200 });
    connectStatement(rep, "BODY", body1);
    connectNext(body1, body2);

    expect(workspaceToDsl(ws)).toBe(
      "REPEAT 3 {\n  HOME\n  WAIT 200\n}\n",
    );
  });

  it("IF/ELSE emits both branches", () => {
    const ws = freshWorkspace();
    const blk = makeBlock(ws, "braccio_if_else");
    const cond = makeBlock(ws, "braccio_cond_tof", {
      CH: "1",
      CMP: "<",
      VALUE: 150,
    });
    const thenBlock = makeBlock(ws, "braccio_move", {
      STATE_NAME: "STOW_COMPACT",
      WAIT_MS: 0,
    });
    const elseBlock = makeBlock(ws, "braccio_move", {
      STATE_NAME: "HOME",
      WAIT_MS: 0,
    });
    connectValue(blk, "COND", cond);
    connectStatement(blk, "THEN", thenBlock);
    connectStatement(blk, "ELSE", elseBlock);

    expect(workspaceToDsl(ws)).toBe(
      "IF TOF[1] < 150 {\n  MOVE STOW_COMPACT\n} ELSE {\n  MOVE HOME\n}\n",
    );
  });
});

describe("dslGenerator — definitions", () => {
  it("STATE definition emits JOINTS row", () => {
    const ws = freshWorkspace();
    makeBlock(ws, "braccio_define_state", {
      NAME: "LEAN_LEFT",
      B: 30,
      S: 60,
      E: 120,
      WV: 90,
      WR: 0,
      G: 73,
    });
    expect(workspaceToDsl(ws)).toBe(
      "STATE LEAN_LEFT {\n  JOINTS 30 60 120 90 0 73\n}\n",
    );
  });

  it("OBSTACLE definition emits POS / RADIUS / SHAPE lines", () => {
    const ws = freshWorkspace();
    makeBlock(ws, "braccio_define_obstacle", {
      NAME: "BOX_A",
      X: 10,
      Y: -20,
      Z: 55,
      RADIUS: 40,
      SHAPE: "BOX",
    });
    expect(workspaceToDsl(ws)).toBe(
      "OBSTACLE BOX_A {\n  POS 10 -20 55\n  RADIUS 40\n  SHAPE BOX\n}\n",
    );
  });
});

describe("dslGenerator — full programs", () => {
  it("a full reactive program emits well-formed DSL", () => {
    const ws = freshWorkspace();

    // Build:
    //   STATE LEAN_LEFT { JOINTS ... }
    //   REPEAT 2 {
    //     MOVE HOME
    //     IF TOF[0] < 250 {
    //       MOVE STOW_COMPACT
    //     }
    //   }

    const stateDef = makeBlock(ws, "braccio_define_state", {
      NAME: "LEAN_LEFT",
      B: 45, S: 90, E: 90, WV: 90, WR: 0, G: 73,
    });

    const rep = makeBlock(ws, "braccio_repeat", { COUNT: 2 });
    const mv1 = makeBlock(ws, "braccio_move", {
      STATE_NAME: "HOME",
      WAIT_MS: 0,
    });
    const ifBlk = makeBlock(ws, "braccio_if");
    const cond = makeBlock(ws, "braccio_cond_tof", {
      CH: "0", CMP: "<", VALUE: 250,
    });
    const mv2 = makeBlock(ws, "braccio_move", {
      STATE_NAME: "STOW_COMPACT",
      WAIT_MS: 0,
    });

    connectValue(ifBlk, "COND", cond);
    connectStatement(ifBlk, "THEN", mv2);
    connectStatement(rep, "BODY", mv1);
    connectNext(mv1, ifBlk);

    // Chain state def → repeat at the top level.
    connectNext(stateDef, rep);

    expect(workspaceToDsl(ws)).toBe(
      "STATE LEAN_LEFT {\n" +
        "  JOINTS 45 90 90 90 0 73\n" +
        "}\n" +
        "REPEAT 2 {\n" +
        "  MOVE HOME\n" +
        "  IF TOF[0] < 250 {\n" +
        "    MOVE STOW_COMPACT\n" +
        "  }\n" +
        "}\n",
    );
  });

  it("empty workspace returns empty string", () => {
    const ws = freshWorkspace();
    expect(workspaceToDsl(ws)).toBe("");
  });
});
