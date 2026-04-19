// toolbox.test.ts — buildToolbox() is a pure function, so the tests
// just assert category membership for each deploy mode.

import { describe, expect, it } from "vitest";
import { buildToolbox } from "./toolbox";

function blockTypesIn(
  tb: ReturnType<typeof buildToolbox>,
  categoryName: string,
): string[] {
  const cat = tb.contents.find((c) => c.name === categoryName);
  if (!cat) return [];
  return cat.contents.map((b) => b.type);
}

describe("buildToolbox", () => {
  it("sim-mode includes both definition blocks", () => {
    const tb = buildToolbox("sim");
    const defs = blockTypesIn(tb, "Definitions");
    expect(defs).toEqual(
      expect.arrayContaining([
        "braccio_define_state",
        "braccio_define_obstacle",
      ]),
    );
  });

  it("hardware-mode drops braccio_define_obstacle", () => {
    const tb = buildToolbox("hardware");
    const defs = blockTypesIn(tb, "Definitions");
    expect(defs).toContain("braccio_define_state");
    expect(defs).not.toContain("braccio_define_obstacle");
  });

  it("hardware-mode keeps the other four categories intact", () => {
    const tb = buildToolbox("hardware");
    const categoryNames = tb.contents.map((c) => c.name);
    expect(categoryNames).toEqual([
      "Movement",
      "Control",
      "Sensing",
      "Definitions",
    ]);

    // Exhaustive check: none of the Movement/Control/Sensing blocks
    // depend on the mode, so they should match sim verbatim.
    const sim = buildToolbox("sim");
    for (const name of ["Movement", "Control", "Sensing"]) {
      expect(blockTypesIn(tb, name)).toEqual(blockTypesIn(sim, name));
    }
  });

  it("default (no arg) is sim-mode", () => {
    const tb = buildToolbox();
    const defs = blockTypesIn(tb, "Definitions");
    expect(defs).toContain("braccio_define_obstacle");
  });
});
