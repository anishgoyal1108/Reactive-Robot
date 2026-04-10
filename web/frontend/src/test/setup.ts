// setup.ts — Vitest environment glue.
//
// Runs once before every test file. jsdom ships most DOM APIs but
// not ResizeObserver or matchMedia, which r3f sometimes touches on
// mount. We install shims here so importing viewer modules never
// crashes the test bootstrap even though we never render them to a
// real canvas.

import "@testing-library/jest-dom/vitest";

class MockResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

if (typeof globalThis.ResizeObserver === "undefined") {
  (globalThis as unknown as { ResizeObserver: typeof ResizeObserver }).ResizeObserver =
    MockResizeObserver as unknown as typeof ResizeObserver;
}

if (
  typeof globalThis.matchMedia === "undefined" &&
  typeof globalThis.window !== "undefined"
) {
  Object.defineProperty(globalThis.window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}
