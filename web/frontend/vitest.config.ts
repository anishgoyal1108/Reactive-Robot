import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Vitest config is separate from vite.config.ts so the production
// build never pulls in jsdom / testing-library. The test environment
// is jsdom so React components can mount headlessly; pure-TypeScript
// state modules pass just as well under a node environment.
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
