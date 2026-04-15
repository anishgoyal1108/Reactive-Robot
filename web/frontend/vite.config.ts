import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// vite.config.ts runs in a Node context (not the browser bundle) so
// `process.env` is available at runtime — but we don't pull in
// @types/node just for this one read, so a minimal local shim keeps
// `tsc --noEmit` happy without bloating devDependencies.
declare const process: { env: Record<string, string | undefined> };

// During development the FastAPI backend runs on :8000 and Vite on
// :5173. Proxy every REST + WebSocket route so the frontend can use
// same-origin URLs (``fetch("/states")``) and never has to care about
// CORS. In production the built bundle is served by the backend, so
// this proxy only matters for ``npm run dev``.
//
// Override the default target by exporting ``VITE_BACKEND_ORIGIN`` in
// the shell or a ``.env.local`` — useful when running an alternate
// backend instance (e.g. the Preview MCP on a non-default port).
const BACKEND_ORIGIN =
  process.env.VITE_BACKEND_ORIGIN ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": BACKEND_ORIGIN,
      "/mode": BACKEND_ORIGIN,
      "/urdf": BACKEND_ORIGIN,
      "/states": BACKEND_ORIGIN,
      "/sequences": BACKEND_ORIGIN,
      "/dsl": BACKEND_ORIGIN,
      "/telemetry": BACKEND_ORIGIN,
      "/ws": {
        target: BACKEND_ORIGIN.replace(/^http/, "ws"),
        ws: true,
      },
    },
  },
});
