import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the FastAPI backend runs on :8000 and Vite on
// :5173. Proxy every REST + WebSocket route so the frontend can use
// same-origin URLs (``fetch("/states")``) and never has to care about
// CORS. In production the built bundle is served by the backend, so
// this proxy only matters for ``npm run dev``.
//
// Override the default target by creating a ``.env.local`` with
// ``VITE_BACKEND_ORIGIN=http://...`` — Vite surfaces those as
// ``import.meta.env.*`` at config-parse time.
const BACKEND_ORIGIN = "http://localhost:8000";

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
