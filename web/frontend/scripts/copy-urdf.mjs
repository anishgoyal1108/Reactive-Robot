// copy-urdf.mjs — prebuild hook.
//
// The Three.js viewer loads its kinematic description from braccio.urdf.
// During local dev the Vite proxy fetches it from the FastAPI backend;
// for static builds (demo mode on GitHub Pages) the file has to live
// under web/frontend/public/ so Vite copies it into dist/.

import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = resolve(here, "../../../braccio_main_runner/braccio_twin/urdf/braccio.urdf");
const target = resolve(here, "../public/braccio.urdf");

if (!existsSync(source)) {
  console.error(`[copy-urdf] source missing: ${source}`);
  process.exit(1);
}

mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
console.log(`[copy-urdf] ${source} -> ${target}`);
