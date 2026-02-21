#!/usr/bin/env node
/**
 * build-dashboard.mjs — Node-only static dashboard build for Cloudflare Workers Builds.
 *
 * Mirrors DashboardBuilder._copy_static() + _compile_typescript() from
 * src/infrastructure/reporting/dashboard/builder.py without requiring Python.
 *
 * Produces out/site/ with:
 *   index.html          (SPA shell)
 *   _headers            (CF cache headers)
 *   .nojekyll
 *   assets/             (JS + CSS)
 *   data/manifest.json  (empty placeholder — real data comes from GH Actions)
 */

import { execSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, relative } from "node:path";

const STATIC_DIR = "src/infrastructure/reporting/dashboard/static";
const OUTPUT_DIR = "out/site";
const ASSETS_DIR = join(OUTPUT_DIR, "assets");

// CF cache headers — must match builder.py CF_HEADERS
const CF_HEADERS = `\
/data/*
  Cache-Control: public, max-age=300
  X-Content-Type-Options: nosniff

/assets/*
  Cache-Control: public, max-age=86400, immutable
  X-Content-Type-Options: nosniff

/index.html
  Cache-Control: public, max-age=300
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
`;

/** Recursively collect all files under a directory. */
function walkDir(dir) {
  const results = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...walkDir(full));
    } else if (entry.isFile()) {
      results.push(full);
    }
  }
  return results;
}

/** Recursively collect files matching an extension under a directory. */
function findByExt(dir, ext) {
  if (!existsSync(dir)) return [];
  return walkDir(dir).filter((f) => f.endsWith(ext));
}

// ── Step 1: Clean + create output ────────────────────────────────────────────

if (existsSync(OUTPUT_DIR)) {
  rmSync(OUTPUT_DIR, { recursive: true });
}
mkdirSync(OUTPUT_DIR, { recursive: true });
mkdirSync(ASSETS_DIR, { recursive: true });

console.log(`[build] Static source: ${STATIC_DIR}`);

if (!existsSync(STATIC_DIR)) {
  console.error(`[build] ERROR: Static assets not found at ${STATIC_DIR}`);
  process.exit(1);
}

// ── Step 2: Copy static files (mirrors _copy_static) ────────────────────────
// index.html → root, everything else → assets/ preserving directory structure

for (const srcFile of walkDir(STATIC_DIR)) {
  const rel = relative(STATIC_DIR, srcFile);

  let dst;
  if (basename(rel) === "index.html") {
    dst = join(OUTPUT_DIR, "index.html");
  } else {
    dst = join(ASSETS_DIR, rel);
  }

  mkdirSync(dirname(dst), { recursive: true });
  cpSync(srcFile, dst);
}

console.log("[build] Static files copied");

// ── Step 3: Compile TypeScript (mirrors _compile_typescript) ─────────────────
// esbuild: per-file transpile, --format=esm --target=es2022

const tsFiles = findByExt(ASSETS_DIR, ".ts");

if (tsFiles.length > 0) {
  let esbuildOk = false;
  try {
    const cmd = [
      "npx",
      "--yes",
      "esbuild@0.24",
      ...tsFiles,
      `--outdir=${ASSETS_DIR}`,
      "--format=esm",
      "--target=es2022",
    ].join(" ");
    execSync(cmd, { stdio: "pipe" });
    esbuildOk = true;
    console.log(`[build] esbuild compiled ${tsFiles.length} TS files`);
  } catch (err) {
    console.error("[build] ERROR: esbuild compilation failed");
    console.error(err.stderr?.toString().slice(0, 500) || err.message);
    process.exit(1);
  }

  // Remove .ts sources from output (keep only compiled .js)
  for (const f of findByExt(ASSETS_DIR, ".ts")) {
    unlinkSync(f);
  }

  // Remove dev-only files: types/ directory and tsconfig.json
  const typesDir = join(ASSETS_DIR, "types");
  if (existsSync(typesDir)) {
    rmSync(typesDir, { recursive: true });
  }
  const tsconfig = join(ASSETS_DIR, "tsconfig.json");
  if (existsSync(tsconfig)) {
    unlinkSync(tsconfig);
  }

  console.log("[build] TS sources and dev files cleaned");
}

// ── Step 4: Write _headers, .nojekyll ────────────────────────────────────────

writeFileSync(join(OUTPUT_DIR, "_headers"), CF_HEADERS);
writeFileSync(join(OUTPUT_DIR, ".nojekyll"), "");

// ── Step 5: Create data/ with placeholder manifest ──────────────────────────

const dataDir = join(OUTPUT_DIR, "data");
mkdirSync(dataDir, { recursive: true });
writeFileSync(
  join(dataDir, "manifest.json"),
  JSON.stringify({ version: "1.0", symbols: [], timeframes: ["1d"] }, null, 2) + "\n"
);

console.log("[build] Placeholder data/manifest.json written");

// ── Done ─────────────────────────────────────────────────────────────────────

const fileCount = walkDir(OUTPUT_DIR).length;
console.log(`[build] Done — ${fileCount} files in ${OUTPUT_DIR}/`);
