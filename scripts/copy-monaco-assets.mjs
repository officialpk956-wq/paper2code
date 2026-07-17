#!/usr/bin/env node
/**
 * Copies monaco-editor's static AMD bundle into public/ so the Dojo code
 * editor loads from our own origin instead of the jsdelivr CDN (the CDN
 * default breaks silently on any network/CSP restriction — editor.main.css
 * fails to load, which un-hides Monaco's raw <textarea> chrome and drops
 * cursor styling). Runs via `postinstall`; safe to re-run any time.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SRC = path.join(ROOT, "node_modules", "monaco-editor", "min", "vs");
const DEST = path.join(ROOT, "public", "monaco-editor", "vs");

if (!fs.existsSync(SRC)) {
  console.warn(`[copy-monaco-assets] skip — ${SRC} not found (monaco-editor not installed yet)`);
  process.exit(0);
}

fs.rmSync(DEST, { recursive: true, force: true });
fs.cpSync(SRC, DEST, { recursive: true });
console.log(`[copy-monaco-assets] copied monaco-editor assets to ${path.relative(ROOT, DEST)}`);
