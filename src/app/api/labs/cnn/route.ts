import { NextRequest, NextResponse } from 'next/server';
import { execFile } from 'child_process';
import { join } from 'path';
import { existsSync } from 'fs';

const TIMEOUT_MS = 30_000;
const PROJECT_ROOT = join(process.cwd());

const _cache = new Map<string, { data: unknown; ts: number }>();
const CACHE_TTL = 5 * 60 * 1000;

function clampInt(v: unknown, min: number, max: number, def: number): number {
  const n = typeof v === 'number' ? v : Number(v);
  if (!Number.isFinite(n)) return def;
  return Math.max(min, Math.min(max, Math.round(n)));
}

function cacheKey(p: Record<string, number>): string {
  return `cnn:${p.base_channels}:${p.depth}:${p.kernel_size}:${p.image_resolution}`;
}

function runPython(p: Record<string, number>): Promise<string> {
  return new Promise((resolve, reject) => {
    const scriptPath = join(PROJECT_ROOT, 'backend', 'services', 'lab_service.py');
    const timer = setTimeout(() => reject(new Error('timeout')), TIMEOUT_MS);
    const pythonPath = process.platform === 'win32' ? join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe') : join(PROJECT_ROOT, '.venv', 'bin', 'python');
    const pythonCmd = existsSync(pythonPath) ? pythonPath : 'python';
    execFile(pythonCmd, [
      scriptPath, '--lab', 'cnn',
      '--base_channels',    String(p.base_channels),
      '--depth',            String(p.depth),
      '--kernel_size',      String(p.kernel_size),
      '--image_resolution', String(p.image_resolution),
    ], { timeout: TIMEOUT_MS, cwd: PROJECT_ROOT }, (err, stdout, stderr) => {
      clearTimeout(timer);
      if (err && !stdout) { reject(new Error(stderr || err.message)); return; }
      resolve(stdout.trim());
    });
  });
}

export async function POST(req: NextRequest) {
  let raw: Record<string, unknown> = {};
  try { raw = await req.json(); } catch { /* use defaults */ }

  const p = {
    base_channels:    clampInt(raw.base_channels,    8,   256, 64),
    depth:            clampInt(raw.depth,            1,   8,   4),
    kernel_size:      clampInt(raw.kernel_size,      1,   7,   3),
    image_resolution: clampInt(raw.image_resolution, 32,  512, 224),
  };

  const key = cacheKey(p);
  const cached = _cache.get(key);
  if (cached && Date.now() - cached.ts < CACHE_TTL) {
    return NextResponse.json(cached.data, { headers: { 'X-Cache': 'HIT' } });
  }

  try {
    const stdout = await runPython(p);
    let data: Record<string, unknown>;
    try { data = JSON.parse(stdout); } catch {
      return NextResponse.json({ error: 'Unexpected response from model service' }, { status: 500 });
    }
    if (data.error) return NextResponse.json({ error: data.error }, { status: 422 });
    _cache.set(key, { data, ts: Date.now() });
    return NextResponse.json(data, { headers: { 'X-Cache': 'MISS' } });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (message === 'timeout') return NextResponse.json({ error: 'Timeout (30s)' }, { status: 408 });
    return NextResponse.json({ error: 'Lab service error' }, { status: 500 });
  }
}
