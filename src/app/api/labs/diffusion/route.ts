import { NextRequest, NextResponse } from 'next/server';
import { execFile } from 'child_process';
import { join } from 'path';

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
  return `diffusion:${p.latent_size}:${p.channels}:${p.diffusion_steps}`;
}

function runPython(p: Record<string, number>): Promise<string> {
  return new Promise((resolve, reject) => {
    const scriptPath = join(PROJECT_ROOT, 'backend', 'services', 'lab_service.py');
    const timer = setTimeout(() => reject(new Error('timeout')), TIMEOUT_MS);
    execFile('python', [
      scriptPath, '--lab', 'diffusion',
      '--latent_size',    String(p.latent_size),
      '--channels',       String(p.channels),
      '--diffusion_steps', String(p.diffusion_steps),
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
    latent_size:    clampInt(raw.latent_size,    8,   128,  32),
    channels:       clampInt(raw.channels,       1,   4,    3),
    diffusion_steps: clampInt(raw.diffusion_steps, 100, 2000, 1000),
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
