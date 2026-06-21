import { NextRequest, NextResponse } from 'next/server';
import { execFile } from 'child_process';
import { join } from 'path';

const TIMEOUT_MS = 30_000;
const PROJECT_ROOT = join(process.cwd());

// Allowlist of valid architecture IDs — prevents command injection via URL params
const VALID_ARCH_IDS = new Set([
  'resnet50', 'resnet', 'deep-residual-learning',
  'vit', 'vit-b16', 'an-image-is-worth-16x16-words',
  'unet', 'unet-biomedical', 'ronneberger2015',
  'transformer', 'attention-is-all-you-need', 'vaswani2017',
]);

// Simple in-process cache — architectures don't change between requests
const _cache = new Map<string, { data: unknown; ts: number }>();
const CACHE_TTL = 10 * 60 * 1000; // 10 minutes

function runPython(architecture: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const scriptPath = join(PROJECT_ROOT, 'backend', 'services', 'block_viz_service.py');
    const timer = setTimeout(() => reject(new Error('timeout')), TIMEOUT_MS);
    // execFile avoids shell interpolation — args passed as separate argv entries
    execFile('python', [scriptPath, '--architecture', architecture, '--action', 'hierarchy'], {
      timeout: TIMEOUT_MS, cwd: PROJECT_ROOT,
    }, (err, stdout, stderr) => {
      clearTimeout(timer);
      if (err && !stdout) { reject(new Error(stderr || err.message)); return; }
      resolve(stdout.trim());
    });
  });
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const archId = id.toLowerCase();

  if (!VALID_ARCH_IDS.has(archId)) {
    return NextResponse.json({ error: `Unknown architecture: ${archId}` }, { status: 404 });
  }

  // Serve from cache if fresh
  const cached = _cache.get(archId);
  if (cached && Date.now() - cached.ts < CACHE_TTL) {
    return NextResponse.json(cached.data, {
      headers: { 'X-Cache': 'HIT', 'Cache-Control': 'public, max-age=600' },
    });
  }

  try {
    const raw = await runPython(archId);
    const data = JSON.parse(raw);

    if (data.error) {
      return NextResponse.json({ error: data.error }, { status: 422 });
    }

    _cache.set(archId, { data, ts: Date.now() });
    return NextResponse.json(data, {
      headers: { 'X-Cache': 'MISS', 'Cache-Control': 'public, max-age=600' },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (message === 'timeout') {
      return NextResponse.json({ error: 'Service timeout (30s)' }, { status: 408 });
    }
    return NextResponse.json({ error: `Block viz error: ${message}` }, { status: 500 });
  }
}
