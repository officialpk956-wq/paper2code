// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';
import { execFile } from 'child_process';

vi.mock('child_process', () => ({ execFile: vi.fn() }));

const MOCK_RESPONSE = {
  params_M: 86.4,
  total_flops_mflops: 17500,
  memory_mb: 346,
  latency_ms: 0.26,
  num_patches: 196,
  head_dim: 64,
  num_heads: 12,
  attention_cost_mflops: 5200,
};

function mockExecSuccess(data: Record<string, unknown>) {
  vi.mocked(execFile).mockImplementationOnce(
    (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
      (cb as (e: null, out: string, err: string) => void)(null, JSON.stringify(data), '');
      return { pid: 0 } as ReturnType<typeof execFile>;
    },
  );
}

async function postVit(body: Record<string, unknown>) {
  const { POST } = await import('@/app/api/labs/vit/route');
  const req = new NextRequest('http://localhost/api/labs/vit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return POST(req);
}

describe('POST /api/labs/vit', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns 200 with metrics on valid params', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const res = await postVit({ image_size: 32, patch_size: 4, hidden_dim: 64, num_blocks: 1 });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.params_M).toBe(86.4);
    expect(data.num_patches).toBe(196);
  });

  it('passes correct lab flag to python', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    await postVit({ image_size: 33, patch_size: 4, hidden_dim: 65, num_blocks: 2 });
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--lab', 'vit']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('clamps image_size to minimum 32', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    await postVit({ image_size: 1, patch_size: 4, hidden_dim: 66, num_blocks: 1 });
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--image_size', '32']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('returns 422 when python response has error field', async () => {
    mockExecSuccess({ error: 'patch_size must divide image_size' });
    const res = await postVit({ image_size: 34, patch_size: 5, hidden_dim: 67, num_blocks: 1 });
    expect(res.status).toBe(422);
  });

  it('returns X-Cache: MISS header', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const res = await postVit({ image_size: 35, patch_size: 4, hidden_dim: 68, num_blocks: 1 });
    expect(res.headers.get('X-Cache')).toBe('MISS');
  });
});
