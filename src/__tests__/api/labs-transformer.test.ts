// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';
import { execFile } from 'child_process';

vi.mock('child_process', () => ({ execFile: vi.fn() }));

const MOCK_RESPONSE = {
  params_M: 85.3,
  total_flops_mflops: 5000,
  memory_mb: 341.2,
  latency_ms: 0.167,
  token_count: 128,
  head_dim: 64,
  num_heads: 8,
  attention_cost_mflops: 1000,
};

function mockExecSuccess(data: Record<string, unknown>) {
  vi.mocked(execFile).mockImplementationOnce(
    (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
      (cb as (e: null, out: string, err: string) => void)(null, JSON.stringify(data), '');
      return { pid: 0 } as ReturnType<typeof execFile>;
    },
  );
}

function mockExecError(message: string) {
  vi.mocked(execFile).mockImplementationOnce(
    (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
      (cb as (e: Error, out: string, err: string) => void)(new Error(message), '', message);
      return { pid: 0 } as ReturnType<typeof execFile>;
    },
  );
}

async function postTransformer(body: Record<string, unknown>, params = {}) {
  const { POST } = await import('@/app/api/labs/transformer/route');
  const req = new NextRequest('http://localhost/api/labs/transformer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, ...params }),
  });
  return POST(req);
}

describe('POST /api/labs/transformer', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('returns 200 with model metrics on valid params', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const res = await postTransformer({ d_model: 64, num_heads: 1, num_layers: 1, seq_len: 16, vocab_size: 1000 });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.params_M).toBe(85.3);
    expect(data.total_flops_mflops).toBe(5000);
  });

  it('returns X-Cache: MISS on first request', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const res = await postTransformer({ d_model: 65, num_heads: 1, num_layers: 1, seq_len: 17, vocab_size: 1001 });
    expect(res.headers.get('X-Cache')).toBe('MISS');
  });

  it('clamps d_model to valid range', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const res = await postTransformer({ d_model: 99999, num_heads: 2, num_layers: 2, seq_len: 32, vocab_size: 2000 });
    expect(res.status).toBe(200);
    // clamped to 1024 max
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--d_model', '1024']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('clamps d_model minimum to 64', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    // Use unique seq_len/vocab_size to avoid cache collision with other tests that also clamp to d_model=64
    await postTransformer({ d_model: 1, num_heads: 1, num_layers: 1, seq_len: 48, vocab_size: 3000 });
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--d_model', '64']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('uses default values when body is empty JSON', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const { POST } = await import('@/app/api/labs/transformer/route');
    const req = new NextRequest('http://localhost/api/labs/transformer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
    });
    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--d_model', '512']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('handles invalid JSON body gracefully with defaults', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const { POST } = await import('@/app/api/labs/transformer/route');
    const req = new NextRequest('http://localhost/api/labs/transformer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: 'not-json',
    });
    const res = await POST(req);
    expect(res.status).toBe(200);
  });

  it('returns 408 on timeout', async () => {
    vi.mocked(execFile).mockImplementationOnce(
      (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
        (cb as (e: Error, out: string, err: string) => void)(new Error('timeout'), '', '');
        return { pid: 0 } as ReturnType<typeof execFile>;
      },
    );
    const res = await postTransformer({ d_model: 66, num_heads: 2, num_layers: 2, seq_len: 18, vocab_size: 1002 });
    expect(res.status).toBe(408);
  });

  it('returns 422 when python returns error field', async () => {
    mockExecSuccess({ error: 'GPU not available' });
    const res = await postTransformer({ d_model: 67, num_heads: 1, num_layers: 1, seq_len: 19, vocab_size: 1003 });
    expect(res.status).toBe(422);
    const data = await res.json();
    expect(data.error).toBe('GPU not available');
  });

  it('returns 500 when python outputs non-JSON', async () => {
    vi.mocked(execFile).mockImplementationOnce(
      (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
        (cb as (e: null, out: string, err: string) => void)(null, 'Traceback (most recent call last):\n  ...', '');
        return { pid: 0 } as ReturnType<typeof execFile>;
      },
    );
    const res = await postTransformer({ d_model: 68, num_heads: 1, num_layers: 1, seq_len: 20, vocab_size: 1004 });
    expect(res.status).toBe(500);
  });
});
