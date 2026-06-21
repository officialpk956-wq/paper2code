// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';
import { execFile } from 'child_process';

vi.mock('child_process', () => ({ execFile: vi.fn() }));

const MOCK_RESPONSE = {
  params_M: 1.2,
  total_flops_mflops: 120,
  memory_mb: 12,
  latency_ms: 0.05,
  receptive_field: 73,
};

function mockExecSuccess(data: Record<string, unknown>) {
  vi.mocked(execFile).mockImplementationOnce(
    (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
      (cb as (e: null, out: string, err: string) => void)(null, JSON.stringify(data), '');
      return { pid: 0 } as ReturnType<typeof execFile>;
    },
  );
}

async function postCnn(body: Record<string, unknown>) {
  const { POST } = await import('@/app/api/labs/cnn/route');
  const req = new NextRequest('http://localhost/api/labs/cnn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return POST(req);
}

describe('POST /api/labs/cnn', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns 200 with metrics on valid params', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const res = await postCnn({ base_channels: 8, depth: 1, kernel_size: 3, image_resolution: 32 });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.params_M).toBe(1.2);
    expect(data.receptive_field).toBe(73);
  });

  it('passes correct lab flag to python', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    await postCnn({ base_channels: 9, depth: 2, kernel_size: 3, image_resolution: 33 });
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--lab', 'cnn']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('clamps base_channels to minimum 8', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    await postCnn({ base_channels: 1, depth: 1, kernel_size: 3, image_resolution: 34 });
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--base_channels', '8']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('returns 500 on python service error', async () => {
    vi.mocked(execFile).mockImplementationOnce(
      (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
        (cb as (e: Error, out: string, err: string) => void)(new Error('service crash'), '', 'service crash');
        return { pid: 0 } as ReturnType<typeof execFile>;
      },
    );
    const res = await postCnn({ base_channels: 10, depth: 1, kernel_size: 3, image_resolution: 35 });
    expect(res.status).toBe(500);
  });

  it('returns 408 on timeout', async () => {
    vi.mocked(execFile).mockImplementationOnce(
      (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
        (cb as (e: Error, out: string, err: string) => void)(new Error('timeout'), '', '');
        return { pid: 0 } as ReturnType<typeof execFile>;
      },
    );
    const res = await postCnn({ base_channels: 11, depth: 1, kernel_size: 3, image_resolution: 36 });
    expect(res.status).toBe(408);
  });
});
