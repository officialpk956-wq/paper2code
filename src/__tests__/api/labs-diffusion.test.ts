// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';
import { execFile } from 'child_process';

vi.mock('child_process', () => ({ execFile: vi.fn() }));

const MOCK_RESPONSE = {
  params_M: 35.7,
  total_flops_mflops: 8500,
  memory_mb: 143,
  latency_ms: 0.4,
  step_flops_mflops: 85,
  inference_flops_mflops: 8500,
};

function mockExecSuccess(data: Record<string, unknown>) {
  vi.mocked(execFile).mockImplementationOnce(
    (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
      (cb as (e: null, out: string, err: string) => void)(null, JSON.stringify(data), '');
      return { pid: 0 } as ReturnType<typeof execFile>;
    },
  );
}

async function postDiffusion(body: Record<string, unknown>) {
  const { POST } = await import('@/app/api/labs/diffusion/route');
  const req = new NextRequest('http://localhost/api/labs/diffusion', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return POST(req);
}

describe('POST /api/labs/diffusion', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns 200 with metrics on valid params', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    const res = await postDiffusion({ latent_size: 8, channels: 1, diffusion_steps: 100 });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.step_flops_mflops).toBe(85);
    expect(data.inference_flops_mflops).toBe(8500);
  });

  it('passes correct lab flag to python', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    await postDiffusion({ latent_size: 9, channels: 1, diffusion_steps: 101 });
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--lab', 'diffusion']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('clamps diffusion_steps to minimum 100', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    await postDiffusion({ latent_size: 10, channels: 1, diffusion_steps: 1 });
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--diffusion_steps', '100']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('clamps diffusion_steps to maximum 2000', async () => {
    mockExecSuccess(MOCK_RESPONSE);
    await postDiffusion({ latent_size: 11, channels: 1, diffusion_steps: 99999 });
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--diffusion_steps', '2000']),
      expect.any(Object),
      expect.any(Function),
    );
  });

  it('returns 500 on exec error', async () => {
    vi.mocked(execFile).mockImplementationOnce(
      (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
        (cb as (e: Error, out: string, err: string) => void)(new Error('crash'), '', 'crash');
        return { pid: 0 } as ReturnType<typeof execFile>;
      },
    );
    const res = await postDiffusion({ latent_size: 12, channels: 1, diffusion_steps: 200 });
    expect(res.status).toBe(500);
  });
});
