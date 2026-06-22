// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';
import { execFile } from 'child_process';

vi.mock('child_process', () => ({ execFile: vi.fn() }));

const MOCK_FORWARD_PASS = {
  architecture: 'resnet50',
  steps: [
    {
      step: 0, node_id: 'conv1', node_name: 'Conv1', type: 'Conv2d',
      input_shape: [1, 3, 224, 224], output_shape: [1, 64, 112, 112],
      flops_mflops: 236, params_M: 0.09, severity: 'low', description: 'Initial convolution',
    },
  ],
};

function mockExecSuccess(data: Record<string, unknown>) {
  vi.mocked(execFile).mockImplementationOnce(
    (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
      (cb as (e: null, out: string, err: string) => void)(null, JSON.stringify(data), '');
      return { pid: 0 } as ReturnType<typeof execFile>;
    },
  );
}

async function getForwardPass(id: string) {
  const { GET } = await import('@/app/api/papers/[id]/forward-pass/route');
  const req = new NextRequest(`http://localhost/api/papers/${id}/forward-pass`);
  return GET(req, { params: Promise.resolve({ id }) });
}

describe('GET /api/papers/[id]/forward-pass', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('returns 200 with forward pass data for valid arch', async () => {
    mockExecSuccess(MOCK_FORWARD_PASS);
    const res = await getForwardPass('resnet50');
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.steps).toHaveLength(1);
    expect(data.steps[0].node_id).toBe('conv1');
  });

  it('accepts vit-b16 id', async () => {
    mockExecSuccess({ ...MOCK_FORWARD_PASS, architecture: 'vit-b16' });
    const res = await getForwardPass('vit-b16');
    expect(res.status).toBe(200);
  });

  it('accepts an-image-is-worth-16x16-words id', async () => {
    mockExecSuccess({ ...MOCK_FORWARD_PASS, architecture: 'an-image-is-worth-16x16-words' });
    const res = await getForwardPass('an-image-is-worth-16x16-words');
    expect(res.status).toBe(200);
  });

  it('returns 404 for unknown architecture', async () => {
    const res = await getForwardPass('gpt4');
    expect(res.status).toBe(404);
    const data = await res.json();
    expect(data.error).toMatch(/Unknown architecture/);
    expect(execFile).not.toHaveBeenCalled();
  });

  it('does not invoke python for unknown id', async () => {
    await getForwardPass('not-in-allowlist');
    expect(execFile).not.toHaveBeenCalled();
  });

  it('returns X-Cache: MISS on fresh request', async () => {
    mockExecSuccess(MOCK_FORWARD_PASS);
    // unet-biomedical is not cached by block-hierarchy tests (different module)
    mockExecSuccess({ ...MOCK_FORWARD_PASS, architecture: 'unet-biomedical' });
    const res = await getForwardPass('unet-biomedical');
    expect(res.headers.get('X-Cache')).toBe('MISS');
  });

  it('returns 408 on timeout', async () => {
    vi.mocked(execFile).mockImplementationOnce(
      (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
        (cb as (e: Error, out: string, err: string) => void)(new Error('timeout'), '', '');
        return { pid: 0 } as ReturnType<typeof execFile>;
      },
    );
    const res = await getForwardPass('ronneberger2015');
    expect(res.status).toBe(408);
  });

  it('returns 422 when python response has error field', async () => {
    mockExecSuccess({ error: 'architecture not supported' });
    const res = await getForwardPass('unet');
    expect(res.status).toBe(422);
  });

  it('passes --action forward-pass to python', async () => {
    mockExecSuccess(MOCK_FORWARD_PASS);
    await getForwardPass('resnet');
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--action', 'forward-pass']),
      expect.any(Object),
      expect.any(Function),
    );
  });
});
