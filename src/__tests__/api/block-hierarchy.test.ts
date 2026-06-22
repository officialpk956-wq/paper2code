// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';
import { execFile } from 'child_process';

vi.mock('child_process', () => ({ execFile: vi.fn() }));

const MOCK_HIERARCHY = {
  paper_id: 'resnet',
  name: 'ResNet-50',
  description: 'Deep Residual Learning',
  input_shape: [1, 3, 224, 224],
  total_flops_mflops: 4100,
  total_params_M: 25.6,
  stages: [],
};

function mockExecSuccess(data: Record<string, unknown>) {
  vi.mocked(execFile).mockImplementationOnce(
    (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
      (cb as (e: null, out: string, err: string) => void)(null, JSON.stringify(data), '');
      return { pid: 0 } as ReturnType<typeof execFile>;
    },
  );
}

async function getHierarchy(id: string) {
  const { GET } = await import('@/app/api/papers/[id]/block-hierarchy/route');
  const req = new NextRequest(`http://localhost/api/papers/${id}/block-hierarchy`);
  return GET(req, { params: Promise.resolve({ id }) });
}

describe('GET /api/papers/[id]/block-hierarchy', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('returns 200 with hierarchy for valid arch id', async () => {
    mockExecSuccess(MOCK_HIERARCHY);
    const res = await getHierarchy('resnet50');
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.name).toBe('ResNet-50');
  });

  it('accepts resnet id', async () => {
    mockExecSuccess(MOCK_HIERARCHY);
    const res = await getHierarchy('resnet');
    expect(res.status).toBe(200);
  });

  it('accepts vit id', async () => {
    mockExecSuccess({ ...MOCK_HIERARCHY, paper_id: 'vit', name: 'ViT-B/16' });
    const res = await getHierarchy('vit');
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.name).toBe('ViT-B/16');
  });

  it('accepts transformer id', async () => {
    mockExecSuccess({ ...MOCK_HIERARCHY, paper_id: 'transformer', name: 'Transformer' });
    const res = await getHierarchy('transformer');
    expect(res.status).toBe(200);
  });

  it('returns 404 for unknown architecture id', async () => {
    const res = await getHierarchy('unknown-arch');
    expect(res.status).toBe(404);
    const data = await res.json();
    expect(data.error).toMatch(/Unknown architecture/);
    expect(execFile).not.toHaveBeenCalled();
  });

  it('returns 404 for sql-injection-style id', async () => {
    const res = await getHierarchy('resnet; rm -rf /');
    expect(res.status).toBe(404);
    expect(execFile).not.toHaveBeenCalled();
  });

  it('returns X-Cache: MISS on first call', async () => {
    mockExecSuccess(MOCK_HIERARCHY);
    // Use unique id each test — but note cache is per arch; resnet50 may be cached from prior test
    // We use a new valid id to guarantee MISS
    mockExecSuccess({ ...MOCK_HIERARCHY, paper_id: 'unet', name: 'U-Net' });
    const res = await getHierarchy('unet');
    expect(res.headers.get('X-Cache')).toBe('MISS');
  });

  it('returns 408 on timeout', async () => {
    vi.mocked(execFile).mockImplementationOnce(
      (_f: unknown, _a: unknown, _o: unknown, cb: unknown) => {
        (cb as (e: Error, out: string, err: string) => void)(new Error('timeout'), '', '');
        return { pid: 0 } as ReturnType<typeof execFile>;
      },
    );
    // Use a fresh arch to avoid cache hit
    const res = await getHierarchy('vaswani2017');
    expect(res.status).toBe(408);
  });

  it('returns 422 when python response has error field', async () => {
    mockExecSuccess({ error: 'architecture not implemented' });
    const res = await getHierarchy('attention-is-all-you-need');
    expect(res.status).toBe(422);
  });

  it('passes --action hierarchy to python', async () => {
    mockExecSuccess(MOCK_HIERARCHY);
    await getHierarchy('deep-residual-learning');
    expect(execFile).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(['--action', 'hierarchy']),
      expect.any(Object),
      expect.any(Function),
    );
  });
});
