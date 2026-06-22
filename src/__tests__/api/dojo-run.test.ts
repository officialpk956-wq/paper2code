// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';
import { exec } from 'child_process';
import { writeFileSync, unlinkSync } from 'fs';

vi.mock('child_process', () => ({ exec: vi.fn() }));
vi.mock('fs', () => ({ writeFileSync: vi.fn(), unlinkSync: vi.fn() }));

const SUCCESS_RESULTS = [
  { index: 0, passed: true, actual: 42, expected: 42, runtime_ms: 0.3, error: null },
];

function mockExecSuccess(results: unknown[]) {
  vi.mocked(exec).mockImplementationOnce(
    (_cmd: unknown, _opts: unknown, cb: unknown) => {
      (cb as (e: null, out: string, err: string) => void)(
        null,
        JSON.stringify({ results, status: 'ok' }),
        '',
      );
      return { pid: 0 } as ReturnType<typeof exec>;
    },
  );
}

function mockExecTimeout() {
  vi.mocked(exec).mockImplementationOnce(
    (_cmd: unknown, _opts: unknown, cb: unknown) => {
      (cb as (e: Error, out: string, err: string) => void)(new Error('timeout'), '', '');
      return { pid: 0 } as ReturnType<typeof exec>;
    },
  );
}

async function postRun(body: Record<string, unknown>) {
  const { POST } = await import('@/app/api/dojo/run/route');
  const req = new NextRequest('http://localhost/api/dojo/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return POST(req);
}

const VALID_BODY = {
  code: 'def solution(x):\n    return x * 2',
  functionName: 'solution',
  testCases: [{ input: { x: 21 }, output: 42, visible: true }],
  visibleOnly: true,
};

describe('POST /api/dojo/run', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns 200 with results on valid request', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    const res = await postRun(VALID_BODY);
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.results).toHaveLength(1);
    expect(data.results[0].passed).toBe(true);
  });

  it('writes a temp script file', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    await postRun(VALID_BODY);
    expect(writeFileSync).toHaveBeenCalledOnce();
  });

  it('deletes temp file after execution', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    await postRun(VALID_BODY);
    expect(unlinkSync).toHaveBeenCalledOnce();
  });

  it('returns 400 when code is missing', async () => {
    const res = await postRun({ functionName: 'f', testCases: [{ input: {}, output: 1, visible: true }] });
    expect(res.status).toBe(400);
  });

  it('returns 400 when code exceeds 20000 chars', async () => {
    const res = await postRun({ ...VALID_BODY, code: 'x'.repeat(20001) });
    expect(res.status).toBe(400);
  });

  it('returns 400 when functionName is missing', async () => {
    const res = await postRun({ code: 'def f(): pass', testCases: [{ input: {}, output: 1, visible: true }] });
    expect(res.status).toBe(400);
  });

  it('returns 400 for invalid Python identifier functionName', async () => {
    const res = await postRun({ ...VALID_BODY, functionName: '1invalid' });
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.error).toMatch(/identifier/);
  });

  it('returns 400 for functionName with special characters', async () => {
    const res = await postRun({ ...VALID_BODY, functionName: 'os.system' });
    expect(res.status).toBe(400);
  });

  it('returns 408 on timeout', async () => {
    mockExecTimeout();
    const res = await postRun(VALID_BODY);
    expect(res.status).toBe(408);
    const data = await res.json();
    expect(data.timeout).toBe(true);
  });

  it('includes totalMs in successful response', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    const res = await postRun(VALID_BODY);
    const data = await res.json();
    expect(typeof data.totalMs).toBe('number');
  });

  it('accepts camelCase functionName as valid identifier', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    const res = await postRun({ ...VALID_BODY, functionName: 'myFunction' });
    expect(res.status).toBe(200);
  });

  it('accepts underscore-prefixed functionName', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    const res = await postRun({ ...VALID_BODY, functionName: '_helper' });
    expect(res.status).toBe(200);
  });
});
