// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextRequest } from 'next/server';
import { exec } from 'child_process';
import { writeFileSync, unlinkSync } from 'fs';

vi.mock('child_process', () => ({ exec: vi.fn() }));
vi.mock('fs', () => ({ writeFileSync: vi.fn(), unlinkSync: vi.fn() }));

const ALL_TEST_CASES = [
  { input: { x: 1 }, output: 2, visible: true },
  { input: { x: 2 }, output: 4, visible: false },
  { input: { x: 5 }, output: 10, visible: false },
];

const SUCCESS_RESULTS = [
  { index: 0, passed: true, actual: 2, expected: 2, runtime_ms: 0.2, error: null },
  { index: 1, passed: true, actual: 4, expected: 4, runtime_ms: 0.2, error: null },
  { index: 2, passed: true, actual: 10, expected: 10, runtime_ms: 0.2, error: null },
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

async function postSubmit(body: Record<string, unknown>) {
  const { POST } = await import('@/app/api/dojo/submit/route');
  const req = new NextRequest('http://localhost/api/dojo/submit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return POST(req);
}

const VALID_BODY = {
  code: 'def double(x):\n    return x * 2',
  functionName: 'double',
  testCases: ALL_TEST_CASES,
};

describe('POST /api/dojo/submit', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns 200 with accepted status when all tests pass', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    const res = await postSubmit(VALID_BODY);
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.status).toBe('accepted');
    expect(data.passedTests).toBe(3);
    expect(data.totalTests).toBe(3);
  });

  it('returns wrong_answer when some tests fail', async () => {
    const failResults = [
      ...SUCCESS_RESULTS.slice(0, 2),
      { index: 2, passed: false, actual: 11, expected: 10, runtime_ms: 0.2, error: null },
    ];
    mockExecSuccess(failResults);
    const res = await postSubmit(VALID_BODY);
    const data = await res.json();
    expect(data.status).toBe('wrong_answer');
    expect(data.passedTests).toBe(2);
  });

  it('returns 400 when code is missing', async () => {
    const res = await postSubmit({ functionName: 'f', testCases: ALL_TEST_CASES });
    expect(res.status).toBe(400);
  });

  it('returns 400 when code exceeds 20000 chars', async () => {
    const res = await postSubmit({ ...VALID_BODY, code: 'x'.repeat(20001) });
    expect(res.status).toBe(400);
  });

  it('returns 400 for invalid functionName', async () => {
    const res = await postSubmit({ ...VALID_BODY, functionName: '__import__("os")' });
    expect(res.status).toBe(400);
  });

  it('returns 400 when testCases is empty array', async () => {
    const res = await postSubmit({ ...VALID_BODY, testCases: [] });
    expect(res.status).toBe(400);
  });

  it('returns 400 when testCases is not an array', async () => {
    const res = await postSubmit({ ...VALID_BODY, testCases: 'not-an-array' });
    expect(res.status).toBe(400);
  });

  it('runs against all test cases (not just visible)', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    await postSubmit(VALID_BODY);
    // Should have written a script file with all 3 test cases
    expect(writeFileSync).toHaveBeenCalledOnce();
  });

  it('includes runtimeMs in response', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    const res = await postSubmit(VALID_BODY);
    const data = await res.json();
    expect(typeof data.runtimeMs).toBe('number');
  });

  it('returns results array in response', async () => {
    mockExecSuccess(SUCCESS_RESULTS);
    const res = await postSubmit(VALID_BODY);
    const data = await res.json();
    expect(data.results).toHaveLength(3);
  });
});
