import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { parseModel, parsePytorchModel } from '@/lib/model-viz-api';

/**
 * Regression: the Model Visualizer client used raw fetch with a hand-built
 * Authorization header, bypassing the shared client's refresh-on-401. Access
 * tokens live 15 minutes, so fifteen minutes after login every upload on the
 * page failed with "Could not validate credentials" while the rest of the app
 * refreshed silently. Reported from production on 2026-09-19.
 */
describe('model-viz-api — goes through the refreshing client', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  function mockBackend(graph: object) {
    const seen: { url: string; auth?: string }[] = [];
    global.fetch = vi.fn().mockImplementation(async (url: string, init?: RequestInit) => {
      const auth = (init?.headers as Record<string, string> | undefined)?.Authorization;
      seen.push({ url, auth });
      if (url.includes('/api/auth/refresh')) {
        return new Response(JSON.stringify({ access_token: 'fresh_token' }), { status: 200 });
      }
      if (auth === 'Bearer expired_token') {
        return new Response(JSON.stringify({ detail: 'Could not validate credentials' }), { status: 401 });
      }
      return new Response(JSON.stringify(graph), { status: 200 });
    }) as unknown as typeof fetch;
    return seen;
  }

  it('an expired token on /api/model/parse is refreshed and the upload retried', async () => {
    localStorage.setItem('access_token', 'expired_token');
    localStorage.setItem('refresh_token', 'valid_refresh');
    const graph = { nodes: [{ id: 'n0' }], edges: [], groups: [], meta: { total_nodes: 1 } };
    const seen = mockBackend(graph);

    const result = await parseModel(new File([new Uint8Array([1, 2, 3])], 'tiny.onnx'));

    expect(result).toEqual(graph);
    const parseCalls = seen.filter((s) => s.url.endsWith('/api/model/parse'));
    expect(parseCalls.map((s) => s.auth)).toEqual(['Bearer expired_token', 'Bearer fresh_token']);
    expect(seen.some((s) => s.url.includes('/api/auth/refresh'))).toBe(true);
  });

  it('the PyTorch parse is given a sandbox-length timeout, not the 15s default', async () => {
    localStorage.setItem('access_token', 'ok_token');
    mockBackend({ nodes: [], edges: [], groups: [], meta: {} });
    const timeoutSpy = vi.spyOn(AbortSignal, 'timeout');

    await parsePytorchModel(new File([new Uint8Array([1])], 'm.pt'), [3, 32, 32]);

    const ms = timeoutSpy.mock.calls.map((c) => c[0]);
    expect(Math.max(...ms)).toBeGreaterThanOrEqual(300_000);
  });
});
