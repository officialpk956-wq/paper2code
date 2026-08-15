import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { apiGet, apiPost } from '@/lib/api';

describe('src/lib/api.ts — Request Timeouts & Silent Token Refresh', () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('Scenario 1: Requests that hang reject with a user-friendly timeout message', async () => {
    // Mock fetch to simulate an aborted request due to timeout
    const abortError = new Error('The operation was aborted due to timeout');
    abortError.name = 'TimeoutError';

    global.fetch = vi.fn().mockRejectedValue(abortError);

    await expect(apiGet('/api/slow-endpoint')).rejects.toThrow(
      'Request timed out — check your connection and try again'
    );
  });

  it('Scenario 2: A 401 response triggers silent refresh and retries the original request', async () => {
    localStorage.setItem('access_token', 'expired_access_token');
    localStorage.setItem('refresh_token', 'valid_refresh_token');

    let callCount = 0;
    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      callCount++;
      if (url.includes('/api/auth/refresh')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              access_token: 'fresh_access_token',
              refresh_token: 'rotated_refresh_token',
              token_type: 'bearer',
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } }
          )
        );
      }

      // First call to protected endpoint -> 401
      if (callCount === 1) {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Token expired' }), {
            status: 401,
            headers: { 'Content-Type': 'application/json' },
          })
        );
      }

      // Retry call -> verify it used the refreshed token
      const authHeader = (init?.headers as Record<string, string>)?.['Authorization'];
      if (authHeader === 'Bearer fresh_access_token') {
        return Promise.resolve(
          new Response(JSON.stringify({ data: 'success_payload' }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          })
        );
      }

      return Promise.resolve(new Response('Unauthorized', { status: 401 }));
    });

    const result = await apiGet<{ data: string }>('/api/protected-data');
    expect(result).toEqual({ data: 'success_payload' });

    // Stored tokens must have been updated to rotated token
    expect(localStorage.getItem('access_token')).toBe('fresh_access_token');
    expect(localStorage.getItem('refresh_token')).toBe('rotated_refresh_token');
  });

  it('Scenario 3: If /api/auth/refresh fails (401), tokens are cleared and error is thrown', async () => {
    localStorage.setItem('access_token', 'expired_token');
    localStorage.setItem('refresh_token', 'revoked_refresh_token');

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/auth/refresh')) {
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Invalid refresh token' }), { status: 401 })
        );
      }
      return Promise.resolve(new Response(JSON.stringify({ detail: 'Unauthorized' }), { status: 401 }));
    });

    await expect(apiGet('/api/protected')).rejects.toThrow('Unauthorized');
    expect(localStorage.getItem('access_token')).toBeNull();
    expect(localStorage.getItem('refresh_token')).toBeNull();
  });

  it('Scenario 4: Concurrent 401s deduplicate and trigger only ONE refresh request', async () => {
    localStorage.setItem('access_token', 'expired_token');
    localStorage.setItem('refresh_token', 'valid_refresh');

    let refreshCalls = 0;
    // Each data URL 401s on its first call (simulating the expired token)
    // and succeeds on retry — only once its Authorization header carries the
    // refreshed token, proving the retry actually used the new token rather
    // than just happening to succeed regardless of dedup.
    const seenOnce = new Set<string>();
    global.fetch = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (url.includes('/api/auth/refresh')) {
        refreshCalls++;
        return new Promise(resolve => {
          setTimeout(() => {
            resolve(
              new Response(
                JSON.stringify({
                  access_token: 'deduped_access_token',
                  refresh_token: 'deduped_refresh_token',
                }),
                { status: 200, headers: { 'Content-Type': 'application/json' } }
              )
            );
          }, 10);
        });
      }

      if (!seenOnce.has(url)) {
        seenOnce.add(url);
        return Promise.resolve(new Response(JSON.stringify({ detail: 'Token expired' }), { status: 401 }));
      }

      const authHeader = (init?.headers as Record<string, string>)?.['Authorization'];
      expect(authHeader).toBe('Bearer deduped_access_token');
      return Promise.resolve(
        new Response(JSON.stringify({ success: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });

    // Fire 2 concurrent requests, each of which will 401 once and need a refresh
    const [res1, res2] = await Promise.all([
      apiGet('/api/data1'),
      apiPost('/api/data2', { test: 123 }),
    ]);

    expect(res1).toEqual({ success: true });
    expect(res2).toEqual({ success: true });
    // The core claim of this scenario: two concurrent 401s share a single
    // in-flight refresh instead of each triggering their own.
    expect(refreshCalls).toBe(1);
  });

  it('Scenario 5: When no refresh token exists in localStorage, 401 immediately clears tokens and throws', async () => {
    localStorage.setItem('access_token', 'expired_token');
    // No refresh_token in storage

    const fetchSpy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Token expired' }), { status: 401 })
    );
    global.fetch = fetchSpy;

    await expect(apiGet('/api/protected')).rejects.toThrow('Unauthorized');

    // No refresh request should have been made
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem('access_token')).toBeNull();
  });
});
