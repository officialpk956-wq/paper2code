import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import { useAuthActions } from '@/components/useAuthActions';

beforeEach(() => { localStorage.clear(); vi.restoreAllMocks(); });

test('ordinary login works, but MFA challenge stores no access token', async () => {
  const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(new Response(JSON.stringify({ access_token: 'ordinary-token', refresh_token: 'refresh' })))
    .mockResolvedValueOnce(new Response(JSON.stringify({ name: 'Tester', email: 'test@example.com' })));
  const { result } = renderHook(() => useAuthActions());
  await act(async () => { await result.current.loginWithPassword('test@example.com', 'password'); });
  expect(localStorage.getItem('access_token')).toBe('ordinary-token');
  localStorage.clear();
  fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ mfa_required: true, mfa_token: 'challenge-only' })))
    .mockResolvedValueOnce(new Response(JSON.stringify({ name: 'Tester', email: 'test@example.com' })));
  await act(async () => { await result.current.loginWithPassword('test@example.com', 'password'); });
  expect(localStorage.getItem('access_token')).toBeNull();
});

test('malformed login response cannot become a session', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ access_token: '' })));
  const { result } = renderHook(() => useAuthActions());
  await expect(result.current.loginWithPassword('test@example.com', 'password')).rejects.toThrow('Invalid login response');
  expect(localStorage.getItem('access_token')).toBeNull();
});
