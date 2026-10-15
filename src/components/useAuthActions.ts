import { apiPost, apiPostForm, setTokens, apiGet, clearTokens } from '@/lib/api';
import { useCallback, useState } from 'react';
import { z } from 'zod';

type User = { id?: number; name: string; email: string };

const AUTH_TIMEOUT_MS = 60000;
const sessionSchema = z.object({ access_token: z.string().min(1), refresh_token: z.string().min(1).optional() });
const challengeSchema = z.object({ mfa_required: z.literal(true), mfa_token: z.string().min(1) });

async function retryColdBackendOnce<T>(request: () => Promise<T>): Promise<T> {
  try {
    return await request();
  } catch (error) {
    if (error instanceof Error && error.name === 'TimeoutError') {
      return request();
    }
    throw error;
  }
}

export function useAuthActions() {
  const [mfaChallenge, setMfaChallenge] = useState<{ token: string; email: string } | null>(null);
  const cancelMfa = useCallback(() => setMfaChallenge(null), []);
  const fetchMe = async (fallback: User): Promise<User> => {
    try {
      const me = await apiGet<User>('/api/auth/me');
      return { ...me, name: me.name || me.email?.split('@')[0] || 'User' };
    } catch {
      return fallback;
    }
  };

  const loginWithPassword = async (email: string, password: string): Promise<User | null> => {
    setMfaChallenge(null);
    const formData = new URLSearchParams();
    formData.append('grant_type', 'password');
    formData.append('username', email); // backend expects 'username' field to contain the email for OAuth2
    formData.append('password', password);
    const loginRes = await retryColdBackendOnce(() =>
      apiPostForm<unknown>(
        '/api/auth/login',
        formData,
        'application/x-www-form-urlencoded',
        AUTH_TIMEOUT_MS
      )
    );
    const challenge = challengeSchema.safeParse(loginRes);
    if (challenge.success) {
      clearTokens();
      setMfaChallenge({ token: challenge.data.mfa_token, email });
      return null;
    }
    const session = sessionSchema.safeParse(loginRes);
    if (!session.success) throw new Error('Invalid login response');
    setTokens(session.data);
    
    // fetch canonical profile
    return fetchMe({ name: email.split('@')[0], email });
  };

  const signupWithPassword = async (name: string, email: string, password: string): Promise<User | null> => {
    const uname = name || email.split('@')[0];
    await retryColdBackendOnce(() =>
      apiPost<Record<string, unknown>>(
        '/api/auth/register',
        {
          name: uname,
          email,
          password,
        },
        AUTH_TIMEOUT_MS
      )
    );
    // /api/auth/register only creates the account and returns the user profile, not tokens.
    // So we log in immediately after.
    return loginWithPassword(email, password);
  };

  const verifyMfa = async (code: string): Promise<User> => {
    if (!mfaChallenge) throw new Error('Sign in again to start verification');
    // A challenge is not a session: preserve the MFA error and never refresh/retry it.
    const base = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    const response = await fetch(`${base}/api/auth/login/mfa`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mfa_token: mfaChallenge.token, code: code.trim() }),
      signal: AbortSignal.timeout(AUTH_TIMEOUT_MS),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Verification failed');
    const session = sessionSchema.safeParse(body);
    if (!session.success) throw new Error('Invalid login response');
    setTokens(session.data);
    const user = await fetchMe({ name: mfaChallenge.email.split('@')[0], email: mfaChallenge.email });
    setMfaChallenge(null);
    return user;
  };

  const loginWithGoogle = async (): Promise<User> => {
    const redirectUri = window.location.origin + '/auth/callback';
    localStorage.setItem('oauth_provider', 'google');
    const res = await retryColdBackendOnce(() =>
      apiGet<{ url: string; state: string }>(
        `/api/auth/oauth/google/authorize-url?redirect_uri=${encodeURIComponent(redirectUri)}`,
        AUTH_TIMEOUT_MS
      )
    );
    window.location.href = res.url;
    return new Promise(() => {}); // page redirects; this never resolves
  };

  return { loginWithPassword, signupWithPassword, loginWithGoogle, mfaChallenge, verifyMfa, cancelMfa };
}
