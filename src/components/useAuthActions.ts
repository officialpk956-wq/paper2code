import { apiPost, apiPostForm, setTokens, apiGet } from '@/lib/api';

type User = { id?: number; name: string; email: string };

const AUTH_TIMEOUT_MS = 60000;

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
  const fetchMe = async (fallback: User): Promise<User> => {
    try {
      const me = await apiGet<User>('/api/auth/me');
      return { ...me, name: me.name || me.email?.split('@')[0] || 'User' };
    } catch {
      return fallback;
    }
  };

  const loginWithPassword = async (email: string, password: string): Promise<User> => {
    const formData = new URLSearchParams();
    formData.append('grant_type', 'password');
    formData.append('username', email); // backend expects 'username' field to contain the email for OAuth2
    formData.append('password', password);
    const loginRes = await retryColdBackendOnce(() =>
      apiPostForm<{ access_token: string; refresh_token?: string }>(
        '/api/auth/login',
        formData,
        'application/x-www-form-urlencoded',
        AUTH_TIMEOUT_MS
      )
    );
    setTokens({ access_token: loginRes.access_token, refresh_token: loginRes.refresh_token });
    
    // fetch canonical profile
    return fetchMe({ name: email.split('@')[0], email });
  };

  const signupWithPassword = async (name: string, email: string, password: string): Promise<User> => {
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

  return { loginWithPassword, signupWithPassword, loginWithGoogle };
}
