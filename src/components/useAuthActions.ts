import { apiPost, apiPostForm, setTokens, apiGet } from '@/lib/api';

type User = { id?: number; name: string; email: string };

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
    const loginRes = await apiPostForm<{ access_token: string; refresh_token?: string }>(
      '/api/auth/login', formData, 'application/x-www-form-urlencoded'
    );
    setTokens({ access_token: loginRes.access_token, refresh_token: loginRes.refresh_token });
    
    // fetch canonical profile
    return fetchMe({ name: email.split('@')[0], email });
  };

  const signupWithPassword = async (name: string, email: string, password: string): Promise<User> => {
    const uname = name || email.split('@')[0];
    await apiPost<Record<string, unknown>>('/api/auth/register', {
      name: uname,
      email,
      password,
    });
    // /api/auth/register only creates the account and returns the user profile, not tokens.
    // So we log in immediately after.
    return loginWithPassword(email, password);
  };

  const loginWithGoogle = async (): Promise<User> => {
    const { GoogleAuthProvider, signInWithPopup } = await import('firebase/auth');
    const { firebaseAuth } = await import('@/lib/firebase');
    const credential = await signInWithPopup(firebaseAuth, new GoogleAuthProvider());
    const idToken = await credential.user.getIdToken();
    const res = await apiPost<{ access_token: string; refresh_token: string; user: { id: number; email: string; name: string } }>(
      '/api/auth/firebase/verify',
      { id_token: idToken },
    );
    setTokens({ access_token: res.access_token, refresh_token: res.refresh_token });
    
    // fetch canonical profile
    return fetchMe(res.user);
  };

  return { loginWithPassword, signupWithPassword, loginWithGoogle };
}
