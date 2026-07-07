'use client';

import { X } from 'lucide-react';
import { useEffect, useState, type FormEvent } from 'react';
import { useAuthModal } from './AuthModalContext';
import { apiPost, apiPostForm, setTokens, apiGet } from '@/lib/api';

const inp ='w-full bg-[#111111] border border-[#262626] rounded-lg px-3 py-2.5 text-sm text-white placeholder:text-[#525252] outline-none focus:border-[#A78BFA] transition-colors';

export function AuthModal() {
  const { isOpen, tab, setTab, close } = useAuthModal();
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    const fn = (e: KeyboardEvent) => { if (e.key === 'Escape') close(); };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, [isOpen, close]);

  if (!isOpen) return null;

  // Backend's /api/auth/login is OAuth2-password-form, not JSON, and only
  // returns {access_token, refresh_token, token_type} — no user profile.
  const doLogin = async (loginEmail: string, loginPassword: string) => {
    const formData = new URLSearchParams();
    formData.append('grant_type', 'password');
    formData.append('username', loginEmail); // backend expects 'username' field to contain the email for OAuth2
    formData.append('password', loginPassword);
    return apiPostForm<{ access_token: string; refresh_token?: string }>(
      '/api/auth/login', formData, 'application/x-www-form-urlencoded',
    );
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (tab === 'signup') {
        // Backend's RegisterRequest requires {email, name, password} exactly —
        // sending {username, ...} (the old shape) 422'd on every signup.
        // /api/auth/register only creates the account and returns the user
        // profile — it does NOT return tokens, so a real login call right
        // after is required (email verification isn't required to log in).
        const uname = username || email.split('@')[0];
        const registeredUser = await apiPost<Record<string, unknown>>('/api/auth/register', {
          name: uname,
          email,
          password,
        });
        const loginRes = await doLogin(email, password);
        setTokens({ access_token: loginRes.access_token, refresh_token: loginRes.refresh_token });
        if (typeof window !== 'undefined') {
          localStorage.setItem('user_profile', JSON.stringify(registeredUser));
        }
        close();
        window.location.reload();
      } else {
        const res = await doLogin(email, password);
        setTokens({ access_token: res.access_token, refresh_token: res.refresh_token });
        // Login only returns a token — persist a minimal profile so the navbar
        // can show the signed-in state after the reload.
        if (typeof window !== 'undefined') {
          localStorage.setItem('user_profile', JSON.stringify({ name: email.split('@')[0], email }));
        }
        close();
        window.location.reload();
      }
    } catch (err: unknown) {
      setError((err as Error).message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const handleOAuth = async (provider: string) => {
    try {
      setLoading(true);
      setError('');
      if (typeof window !== 'undefined') {
        localStorage.setItem('oauth_provider', provider);
        const redirectUri = window.location.origin + '/auth/callback';
        const res = await apiGet<{ url: string }>(`/api/auth/oauth/${provider}/authorize-url?redirect_uri=${encodeURIComponent(redirectUri)}`);
        window.location.href = res.url;
      }
    } catch (err: unknown) {
      setError((err as Error).message || `${provider} login failed`);
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    try {
      setLoading(true);
      setError('');
      const { GoogleAuthProvider, signInWithPopup } = await import('firebase/auth');
      const { firebaseAuth } = await import('@/lib/firebase');
      const credential = await signInWithPopup(firebaseAuth, new GoogleAuthProvider());
      const idToken = await credential.user.getIdToken();
      const res = await apiPost<{ access_token: string; refresh_token: string; user: { id: number; email: string; name: string } }>(
        '/api/auth/firebase/verify',
        { id_token: idToken },
      );
      setTokens({ access_token: res.access_token, refresh_token: res.refresh_token });
      localStorage.setItem('user_profile', JSON.stringify(res.user));
      close();
      window.location.reload();
    } catch (err: unknown) {
      const msg = (err as { message?: string; code?: string }).message || 'Google sign-in failed';
      if ((err as { code?: string }).code !== 'auth/popup-closed-by-user') setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" onClick={close}>
      <div
        className="relative mx-auto mt-[12vh] w-full max-w-[420px] rounded-2xl border border-[#262626] bg-[#111111] p-8"
        onClick={e => e.stopPropagation()}
      >
        <button type="button" onClick={close} aria-label="Close"
          className="absolute right-4 top-4 text-[#525252] transition-colors hover:text-white">
          <X size={18} />
        </button>

        <div className="mb-6 flex items-center gap-2">
          {(['signin', 'signup'] as const).map(t => (
            <button key={t} type="button" onClick={() => setTab(t)}
              className={'rounded-full px-4 py-1.5 text-sm transition-colors ' +
                (tab === t ? 'bg-[#A78BFA] font-semibold text-black' : 'text-[#A3A3A3] hover:text-white')}>
              {t === 'signin' ? 'Sign In' : 'Sign Up'}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="flex flex-col gap-3">
          {tab === 'signup' && (
            <input className={inp} placeholder="Username" value={username}
              onChange={e => setUsername(e.target.value)} autoComplete="username" />
          )}
          <input className={inp} placeholder="Email" type="email" value={email}
            onChange={e => setEmail(e.target.value)} autoComplete="email" />
          <input className={inp} placeholder="Password" type="password" value={password}
            onChange={e => setPassword(e.target.value)}
            minLength={tab === 'signup' ? 10 : undefined}
            autoComplete={tab === 'signup' ? 'new-password' : 'current-password'} />
          {tab === 'signup' && (
            <div className="text-[11px] text-[#525252] -mt-1.5">At least 10 characters</div>
          )}
          {tab === 'signin' && (
            <div className="flex justify-end">
              <a href="/auth/forgot-password" className="text-xs text-[#A3A3A3] transition-colors hover:text-[#A78BFA]">
                Forgot password?
              </a>
            </div>
          )}
          {error && <div className="text-xs text-red-500 bg-red-500/10 border border-red-500/20 p-2 rounded">{error}</div>}
          <button type="submit" disabled={loading}
            className="mt-4 w-full rounded-lg bg-[#A78BFA] py-2.5 font-semibold text-black transition-colors hover:bg-[#C4B5FD] disabled:opacity-50">
            {loading ? 'Processing...' : (tab === 'signin' ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <div className="mt-6 mb-6 flex items-center gap-4">
          <div className="h-px flex-1 bg-[#262626]"></div>
          <span className="text-xs text-[#525252]">OR</span>
          <div className="h-px flex-1 bg-[#262626]"></div>
        </div>

        <div className="flex flex-col gap-3">
          <button 
            onClick={() => handleOAuth('github')}
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#262626] bg-[#111111] py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#1A1A1A] hover:border-[#404040] disabled:opacity-50"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5 fill-current" aria-hidden="true">
              <path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.6.113.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
            </svg>
            Continue with GitHub
          </button>

          <button
            onClick={handleGoogleSignIn}
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#262626] bg-[#111111] py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#1A1A1A] hover:border-[#404040] disabled:opacity-50"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
            </svg>
            Continue with Google
          </button>
        </div>

      </div>
    </div>
  );
}

export default AuthModal;
