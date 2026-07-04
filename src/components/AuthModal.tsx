'use client';

import { X } from 'lucide-react';
import { useEffect, useState, type FormEvent } from 'react';
import { useAuthModal } from './AuthModalContext';
import { apiPost, apiPostForm, setTokens } from '@/lib/api';

const inp ='w-full bg-[#16241B] border border-[#223429] rounded-lg px-3 py-2.5 text-sm text-white placeholder:text-[#525252] outline-none focus:border-[#34D399] transition-colors';

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
        const registeredUser = await apiPost<Record<string, any>>('/api/auth/register', {
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
    } catch (err: any) {
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm" onClick={close}>
      <div
        className="relative mx-auto mt-[12vh] w-full max-w-[420px] rounded-2xl border border-[#223429] bg-[#121D16] p-8"
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
                (tab === t ? 'bg-[#34D399] font-semibold text-black' : 'text-[#A3A3A3] hover:text-white')}>
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
              <button type="button" className="text-xs text-[#A3A3A3] transition-colors hover:text-[#34D399]">
                Forgot password?
              </button>
            </div>
          )}
          {error && <div className="text-xs text-red-500 bg-red-500/10 border border-red-500/20 p-2 rounded">{error}</div>}
          <button type="submit" disabled={loading}
            className="mt-4 w-full rounded-lg bg-[#34D399] py-2.5 font-semibold text-black transition-colors hover:bg-[#4ADEA8] disabled:opacity-50">
            {loading ? 'Processing...' : (tab === 'signin' ? 'Sign In' : 'Create Account')}
          </button>
        </form>

      </div>
    </div>
  );
}

export default AuthModal;
