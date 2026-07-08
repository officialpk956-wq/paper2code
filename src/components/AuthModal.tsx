'use client';

import { X } from 'lucide-react';
import { useEffect, useState, type FormEvent } from 'react';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { useAuthModal } from './AuthModalContext';
import { useAuthActions } from './useAuthActions';

const inp ='w-full bg-[#111111] border border-[#262626] rounded-lg px-3 py-2.5 text-sm text-white placeholder:text-[#525252] outline-none focus:border-[#A78BFA] transition-colors';

export function AuthModal() {
  const { isOpen, tab, setTab, close, signIn } = useAuthModal();
  const { loginWithPassword, signupWithPassword, loginWithGoogle } = useAuthActions();
  const shouldReduceMotion = useReducedMotion();
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

  const cardVariants = {
    hidden:  { opacity: 0, scale: shouldReduceMotion ? 1 : 0.95, y: shouldReduceMotion ? 0 : 8 },
    visible: { opacity: 1, scale: 1, y: 0 },
    exit:    { opacity: 0, scale: shouldReduceMotion ? 1 : 0.97, y: shouldReduceMotion ? 0 : 4 },
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (tab === 'signup') {
        const user = await signupWithPassword(username, email, password);
        signIn(user);
      } else {
        const user = await loginWithPassword(email, password);
        signIn(user);
      }
    } catch (err: unknown) {
      setError((err as Error).message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    try {
      setLoading(true);
      setError('');
      const user = await loginWithGoogle();
      signIn(user);
    } catch (err: unknown) {
      const msg = (err as { message?: string; code?: string }).message || 'Google sign-in failed';
      if ((err as { code?: string }).code !== 'auth/popup-closed-by-user') setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          key="auth-backdrop"
          className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm"
          onClick={close}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18, ease: 'easeOut' }}
        >
          <motion.div
            key="auth-card"
            className="relative mx-auto mt-[12vh] w-full max-w-[420px] rounded-2xl border border-[#262626] bg-[#111111] p-8"
            onClick={e => e.stopPropagation()}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            transition={{
              type: 'spring',
              stiffness: 420,
              damping: 28,
              mass: 0.8,
            }}
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
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default AuthModal;
