'use client';

import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';
import { apiPost } from '@/lib/api';

type State = 'pending' | 'success' | 'used' | 'expired' | 'invalid' | 'missing' | 'error';
const headings: Record<State, string> = {
  pending: 'Verifying your email…', success: 'Email verified', used: 'Link already used',
  expired: 'Link expired', invalid: 'Invalid verification link', missing: 'Verification token missing', error: 'Verification unavailable',
};
export function VerifyEmail() {
  const token = useSearchParams().get('token');
  const [result, setResult] = useState<{ token: string; state: State } | null>(null);
  const request = useRef<{ token: string; promise: Promise<State> } | null>(null);
  const state: State = !token ? 'missing' : result?.token === token ? result.state : 'pending';
  useEffect(() => {
    if (!token) return;
    let active = true;
    // Reuse the in-flight redemption during React's effect replay; tokens are single use.
    if (request.current?.token !== token) {
      const promise = apiPost<{ status: string }>('/api/auth/verify-email', { token })
        .then<State>(response => response.status === 'ok' ? 'success' : 'error')
        .catch((error): State => {
          const message = error instanceof Error ? error.message : '';
          if (message === 'Verification link already used') return 'used';
          if (message === 'Verification token has expired') return 'expired';
          if (message === 'Invalid verification token') return 'invalid';
          return 'error';
        });
      request.current = { token, promise };
    }
    request.current.promise.then(next => { if (active) setResult({ token, state: next }); });
    return () => { active = false; };
  }, [token]);
  return <div className="flex min-h-screen items-center justify-center bg-[#0A120D] px-4">
    <div className="w-full max-w-[420px] rounded-2xl border border-[#262626] bg-[#111111] p-8 text-center">
      <h1 className="text-xl font-bold text-white">{headings[state]}</h1>
      {state === 'pending' ? <p role="status" className="mt-3 text-[#A3A3A3]">Verifying your email…</p> :
        <p className="mt-3 text-sm text-[#A3A3A3]">{state === 'success' ? 'Your email address is confirmed.' :
          state === 'used' ? 'This link has already been redeemed. Sign in to your account.' :
          state === 'error' ? 'We could not confirm verification. Check your connection and try reopening the link.' :
          'Open a complete, current verification link from your email, or request a new one.'}</p>}
      <Link href="/" className="mt-6 inline-block rounded-lg bg-[#A78BFA] px-6 py-2.5 text-sm font-semibold text-black">Go to paper2code</Link>
    </div>
  </div>;
}
