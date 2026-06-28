'use client';

import { useEffect, useState, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

function OAuthCallbackHandler() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [error, setError] = useState('');

  useEffect(() => {
    const code = searchParams.get('code');
    const state = searchParams.get('state');
    const provider = localStorage.getItem('oauth_provider') || 'google';

    if (!code) {
      setError('No authorization code received. Please try again.');
      return;
    }

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const redirectUri = window.location.origin + '/auth/callback';

    fetch(`${apiUrl}/api/auth/oauth/${provider}/exchange`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, state: state || '', redirect_uri: redirectUri }),
    })
      .then(res => {
        if (!res.ok) throw new Error(`Exchange failed: ${res.status}`);
        return res.json();
      })
      .then(data => {
        localStorage.setItem('access_token', data.access_token);
        localStorage.removeItem('oauth_provider');
        router.replace('/dashboard');
      })
      .catch(() => {
        setError('OAuth sign-in failed. Please try again or use email/password.');
      });
  }, [router, searchParams]);

  if (error) {
    return (
      <div className="text-center">
        <div className="w-12 h-12 mx-auto rounded-full bg-[rgba(239,68,68,0.1)] flex items-center justify-center mb-4">
          <svg className="w-6 h-6 text-[--status-error]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
        <h2 className="text-xl font-heading font-bold text-[--color-text-primary] mb-2">Sign-in Failed</h2>
        <p className="text-sm text-[--color-text-secondary] mb-6">{error}</p>
        <a href="/login" className="text-[--accent-primary] hover:text-[--accent-light] text-sm font-medium">
          Back to Log In
        </a>
      </div>
    );
  }

  return (
    <div className="text-center">
      <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-[0_0_16px_rgba(139,92,246,0.3)] mb-4 animate-pulse">
        <span className="text-lg font-black text-white">P</span>
      </div>
      <h2 className="text-xl font-heading font-bold text-[--color-text-primary] mb-2">Signing you in…</h2>
      <p className="text-sm text-[--color-text-tertiary]">Please wait while we complete your sign-in.</p>
    </div>
  );
}

export default function OAuthCallbackPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[--bg-body] p-4">
      <div className="w-full max-w-md bg-[--bg-surface] border border-[--color-border] rounded-xl p-8 shadow-[0_4px_24px_rgba(0,0,0,0.2)]">
        <Suspense fallback={
          <div className="text-center text-[--color-text-tertiary] text-sm">Loading…</div>
        }>
          <OAuthCallbackHandler />
        </Suspense>
      </div>
    </div>
  );
}
