'use client';

import { useEffect, useState, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { apiPost, setTokens } from '@/lib/api';

function OAuthCallbackContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [error, setError] = useState<string>('');

  useEffect(() => {
    const code = searchParams.get('code');
    const state = searchParams.get('state');
    // Provider is stored in localStorage before the OAuth redirect since the
    // callback URL doesn't include it (OAuth providers only append ?code&state).
    const storedProvider = typeof window !== 'undefined' ? localStorage.getItem('oauth_provider') : null;

    if (!code || !state || !storedProvider) {
      setError('Missing callback parameters');
      return;
    }

    const exchange = async () => {
      try {
        const payload = {
          code,
          state,
          redirect_uri: window.location.origin + '/auth/callback'
        };
        const res = await apiPost<any>(`/api/auth/oauth/${storedProvider}/exchange`, payload);
        
        setTokens({ access_token: res.access_token, refresh_token: res.refresh_token });
        if (typeof window !== 'undefined') {
          localStorage.setItem('user_profile', JSON.stringify(res.user));
          localStorage.removeItem('oauth_provider'); // cleanup
        }
        
        // Go back home or dashboard
        window.location.href = '/'; 
      } catch (err: any) {
        setError(err.message || 'OAuth exchange failed');
      }
    };

    exchange();
  }, [searchParams, router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-black text-white">
      {error ? (
        <div className="text-center">
          <p className="text-red-500 mb-4">{error}</p>
          <button onClick={() => router.push('/')} className="text-[#A78BFA] hover:underline">
            Return to Home
          </button>
        </div>
      ) : (
        <p className="animate-pulse">Authenticating...</p>
      )}
    </div>
  );
}

export default function OAuthCallbackPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-black text-white">Loading...</div>}>
      <OAuthCallbackContent />
    </Suspense>
  );
}
