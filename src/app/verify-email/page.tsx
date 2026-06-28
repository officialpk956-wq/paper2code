'use client';

import { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Spinner } from '@/components/ui/Spinner';

function VerifyEmailAction() {
  const searchParams = useSearchParams();
  const token = searchParams.get('token') || '';

  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');

  useEffect(() => {
    if (!token) {
      setStatus('error');
      return;
    }

    const verifyToken = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
        const res = await fetch(`${apiUrl}/api/auth/verify-email?token=${encodeURIComponent(token)}`, {
          method: 'GET',
        });

        if (res.ok) {
          setStatus('success');
        } else {
          setStatus('error');
        }
      } catch (err) {
        setStatus('error');
      }
    };

    verifyToken();
  }, [token]);

  if (status === 'loading') {
    return (
      <div className="text-center py-8">
        <Spinner size={32} className="mx-auto mb-4" />
        <p className="text-sm text-[--color-text-secondary]">Verifying your email...</p>
      </div>
    );
  }

  if (status === 'success') {
    return (
      <div className="text-center">
        <div className="w-12 h-12 mx-auto rounded-full bg-[rgba(16,185,129,0.1)] flex items-center justify-center mb-4">
          <svg className="w-6 h-6 text-[--status-success]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
          </svg>
        </div>
        <h2 className="text-xl font-heading font-bold text-[--color-text-primary] mb-2">Email verified!</h2>
        <p className="text-sm text-[--color-text-secondary] mb-6">You can now log in to your account.</p>
        <Link href="/login">
          <Button variant="primary" style={{ width: '100%' }}>Log In</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="text-center">
      <div className="w-12 h-12 mx-auto rounded-full bg-[rgba(239,68,68,0.1)] flex items-center justify-center mb-4">
        <svg className="w-6 h-6 text-[--status-error]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
        </svg>
      </div>
      <h2 className="text-xl font-heading font-bold text-[--color-text-primary] mb-2">Verification Failed</h2>
      <p className="text-sm text-[--color-text-secondary] mb-6">This verification link has expired or was already used.</p>
      <Link href="/login">
        <Button variant="secondary" style={{ width: '100%' }}>Back to Login</Button>
      </Link>
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[--bg-body] p-4">
      <div className="w-full max-w-md bg-[--bg-surface] border border-[--color-border] rounded-xl p-8 shadow-[0_4px_24px_rgba(0,0,0,0.2)]">
        <div className="text-center mb-8">
          <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-[0_0_16px_rgba(139,92,246,0.3)] mb-4">
            <span className="text-lg font-black text-white">P</span>
          </div>
          <h1 className="text-2xl font-heading font-bold text-[--color-text-primary]">Email Verification</h1>
        </div>

        <Suspense fallback={<div className="text-center py-8"><Spinner size={32} className="mx-auto" /></div>}>
          <VerifyEmailAction />
        </Suspense>
      </div>
    </div>
  );
}
