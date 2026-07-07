'use client';

import { useState, type FormEvent } from 'react';
import Link from 'next/link';
import { apiPost } from '@/lib/api';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await apiPost('/api/auth/forgot-password', { email });
      setSent(true);
    } catch (err: unknown) {
      setError((err as Error).message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0A120D] px-4">
      <div className="w-full max-w-[420px] rounded-2xl border border-[#262626] bg-[#111111] p-8">
        <Link href="/" className="mb-6 flex items-center gap-2 text-sm text-[#525252] hover:text-white transition-colors">
          ← Back
        </Link>

        <h1 className="text-xl font-bold text-white">Reset your password</h1>
        <p className="mt-1 text-sm text-[#525252]">
          Enter your email and we&apos;ll send a reset link if your account exists.
        </p>

        {sent ? (
          <div className="mt-6 rounded-lg border border-[#4ADE80]/20 bg-[#4ADE80]/10 p-4 text-sm text-[#4ADE80]">
            Check your inbox — a reset link is on its way. It expires in 30 minutes.
          </div>
        ) : (
          <form onSubmit={submit} className="mt-6 flex flex-col gap-3">
            <input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
              className="w-full rounded-lg border border-[#262626] bg-[#0A120D] px-3 py-2.5 text-sm text-white placeholder:text-[#525252] outline-none focus:border-[#A78BFA] transition-colors"
            />
            {error && (
              <div className="rounded border border-red-500/20 bg-red-500/10 p-2 text-xs text-red-400">{error}</div>
            )}
            <button
              type="submit"
              disabled={loading}
              className="mt-2 w-full rounded-lg bg-[#A78BFA] py-2.5 text-sm font-semibold text-black transition-colors hover:bg-[#C4B5FD] disabled:opacity-50"
            >
              {loading ? 'Sending...' : 'Send reset link'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
