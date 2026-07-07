'use client';

import { useState, useEffect, type FormEvent, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { apiPost } from '@/lib/api';

function ResetPasswordContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const token = searchParams.get('token') ?? '';

  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [done, setDone] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!token) setError('Missing reset token. Use the link from your email.');
  }, [token]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (password !== confirm) { setError('Passwords do not match'); return; }
    if (password.length < 8) { setError('Password must be at least 8 characters'); return; }
    setError('');
    setLoading(true);
    try {
      await apiPost('/api/auth/reset-password', { token, new_password: password });
      setDone(true);
      setTimeout(() => router.push('/'), 3000);
    } catch (err: unknown) {
      setError((err as Error).message || 'Reset failed. The link may have expired.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0A120D] px-4">
      <div className="w-full max-w-[420px] rounded-2xl border border-[#262626] bg-[#111111] p-8">
        <h1 className="text-xl font-bold text-white">Choose a new password</h1>
        <p className="mt-1 text-sm text-[#525252]">At least 8 characters.</p>

        {done ? (
          <div className="mt-6 rounded-lg border border-[#4ADE80]/20 bg-[#4ADE80]/10 p-4 text-sm text-[#4ADE80]">
            Password updated. Redirecting you home…
          </div>
        ) : (
          <form onSubmit={submit} className="mt-6 flex flex-col gap-3">
            <input
              type="password"
              placeholder="New password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              minLength={8}
              required
              autoComplete="new-password"
              className="w-full rounded-lg border border-[#262626] bg-[#0A120D] px-3 py-2.5 text-sm text-white placeholder:text-[#525252] outline-none focus:border-[#A78BFA] transition-colors"
            />
            <input
              type="password"
              placeholder="Confirm password"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              minLength={8}
              required
              autoComplete="new-password"
              className="w-full rounded-lg border border-[#262626] bg-[#0A120D] px-3 py-2.5 text-sm text-white placeholder:text-[#525252] outline-none focus:border-[#A78BFA] transition-colors"
            />
            {error && (
              <div className="rounded border border-red-500/20 bg-red-500/10 p-2 text-xs text-red-400">{error}</div>
            )}
            <button
              type="submit"
              disabled={loading || !token}
              className="mt-2 w-full rounded-lg bg-[#A78BFA] py-2.5 text-sm font-semibold text-black transition-colors hover:bg-[#C4B5FD] disabled:opacity-50"
            >
              {loading ? 'Updating...' : 'Update password'}
            </button>
          </form>
        )}

        <Link href="/" className="mt-4 block text-center text-xs text-[#525252] hover:text-white transition-colors">
          Back to home
        </Link>
      </div>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center bg-[#0A120D] text-white">Loading…</div>}>
      <ResetPasswordContent />
    </Suspense>
  );
}
