'use client';

import { useState, Suspense } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';

function ResetPasswordForm() {
  const searchParams = useSearchParams();
  const token = searchParams.get('token') || '';

  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!token) {
      setError('Missing reset token in URL.');
      return;
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setIsLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

      const res = await fetch(`${apiUrl}/api/auth/reset-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token, new_password: password }),
      });

      if (res.ok) {
        setSuccess(true);
      } else if (res.status === 400 || res.status === 404) {
        setError('This reset link has expired or is invalid.');
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  if (success) {
    return (
      <div className="text-center">
        <div className="w-12 h-12 mx-auto rounded-full bg-[rgba(16,185,129,0.1)] flex items-center justify-center mb-4">
          <svg className="w-6 h-6 text-[--status-success]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
          </svg>
        </div>
        <h2 className="text-xl font-heading font-bold text-[--color-text-primary] mb-2">Password reset!</h2>
        <p className="text-sm text-[--color-text-secondary] mb-6">You can now log in with your new password.</p>
        <Link href="/login">
          <Button variant="primary" style={{ width: '100%' }}>Log In</Button>
        </Link>
      </div>
    );
  }

  return (
    <>
      <div className="text-center mb-8">
        <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-[0_0_16px_rgba(139,92,246,0.3)] mb-4">
          <span className="text-lg font-black text-white">P</span>
        </div>
        <h1 className="text-2xl font-heading font-bold text-[--color-text-primary] mb-2">Set New Password</h1>
        <p className="text-sm text-[--color-text-tertiary]">Enter your new password below.</p>
      </div>

      {error && (
        <div className="mb-6 p-3 rounded-lg bg-[rgba(239,68,68,0.1)] border border-[rgba(239,68,68,0.2)] text-[--status-error] text-sm text-center">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <Input 
          label="New Password" 
          type="password" 
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required 
          placeholder="Min 8 characters"
          minLength={8}
        />
        
        <Input 
          label="Confirm Password" 
          type="password" 
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required 
          placeholder="Confirm your new password"
          minLength={8}
        />

        <div className="pt-2">
          <Button type="submit" variant="primary" style={{ width: '100%' }} disabled={isLoading}>
            {isLoading ? 'Resetting...' : 'Reset Password'}
          </Button>
        </div>
      </form>
    </>
  );
}

export default function ResetPasswordPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[--bg-body] p-4">
      <div className="w-full max-w-md bg-[--bg-surface] border border-[--color-border] rounded-xl p-8 shadow-[0_4px_24px_rgba(0,0,0,0.2)]">
        <Suspense fallback={<div className="text-center text-[--color-text-tertiary]">Loading...</div>}>
          <ResetPasswordForm />
        </Suspense>
      </div>
    </div>
  );
}
