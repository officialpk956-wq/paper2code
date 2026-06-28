'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setIsLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

      await fetch(`${apiUrl}/api/auth/forgot-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      });

      // Always show success message regardless of response (security requirement)
      setSuccess('If that email exists, a reset link has been sent.');
      setEmail('');
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[--bg-body] p-4">
      <div className="w-full max-w-md bg-[--bg-surface] border border-[--color-border] rounded-xl p-8 shadow-[0_4px_24px_rgba(0,0,0,0.2)]">
        <div className="text-center mb-8">
          <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-[0_0_16px_rgba(139,92,246,0.3)] mb-4">
            <span className="text-lg font-black text-white">P</span>
          </div>
          <h1 className="text-2xl font-heading font-bold text-[--color-text-primary] mb-2">Reset Password</h1>
          <p className="text-sm text-[--color-text-tertiary]">Enter your email to receive a reset link.</p>
        </div>

        {error && (
          <div className="mb-6 p-3 rounded-lg bg-[rgba(239,68,68,0.1)] border border-[rgba(239,68,68,0.2)] text-[--status-error] text-sm text-center">
            {error}
          </div>
        )}
        
        {success && (
          <div className="mb-6 p-3 rounded-lg bg-[rgba(16,185,129,0.1)] border border-[rgba(16,185,129,0.2)] text-[--status-success] text-sm text-center">
            {success}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <Input 
            label="Email" 
            type="email" 
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required 
            placeholder="you@example.com"
          />

          <div className="pt-2">
            <Button type="submit" variant="primary" style={{ width: '100%' }} disabled={isLoading || !!success}>
              {isLoading ? 'Sending...' : 'Send Reset Link'}
            </Button>
          </div>
        </form>

        <p className="mt-8 text-center text-sm text-[--color-text-secondary]">
          Remember your password?{' '}
          <Link href="/login" className="font-semibold text-[--accent-primary] hover:text-[--accent-light] transition-colors">
            Log in
          </Link>
        </p>
      </div>
    </div>
  );
}
