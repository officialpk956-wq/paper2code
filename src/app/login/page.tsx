'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);

      const res = await fetch(`${apiUrl}/api/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formData.toString(),
      });

      if (res.ok) {
        const data = await res.json();
        localStorage.setItem('access_token', data.access_token);
        router.push('/dashboard');
      } else if (res.status === 401) {
        setError('Invalid email or password');
      } else if (res.status === 429) {
        setError('Too many attempts. Please wait 60 seconds.');
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleOAuth = (provider: 'google' | 'github') => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const redirectUri = window.location.origin + '/auth/callback';
    fetch(`${apiUrl}/api/auth/oauth/${provider}/authorize-url?redirect_uri=${encodeURIComponent(redirectUri)}`)
      .then(res => res.json())
      .then(data => {
        if (data.url) {
          window.location.href = data.url;
        }
      })
      .catch(() => setError(`Failed to initialize ${provider} login.`));
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[--bg-body] p-4">
      <div className="w-full max-w-md bg-[--bg-surface] border border-[--color-border] rounded-xl p-8 shadow-[0_4px_24px_rgba(0,0,0,0.2)]">
        <div className="text-center mb-8">
          <div className="w-12 h-12 mx-auto rounded-xl bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-[0_0_16px_rgba(139,92,246,0.3)] mb-4">
            <span className="text-lg font-black text-white">P</span>
          </div>
          <h1 className="text-2xl font-heading font-bold text-[--color-text-primary] mb-2">Welcome Back</h1>
          <p className="text-sm text-[--color-text-tertiary]">Log in to continue your learning journey.</p>
        </div>

        {error && (
          <div className="mb-6 p-3 rounded-lg bg-[rgba(239,68,68,0.1)] border border-[rgba(239,68,68,0.2)] text-[--status-error] text-sm text-center">
            {error}
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
          
          <div className="space-y-1">
            <Input 
              label="Password" 
              type="password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required 
              placeholder="••••••••"
            />
            <div className="flex justify-end">
              <Link href="/forgot-password" className="text-xs text-[--accent-primary] hover:text-[--accent-light] transition-colors">
                Forgot password?
              </Link>
            </div>
          </div>

          <Button type="submit" variant="primary" style={{ width: '100%' }} disabled={isLoading}>
            {isLoading ? 'Logging in...' : 'Log In'}
          </Button>
        </form>

        <div className="mt-6">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-[--color-border]"></div>
            </div>
            <div className="relative flex justify-center text-xs">
              <span className="bg-[--bg-surface] px-2 text-[--color-text-tertiary]">Or continue with</span>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-3">
            <Button variant="secondary" onClick={() => handleOAuth('google')}>
              Google
            </Button>
            <Button variant="secondary" onClick={() => handleOAuth('github')}>
              GitHub
            </Button>
          </div>
        </div>

        <p className="mt-8 text-center text-sm text-[--color-text-secondary]">
          Don't have an account?{' '}
          <Link href="/signup" className="font-semibold text-[--accent-primary] hover:text-[--accent-light] transition-colors">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
