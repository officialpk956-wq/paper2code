import type { Metadata } from 'next';
import { Suspense } from 'react';
import { VerifyEmail } from './VerifyEmail';

export const metadata: Metadata = { title: 'Verify email — paper2code' };

export default function EmailVerifiedPage() {
  return <Suspense fallback={<p role="status">Verifying your email…</p>}><VerifyEmail /></Suspense>;
}
