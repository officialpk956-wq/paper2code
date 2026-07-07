import Link from 'next/link';
import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'Email verified — paper2code' };

export default function EmailVerifiedPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-[#0A120D] px-4">
      <div className="w-full max-w-[420px] rounded-2xl border border-[#262626] bg-[#111111] p-8 text-center">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-[#4ADE80]/10">
          <svg className="h-6 w-6 text-[#4ADE80]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h1 className="text-xl font-bold text-white">Email verified</h1>
        <p className="mt-2 text-sm text-[#525252]">
          Your account is confirmed. You can now use all features of paper2code.
        </p>
        <Link
          href="/"
          className="mt-6 inline-block rounded-lg bg-[#A78BFA] px-6 py-2.5 text-sm font-semibold text-black transition-colors hover:bg-[#C4B5FD]"
        >
          Go to paper2code →
        </Link>
      </div>
    </div>
  );
}
