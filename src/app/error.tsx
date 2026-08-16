'use client';

import * as Sentry from '@sentry/nextjs';
import { useEffect } from 'react';
import Link from 'next/link';
import { AlertTriangle, RotateCcw, Home } from 'lucide-react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Unhandled route error caught by error boundary:', error);
    Sentry.captureException(error);
  }, [error]);

  return (
    <div className="min-h-[calc(100dvh-56px)] bg-[#0A0A0A] text-[#EDEDED] flex flex-col items-center justify-center p-6 text-center">
      <div className="w-16 h-16 rounded-2xl bg-[#EF4444]/10 border border-[#EF4444]/20 flex items-center justify-center mb-6 text-[#EF4444]">
        <AlertTriangle size={32} />
      </div>

      <h1 className="text-2xl font-bold tracking-tight mb-2">Something went wrong</h1>
      <p className="text-[#A1A1AA] max-w-md mb-8 text-sm leading-relaxed">
        An unexpected error occurred while rendering this page. You can try reloading the component or return to the homepage.
      </p>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <button
          onClick={() => reset()}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#A78BFA] text-black font-semibold text-sm hover:bg-[#9065FA] transition-colors cursor-pointer shadow-lg shadow-[#A78BFA]/10"
        >
          <RotateCcw size={16} />
          Try again
        </button>

        <Link
          href="/"
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#18181B] border border-[#27272A] text-[#EDEDED] font-medium text-sm hover:bg-[#27272A] transition-colors"
        >
          <Home size={16} />
          Back to Home
        </Link>
      </div>

      {error.digest && (
        <p className="mt-8 text-xs text-[#71717A] font-mono">
          Error ID: {error.digest}
        </p>
      )}
    </div>
  );
}
