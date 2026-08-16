'use client';

import * as Sentry from '@sentry/nextjs';
import { useEffect } from 'react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Critical root layout error caught by global error boundary:', error);
    Sentry.captureException(error);
  }, [error]);

  return (
    <html lang="en" className="dark">
      <body
        style={{
          margin: 0,
          padding: '24px',
          minHeight: '100vh',
          backgroundColor: '#0A0A0A',
          color: '#EDEDED',
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          boxSizing: 'border-box',
        }}
      >
        <div
          style={{
            width: '64px',
            height: '64px',
            borderRadius: '16px',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '24px',
            color: '#EF4444',
            fontSize: '28px',
          }}
        >
          ⚠️
        </div>

        <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: '0 0 8px 0' }}>
          Application Error
        </h1>
        <p
          style={{
            color: '#A1A1AA',
            maxWidth: '440px',
            margin: '0 0 32px 0',
            fontSize: '14px',
            lineHeight: '1.6',
          }}
        >
          A critical system error prevented the application layout from rendering. Please try again.
        </p>

        <button
          onClick={() => reset()}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px 20px',
            borderRadius: '12px',
            backgroundColor: '#A78BFA',
            color: '#000000',
            fontWeight: 600,
            fontSize: '14px',
            border: 'none',
            cursor: 'pointer',
          }}
        >
          Try again
        </button>

        {error.digest && (
          <p
            style={{
              marginTop: '32px',
              fontSize: '12px',
              color: '#71717A',
              fontFamily: 'monospace',
            }}
          >
            Digest: {error.digest}
          </p>
        )}
      </body>
    </html>
  );
}
