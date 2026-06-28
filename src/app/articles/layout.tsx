import type { ReactNode } from 'react';
import Link from 'next/link';

export default function ArticlesLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-body)' }}>
      <div className="max-w-3xl mx-auto px-5 py-12">
        <div className="mb-8">
          <Link
            href="/learn"
            className="text-sm font-medium transition-colors"
            style={{ color: 'var(--accent-primary)' }}
          >
            ← Back to Learn
          </Link>
        </div>
        <article className="prose prose-invert max-w-none">
          {children}
        </article>
      </div>
    </div>
  );
}
