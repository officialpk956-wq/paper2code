import Link from 'next/link';
import { Home, BookOpen, FileText, Code2 } from 'lucide-react';
import { AppShell } from '@/components/layout/app-shell';

export default function NotFound() {
  return (
    <AppShell>
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center" style={{ background: 'var(--bg-body)' }}>
        <div className="mb-8 relative">
          <div className="text-9xl font-black text-transparent bg-clip-text opacity-20" style={{ backgroundImage: 'linear-gradient(135deg, #7C3AED, #06B6D4)' }}>
            404
          </div>
          <div className="absolute inset-0 flex items-center justify-center text-4xl" aria-hidden="true">
            🔍
          </div>
        </div>

        <h1 className="text-3xl md:text-4xl font-black mb-4" style={{ color: 'var(--color-text-primary)' }}>
          Page Not Found
        </h1>
        <p className="text-lg max-w-md mb-12" style={{ color: 'var(--color-text-secondary)' }}>
          We couldn&apos;t find the paper, topic, or path you&apos;re looking for. It might have been moved or doesn&apos;t exist.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full max-w-4xl">
          <Link
            href="/"
            className="flex flex-col items-center gap-3 p-6 rounded-2xl transition-all group"
            style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <Home className="text-purple-500" size={24} />
            <span className="font-semibold group-hover:text-[--accent-primary]" style={{ color: 'var(--color-text-primary)' }}>Home</span>
          </Link>

          <Link
            href="/learn"
            className="flex flex-col items-center gap-3 p-6 rounded-2xl transition-all group"
            style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <BookOpen className="text-cyan-500" size={24} />
            <span className="font-semibold group-hover:text-[--accent-cyan]" style={{ color: 'var(--color-text-primary)' }}>Learn</span>
          </Link>

          <Link
            href="/papers"
            className="flex flex-col items-center gap-3 p-6 rounded-2xl transition-all group"
            style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <FileText className="text-emerald-500" size={24} />
            <span className="font-semibold group-hover:text-[--accent-amber]" style={{ color: 'var(--color-text-primary)' }}>Papers</span>
          </Link>

          <Link
            href="/dojo"
            className="flex flex-col items-center gap-3 p-6 rounded-2xl transition-all group"
            style={{ background: 'var(--bg-surface)', border: '1px solid rgba(255,255,255,0.06)' }}
          >
            <Code2 className="text-amber-500" size={24} />
            <span className="font-semibold group-hover:text-[--accent-amber]" style={{ color: 'var(--color-text-primary)' }}>Dojo</span>
          </Link>
        </div>
      </div>
    </AppShell>
  );
}
