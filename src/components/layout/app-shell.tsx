'use client';

import { usePathname } from 'next/navigation';
import Link from 'next/link';
import { LeftRail } from './left-rail';
import { Search, Flame, Zap, Compass, FlaskConical, Hammer, LayoutDashboard } from 'lucide-react';
import { useUserStats } from '@/hooks/use-user-stats';
import { CommandPalette } from '@/components/command-palette';

interface AppShellProps {
  children: React.ReactNode;
}

const PILLARS = [
  { id: 'learn', label: 'Learn', icon: Compass, match: ['/learn', '/math'] },
  { id: 'research', label: 'Research', icon: FlaskConical, match: ['/papers', '/architectures', '/knowledge-intelligence'] },
  { id: 'build', label: 'Build', icon: Hammer, match: ['/dojo', '/problems', '/paper-to-code', '/labs', '/system-design'] },
  { id: 'workspace', label: 'Workspace', icon: LayoutDashboard, match: ['/dashboard', '/roadmaps', '/assessment', '/settings'] },
];

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();
  const stats = useUserStats();

  // Landing page gets its own full-viewport layout without the sidebar shell
  if (pathname === '/') {
    return <>{children}</>;
  }

  return (
    <div className="flex h-screen" style={{ background: '#070B18' }}>
      {/* Skip Navigation */}
      <a
        href="#main-content"
        className="sr-only focus-visible:not-sr-only focus-visible:fixed focus-visible:top-2
          focus-visible:left-1/2 focus-visible:-translate-x-1/2 focus-visible:z-[100]
          focus-visible:px-4 focus-visible:py-2 focus-visible:rounded focus-visible:text-sm
          focus-visible:font-semibold focus-visible:bg-[--accent-primary] focus-visible:text-white
          focus-visible:outline-none"
      >
        Skip to main content
      </a>

      <LeftRail />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Navigation */}
        <header
          role="banner"
          className="h-14 flex items-center gap-4 px-5 sticky top-0 z-30 flex-shrink-0"
          style={{
            background: 'rgba(7,11,24,0.95)',
            borderBottom: '1px solid rgba(255,255,255,0.06)',
            backdropFilter: 'blur(12px)',
          }}
        >
          {/* Pillar Navigation */}
          <div className="flex items-center gap-1">
            {PILLARS.map(pillar => {
              const isActive = pillar.match.some(prefix => pathname.startsWith(prefix));
              return (
                <Link
                  key={pillar.id}
                  href={pillar.match[0]}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all"
                  style={{
                    background: isActive ? 'rgba(255,255,255,0.08)' : 'transparent',
                    color: isActive ? '#fff' : '#94A3B8',
                  }}
                  onMouseEnter={e => {
                    if (!isActive) (e.currentTarget as HTMLElement).style.color = '#fff';
                  }}
                  onMouseLeave={e => {
                    if (!isActive) (e.currentTarget as HTMLElement).style.color = '#94A3B8';
                  }}
                >
                  <pillar.icon size={14} />
                  {pillar.label}
                </Link>
              );
            })}
          </div>

          <div className="flex-1" />

          {/* Search */}
          <div className="w-64">
            <button
              aria-label="Open command palette (Ctrl+K)"
              className="w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg text-xs transition-all text-left"
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.07)',
                color: '#475569',
              }}
              onClick={() => {
                window.dispatchEvent(
                  new KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true })
                );
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.borderColor = 'rgba(139,92,246,0.35)';
                (e.currentTarget as HTMLElement).style.color = '#94A3B8';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.07)';
                (e.currentTarget as HTMLElement).style.color = '#475569';
              }}
            >
              <Search size={13} className="flex-shrink-0" />
              <span className="flex-1 truncate">Search...</span>
              <span
                className="px-1 py-0.5 rounded text-[10px] font-mono flex-shrink-0"
                style={{ background: 'rgba(255,255,255,0.06)', color: '#334155' }}
              >
                ⌘K
              </span>
            </button>
          </div>

          {/* Right: streak / XP / profile */}
          <div className="flex items-center gap-1.5 flex-shrink-0">
            {/* Streak */}
            {stats.loaded && (
              <Link
                href="/dashboard"
                className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-300 hover:opacity-80"
                style={{ background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.2)', color: '#FCD34D' }}
                aria-label="View streak on dashboard"
              >
                <Flame size={13} />
                <span>{stats.streak} day streak</span>
              </Link>
            )}

            {/* XP Badge */}
            {stats.loaded && (
              <Link
                href="/dashboard"
                className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-300 hover:opacity-80"
                style={{ background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.2)', color: '#A78BFA' }}
                aria-label="View XP on dashboard"
              >
                <Zap size={13} />
                <span>Level {stats.level}</span>
                <span style={{ color: '#6D28D9' }}>·</span>
                <span>{stats.xp.toLocaleString()} XP</span>
              </Link>
            )}

            {/* Profile */}
            <Link
              href="/dashboard"
              aria-label="Profile dashboard"
              className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white transition-all"
              style={{ background: 'linear-gradient(135deg, #7C3AED, #06B6D4)' }}
              onMouseEnter={e => (e.currentTarget as HTMLElement).style.boxShadow = '0 0 12px rgba(139,92,246,0.5)'}
              onMouseLeave={e => (e.currentTarget as HTMLElement).style.boxShadow = 'none'}
            >
              U
            </Link>
          </div>
        </header>

        {/* Main Content */}
        <main id="main-content" className="flex-1 overflow-hidden" tabIndex={-1}>
          {children}
        </main>
      </div>

      <CommandPalette />
    </div>
  );
}
