'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X, ChevronDown } from 'lucide-react';
import { useState } from 'react';
import { useAuthModal } from './AuthModalContext';

type NavItem = { label: string; href: string; desc?: string };
type NavEntry = { label: string; href?: string; items?: NavItem[] };

const ACCENT = '#A78BFA';

// Grouped navigation — only routes that exist in this app (nothing 404s).
const NAV: NavEntry[] = [
  { label: 'Papers', href: '/papers' },
  { label: 'Dojo', href: '/dojo' },
  {
    label: 'Learn',
    items: [
      { label: 'Curriculum', href: '/learn', desc: 'Structured learning paths' },
      { label: 'Architectures', href: '/architectures', desc: 'Model blueprints' },
      { label: 'System Design', href: '/system-design', desc: 'ML case studies' },
      { label: 'Labs', href: '/labs', desc: 'Interactive experiments' },
    ],
  },
  {
    label: 'Tools',
    items: [
      { label: 'Model Viz', href: '/model-viz', desc: 'Visualize architectures' },
      { label: 'Extract Code', href: '/extract-code', desc: 'PDF → runnable code' },
      { label: 'Compare Architectures', href: '/architectures/compare', desc: 'Side-by-side' },
    ],
  },
];

export function TopNavbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const { open: openAuth, user, signOut } = useAuthModal();

  const isActive = (href: string) => pathname === href || pathname.startsWith(href + '/');
  const groupActive = (e: NavEntry) =>
    e.href ? isActive(e.href) : !!e.items?.some((i) => isActive(i.href));

  const triggerCls = (active: boolean) =>
    'flex h-full items-center gap-1 px-4 text-[13px] transition-colors ' +
    (active
      ? 'text-[#A78BFA] border-b-2 border-[#A78BFA] bg-[#A78BFA]/[0.08]'
      : 'text-[#A3A3A3] hover:text-white border-b-2 border-transparent');

  return (
    <header
      className="sticky top-0 z-50 h-14 border-b backdrop-blur"
      style={{ background: 'rgba(10,10,10,0.95)', borderColor: '#1A1A1A' }}
    >
      <div className="mx-auto flex h-full items-center justify-between px-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2">
          <span className="inline-block rounded-full" style={{ width: 10, height: 10, background: ACCENT }} />
          <span className="text-[15px] font-bold text-white">paper2code</span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden h-full items-center md:flex">
          {NAV.map((entry) => {
            const active = groupActive(entry);

            if (!entry.items) {
              return (
                <Link key={entry.label} href={entry.href!} className={triggerCls(active)}>
                  {entry.label}
                </Link>
              );
            }

            return (
              <div key={entry.label} className="group relative h-full">
                <button type="button" className={triggerCls(active) + ' outline-none'}>
                  {entry.label}
                  <ChevronDown size={12} className="opacity-60 transition-transform duration-200 group-hover:rotate-180" />
                </button>
                {/* Dropdown (pt-2 bridges the hover gap) */}
                <div className="invisible absolute left-0 top-full z-50 min-w-[260px] pt-2 opacity-0 transition-all duration-150 group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
                  <div className="rounded-xl border border-[#262626] bg-[#0D0D0D] p-1.5 shadow-[0_20px_50px_-12px_rgba(0,0,0,0.8)]">
                    {entry.items.map((item) => {
                      const iActive = isActive(item.href);
                      return (
                        <Link
                          key={item.href}
                          href={item.href}
                          className={
                            'block rounded-lg px-3 py-2 transition-colors ' +
                            (iActive ? 'bg-[#A78BFA]/10' : 'hover:bg-white/[0.06]')
                          }
                        >
                          <div className={'text-[13px] font-medium ' + (iActive ? 'text-[#A78BFA]' : 'text-white')}>
                            {item.label}
                          </div>
                          {item.desc && <div className="mt-0.5 text-[11px] text-[#525252]">{item.desc}</div>}
                        </Link>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-2">
          <div className="hidden items-center gap-2 md:flex">
            {user ? (
              <>
                <div title={user.name}
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-[#A78BFA] text-xs font-bold text-black">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <button type="button" onClick={signOut}
                  className="rounded-lg border border-[#262626] bg-[#111111] px-3 py-1.5 text-xs text-[#A3A3A3] transition-colors hover:text-white hover:bg-[#1A1A1A]">
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <button type="button" onClick={() => openAuth('signin')}
                  className="rounded-lg border border-[#262626] bg-[#111111] px-3 py-1.5 text-xs text-[#FAFAFA] transition-colors hover:bg-[#1A1A1A]">
                  Sign In
                </button>
                <button type="button" onClick={() => openAuth('signup')}
                  className="rounded-full bg-[#A78BFA] px-4 py-1.5 text-xs font-semibold text-black transition-colors hover:bg-[#C4B5FD]">
                  Get Started
                </button>
              </>
            )}
          </div>

          {/* Mobile menu toggle */}
          <button type="button" aria-label="Open menu" onClick={() => setOpen(true)}
            className="flex items-center justify-center p-1.5 text-[#A3A3A3] hover:text-white md:hidden">
            <Menu size={20} />
          </button>
        </div>
      </div>

      {/* Mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <aside className="absolute left-0 top-0 flex h-full w-72 flex-col overflow-y-auto bg-[#111111] p-4"
            style={{ borderRight: '1px solid #1A1A1A' }}>
            <div className="mb-6 flex items-center justify-between">
              <Link href="/" onClick={() => setOpen(false)} className="flex items-center gap-2">
                <span className="inline-block rounded-full" style={{ width: 10, height: 10, background: ACCENT }} />
                <span className="text-[15px] font-bold text-white">paper2code</span>
              </Link>
              <button type="button" aria-label="Close menu" onClick={() => setOpen(false)}
                className="p-1.5 text-[#A3A3A3] hover:text-white">
                <X size={20} />
              </button>
            </div>
            <nav className="flex flex-col gap-1">
              {NAV.map((entry) => {
                if (!entry.items) {
                  const active = isActive(entry.href!);
                  return (
                    <Link key={entry.label} href={entry.href!} onClick={() => setOpen(false)}
                      className={'rounded-md px-3 py-2.5 text-sm transition-colors ' +
                        (active ? 'bg-[#A78BFA]/10 text-[#A78BFA]' : 'text-[#A3A3A3] hover:bg-white/5 hover:text-white')}>
                      {entry.label}
                    </Link>
                  );
                }
                return (
                  <div key={entry.label} className="mt-2">
                    <div className="px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#525252]">
                      {entry.label}
                    </div>
                    {entry.items.map((item) => {
                      const active = isActive(item.href);
                      return (
                        <Link key={item.href} href={item.href} onClick={() => setOpen(false)}
                          className={'rounded-md px-3 py-2 text-sm transition-colors ' +
                            (active ? 'bg-[#A78BFA]/10 text-[#A78BFA]' : 'text-[#A3A3A3] hover:bg-white/5 hover:text-white')}>
                          {item.label}
                        </Link>
                      );
                    })}
                  </div>
                );
              })}
            </nav>
            <div className="mt-6 flex flex-col gap-2">
              {user ? (
                <>
                  <div className="flex items-center gap-2 px-1 py-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#A78BFA] text-xs font-bold text-black">
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                    <span className="text-sm text-white truncate">{user.name}</span>
                  </div>
                  <button type="button" onClick={() => { setOpen(false); signOut(); }}
                    className="rounded-lg border border-[#262626] bg-[#111111] px-3 py-2 text-xs text-[#FAFAFA]">
                    Sign Out
                  </button>
                </>
              ) : (
                <>
                  <button type="button" onClick={() => { setOpen(false); openAuth('signin'); }}
                    className="rounded-lg border border-[#262626] bg-[#111111] px-3 py-2 text-xs text-[#FAFAFA]">
                    Sign In
                  </button>
                  <button type="button" onClick={() => { setOpen(false); openAuth('signup'); }}
                    className="rounded-full bg-[#A78BFA] px-4 py-2 text-xs font-semibold text-black">
                    Get Started
                  </button>
                </>
              )}
            </div>
          </aside>
        </div>
      )}
    </header>
  );
}

export default TopNavbar;
