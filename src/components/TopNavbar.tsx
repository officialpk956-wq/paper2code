'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X } from 'lucide-react';
import { useState } from 'react';
import { useAuthModal } from './AuthModalContext';

const NAV_LINKS = [
  { label: 'Dojo',          href: '/dojo' },
  { label: 'Papers',        href: '/papers' },
  { label: 'Learn',         href: '/learn' },
  { label: 'Architectures', href: '/architectures' },
  { label: 'System Design', href: '/system-design' },
  { label: 'Labs',          href: '/labs' },
  { label: 'Pricing',       href: '/pricing' },
] as const;

export function TopNavbar() {
  const pathname         = usePathname();
  const [open, setOpen]  = useState(false);
  const { open: openAuth, user, signOut } = useAuthModal();

  const isActive = (href: string) => pathname === href || pathname.startsWith(href + '/');

  return (
    <header
      className="sticky top-0 z-50 h-14 border-b backdrop-blur"
      style={{ background: 'rgba(10,18,13,0.95)', borderColor: '#1B2A20' }}
    >
      <div className="mx-auto flex h-full items-center justify-between px-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2">
          <span className="inline-block rounded-full" style={{ width: 10, height: 10, background: '#34D399' }} />
          <span className="text-[15px] font-bold text-white">paper2code</span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden h-full items-center md:flex">
          {NAV_LINKS.map(link => {
            const active = isActive(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={
                  'flex h-full items-center px-4 text-[13px] transition-colors ' +
                  (active
                    ? 'text-[#34D399] border-b-2 border-[#34D399] bg-[#34D399]/8'
                    : 'text-[#A3A3A3] hover:text-white')
                }
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-2">
          <div className="hidden items-center gap-2 md:flex">
            {user ? (
              <>
                <div title={user.name}
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-[#34D399] text-xs font-bold text-black">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <button type="button" onClick={signOut}
                  className="rounded-lg border border-[#223429] bg-[#16241B] px-3 py-1.5 text-xs text-[#A3A3A3] transition-colors hover:text-white hover:bg-[#1B2C21]">
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <button type="button" onClick={() => openAuth('signin')}
                  className="rounded-lg border border-[#223429] bg-[#16241B] px-3 py-1.5 text-xs text-[#FAFAFA] transition-colors hover:bg-[#1B2C21]">
                  Sign In
                </button>
                <button type="button" onClick={() => openAuth('signup')}
                  className="rounded-full bg-[#34D399] px-4 py-1.5 text-xs font-semibold text-black transition-colors hover:bg-[#4ADEA8]">
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
          <aside className="absolute left-0 top-0 flex h-full w-72 flex-col bg-[#121D16] p-4"
            style={{ borderRight: '1px solid #1B2A20' }}>
            <div className="mb-6 flex items-center justify-between">
              <Link href="/" onClick={() => setOpen(false)} className="flex items-center gap-2">
                <span className="inline-block rounded-full" style={{ width: 10, height: 10, background: '#34D399' }} />
                <span className="text-[15px] font-bold text-white">paper2code</span>
              </Link>
              <button type="button" aria-label="Close menu" onClick={() => setOpen(false)}
                className="p-1.5 text-[#A3A3A3] hover:text-white">
                <X size={20} />
              </button>
            </div>
            <nav className="flex flex-col">
              {NAV_LINKS.map(link => {
                const active = isActive(link.href);
                return (
                  <Link key={link.href} href={link.href} onClick={() => setOpen(false)}
                    className={'rounded-md px-3 py-2.5 text-sm transition-colors ' +
                      (active ? 'bg-[#34D399]/10 text-[#34D399]' : 'text-[#A3A3A3] hover:bg-white/5 hover:text-white')}>
                    {link.label}
                  </Link>
                );
              })}
            </nav>
            <div className="mt-6 flex flex-col gap-2">
              {user ? (
                <>
                  <div className="flex items-center gap-2 px-1 py-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#34D399] text-xs font-bold text-black">
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                    <span className="text-sm text-white truncate">{user.name}</span>
                  </div>
                  <button type="button" onClick={() => { setOpen(false); signOut(); }}
                    className="rounded-lg border border-[#223429] bg-[#16241B] px-3 py-2 text-xs text-[#FAFAFA]">
                    Sign Out
                  </button>
                </>
              ) : (
                <>
                  <button type="button" onClick={() => { setOpen(false); openAuth('signin'); }}
                    className="rounded-lg border border-[#223429] bg-[#16241B] px-3 py-2 text-xs text-[#FAFAFA]">
                    Sign In
                  </button>
                  <button type="button" onClick={() => { setOpen(false); openAuth('signup'); }}
                    className="rounded-full bg-[#34D399] px-4 py-2 text-xs font-semibold text-black">
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
