'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

const NAV_LINKS = [
  { label: 'Learn', href: '/learn' },
  { label: 'Practice', href: '/problems' },
  { label: 'Research', href: '/papers' },
  { label: 'System Design', href: '/system-design' },
  { label: 'Roadmaps', href: '/roadmaps' },
];

export function LandingNav() {
  const [scrolled, setScrolled] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    
    // Check if user is logged in
    const token = localStorage.getItem("access_token");
    setIsLoggedIn(!!token);
    
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 md:px-10 h-16"
      style={{
        background: scrolled
          ? 'rgba(5, 8, 22, 0.85)'
          : 'transparent',
        backdropFilter: scrolled ? 'blur(20px)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(255,255,255,0.06)' : 'none',
        transition: 'background 0.3s ease, backdrop-filter 0.3s ease, border-bottom 0.3s ease',
      }}
    >
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2 group">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center shadow-[0_0_16px_rgba(139,92,246,0.5)]">
          <span className="text-xs font-black text-white">P</span>
        </div>
        <span className="text-sm font-bold tracking-tight text-white group-hover:text-violet-300 transition-colors">
          Paper2Code
        </span>
      </Link>

      {/* Nav links */}
      <div className="hidden md:flex items-center gap-1">
        {NAV_LINKS.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className="px-4 py-2 text-sm text-slate-400 hover:text-white rounded-lg hover:bg-white/5 transition-all duration-150"
          >
            {link.label}
          </Link>
        ))}
      </div>

      {/* CTA */}
      <div className="hidden md:flex items-center gap-3">
        {!isLoggedIn ? (
          <>
            <Link
              href="/login"
              className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white transition-colors"
            >
              Log In
            </Link>
            <Link
              href="/signup"
              className="inline-flex items-center gap-2 px-5 py-2 rounded-full text-sm font-semibold text-white
                bg-gradient-to-r from-violet-600 to-cyan-500
                hover:from-violet-500 hover:to-cyan-400
                shadow-[0_0_20px_rgba(139,92,246,0.3)]
                hover:shadow-[0_0_28px_rgba(139,92,246,0.5)]
                transition-all duration-200"
            >
              Start Learning
              <ArrowRight size={14} />
            </Link>
          </>
        ) : (
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 px-5 py-2 rounded-full text-sm font-semibold text-white
              bg-gradient-to-r from-violet-600 to-cyan-500
              hover:from-violet-500 hover:to-cyan-400
              shadow-[0_0_20px_rgba(139,92,246,0.3)]
              hover:shadow-[0_0_28px_rgba(139,92,246,0.5)]
              transition-all duration-200"
          >
            Dashboard
            <ArrowRight size={14} />
          </Link>
        )}
      </div>

      {/* Mobile menu button */}
      <button className="md:hidden p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/5 transition-colors">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <rect x="2" y="4" width="14" height="1.5" rx="0.75" fill="currentColor" />
          <rect x="2" y="8.25" width="10" height="1.5" rx="0.75" fill="currentColor" />
          <rect x="2" y="12.5" width="14" height="1.5" rx="0.75" fill="currentColor" />
        </svg>
      </button>
    </motion.nav>
  );
}
