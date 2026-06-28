'use client';

import Link from 'next/link';

const LINKS = {
  Platform: [
    { label: 'Learn', href: '/learn' },
    { label: 'Architectures', href: '/architectures' },
    { label: 'Practice', href: '/problems' },
    { label: 'System Design', href: '/system-design' },
  ],
  Research: [
    { label: 'Papers', href: '/papers' },
    { label: 'Paper to Code', href: '/paper-to-code' },
  ],
  Career: [
    { label: 'Roadmaps', href: '/roadmaps' },
    { label: 'Dashboard', href: '/dashboard' },
  ],
  Legal: [
    { label: 'Privacy Policy', href: '/privacy' },
    { label: 'Terms of Service', href: '/terms' },
  ],
};

export function LandingFooter() {
  return (
    <footer
      className="py-16 relative"
      style={{ background: '#050816', borderTop: '1px solid rgba(255,255,255,0.06)' }}
    >
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-10 mb-12">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-cyan-500 flex items-center justify-center"
                style={{ boxShadow: '0 0 14px rgba(139,92,246,0.4)' }}>
                <span className="text-xs font-black text-white">P</span>
              </div>
              <span className="text-sm font-bold text-white">Paper2Code</span>
            </div>
            <p className="text-sm text-slate-500 leading-relaxed max-w-xs">
              The complete visual learning ecosystem for Data Science, Machine Learning, and AI Engineering.
            </p>
          </div>

          {/* Links */}
          {Object.entries(LINKS).map(([group, items]) => (
            <div key={group}>
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-4">{group}</h4>
              <ul className="space-y-3">
                {items.map(({ label, href }) => (
                  <li key={label}>
                    <Link
                      href={href}
                      className="text-sm text-slate-400 hover:text-white transition-colors"
                    >
                      {label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div
          className="flex flex-col md:flex-row items-center justify-between gap-4 pt-8"
          style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
        >
          <p className="text-xs text-slate-600">
            © 2025 Paper2Code. Built for AI engineers.
          </p>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-xs text-slate-600">All systems operational</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
