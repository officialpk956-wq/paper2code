'use client';

import Link from 'next/link';
import { LEGAL } from '@/lib/legal';

const FOOTER_COLS = [
  { 
    title: 'Product', 
    links: [
      { label: 'Dojo', href: '/dojo' }, 
      { label: 'Papers', href: '/papers' }, 
      { label: 'Learn', href: '/learn' }, 
      { label: 'Labs', href: '/labs' }
    ] 
  },
  { 
    title: 'Company', 
    links: [
      { label: 'About', href: '/about' },
      { label: 'System Design', href: '/system-design' }, 
      { label: 'Architectures', href: '/architectures' }, 
      { label: 'Contact', href: `mailto:${LEGAL.contactEmail}` }
    ] 
  },
  { 
    title: 'Legal',   
    links: [
      { label: 'Privacy', href: '/privacy' }, 
      { label: 'Terms', href: '/terms' }, 
      { label: 'Security', href: '/security' }, 
      { label: 'Cookies', href: '/cookies' }
    ] 
  },
];

export function Footer() {
  return (
    <footer className="mt-auto border-t border-[#1A1A1A] bg-[#0A0A0A] px-6 md:px-12 py-16">
      <div className="mx-auto flex max-w-7xl flex-col justify-between gap-10 md:flex-row">
        <div className="max-w-sm">
          <div className="flex items-center gap-2">
            <span className="inline-block rounded-full w-2.5 h-2.5 bg-[#A78BFA]" />
            <span className="text-[15px] font-bold text-white">paper2code</span>
          </div>
          <p className="mt-2 text-sm text-[#A3A3A3]">Bridge research and practice.</p>
        </div>
        <div className="flex flex-wrap gap-12">
          {FOOTER_COLS.map(col => (
            <div key={col.title} className="flex flex-col gap-2">
              <div className="text-xs font-semibold uppercase tracking-wider text-white">{col.title}</div>
              {col.links.map(l => (
                <Link key={l.label} href={l.href} className="text-sm text-[#A3A3A3] transition-colors hover:text-white">
                  {l.label}
                </Link>
              ))}
            </div>
          ))}
        </div>
      </div>
      <div className="mx-auto mt-12 flex max-w-7xl items-center justify-between border-t border-[#1A1A1A] pt-6">
        <div className="text-xs text-[#A3A3A3]">© 2026 paper2code</div>
        <div className="text-xs text-[#A3A3A3] flex gap-4">
          <Link href="/privacy" className="hover:text-white transition-colors">Privacy</Link>
          <Link href="/terms" className="hover:text-white transition-colors">Terms</Link>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
