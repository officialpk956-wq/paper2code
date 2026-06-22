'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard, BookOpen, BarChart2, Brain, Layers3,
  MessageSquare, Zap, Eye, Network, Server, FolderOpen,
  Code2, Trophy, Microscope, RotateCcw,
  Map, TrendingUp, ChevronLeft, ChevronRight, Upload,
} from 'lucide-react';

interface NavItem {
  icon: React.ReactNode;
  label: string;
  href: string;
}
interface NavSection {
  title: string;
  items: NavItem[];
}

const NAV: NavSection[] = [
  {
    title: 'HOME',
    items: [
      { icon: <LayoutDashboard size={16} />, label: 'Dashboard', href: '/dashboard' },
    ],
  },
  {
    title: 'LEARN',
    items: [
      { icon: <BookOpen size={16} />, label: 'Foundations', href: '/learn' },
      { icon: <BarChart2 size={16} />, label: 'Statistics', href: '/learn/statistics' },
      { icon: <Brain size={16} />, label: 'Machine Learning', href: '/learn/machine-learning' },
      { icon: <Layers3 size={16} />, label: 'Deep Learning', href: '/learn/deep-learning' },
      { icon: <MessageSquare size={16} />, label: 'NLP', href: '/learn/nlp' },
      { icon: <Zap size={16} />, label: 'LLMs', href: '/learn/llms' },
      { icon: <Eye size={16} />, label: 'Computer Vision', href: '/learn/computer-vision' },
    ],
  },
  {
    title: 'EXPLORE',
    items: [
      { icon: <Network size={16} />, label: 'Architecture Explorer', href: '/architectures' },
      { icon: <Server size={16} />, label: 'System Design', href: '/system-design' },
    ],
  },
  {
    title: 'BUILD',
    items: [
      { icon: <FolderOpen size={16} />, label: 'Projects', href: '/paper-to-code' },
    ],
  },
  {
    title: 'PRACTICE',
    items: [
      { icon: <Code2 size={16} />, label: 'Coding Problems', href: '/problems' },
      { icon: <Trophy size={16} />, label: 'Quizzes', href: '/dojo' },
    ],
  },
  {
    title: 'RESEARCH',
    items: [
      { icon: <Upload size={16} />, label: 'Upload Paper', href: '/papers/upload' },
      { icon: <Microscope size={16} />, label: 'Research Lab', href: '/papers' },
      { icon: <RotateCcw size={16} />, label: 'Reproducibility', href: '/labs' },
    ],
  },
  {
    title: 'CAREER',
    items: [
      { icon: <Map size={16} />, label: 'Roadmaps', href: '/roadmaps' },
    ],
  },
  {
    title: 'ANALYTICS',
    items: [
      { icon: <TrendingUp size={16} />, label: 'Progress', href: '/dashboard' },
    ],
  },
];

export function LeftRail() {
  const [expanded, setExpanded] = useState(true);
  const pathname = usePathname();
  const isActive = (href: string) => pathname === href || (href !== '/' && pathname.startsWith(href));

  return (
    <>
      <nav
        aria-label="Main navigation"
        className="fixed left-0 top-0 h-screen flex flex-col z-40 transition-all duration-300 ease-out"
        style={{
          width: expanded ? 272 : 60,
          background: 'rgba(7,11,24,0.98)',
          borderRight: '1px solid rgba(255,255,255,0.07)',
        }}
      >
        {/* Logo */}
        <div className="h-14 flex items-center justify-between px-4 flex-shrink-0"
          style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          {expanded ? (
            <>
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: 'linear-gradient(135deg, #7C3AED, #06B6D4)', boxShadow: '0 0 12px rgba(139,92,246,0.4)' }}>
                  <span className="text-[11px] font-black text-white">P</span>
                </div>
                <div>
                  <div className="text-[13px] font-bold text-white leading-none">Paper2Code</div>
                  <div className="text-[10px] text-slate-500 mt-0.5 leading-none">Learn. Practice. Build.</div>
                </div>
              </div>
              <button onClick={() => setExpanded(false)}
                className="w-6 h-6 rounded flex items-center justify-center text-slate-600 hover:text-slate-300 hover:bg-white/5 transition-colors"
                aria-label="Collapse sidebar">
                <ChevronLeft size={14} />
              </button>
            </>
          ) : (
            <div className="w-full flex justify-center">
              <div className="w-7 h-7 rounded-lg flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg, #7C3AED, #06B6D4)', boxShadow: '0 0 10px rgba(139,92,246,0.35)' }}>
                <span className="text-[11px] font-black text-white">P</span>
              </div>
            </div>
          )}
        </div>

        {/* Nav sections */}
        <div className="flex-1 overflow-y-auto py-3 px-2 space-y-0.5"
          style={{ scrollbarWidth: 'none' }}>
          {NAV.map((section, si) => (
            <div key={si} className={si > 0 ? 'pt-3' : ''}>
              {expanded && (
                <div className="px-3 pb-1.5">
                  <span className="text-[9px] font-bold tracking-[0.14em] text-slate-600">
                    {section.title}
                  </span>
                </div>
              )}
              {!expanded && si > 0 && (
                <div className="my-1.5 mx-3 h-px bg-white/5" />
              )}
              <div className="space-y-0.5">
                {section.items.map((item) => {
                  const active = isActive(item.href);
                  return (
                    <Link
                      key={item.label + item.href}
                      href={item.href}
                      aria-label={item.label}
                      title={!expanded ? item.label : undefined}
                      className="flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-150 group relative focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500"
                      style={{
                        background: active ? 'rgba(139,92,246,0.15)' : 'transparent',
                        color: active ? '#A78BFA' : '#64748B',
                      }}
                      onMouseEnter={e => {
                        if (!active) {
                          (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)';
                          (e.currentTarget as HTMLElement).style.color = '#94A3B8';
                        }
                      }}
                      onMouseLeave={e => {
                        if (!active) {
                          (e.currentTarget as HTMLElement).style.background = 'transparent';
                          (e.currentTarget as HTMLElement).style.color = '#64748B';
                        }
                      }}
                    >
                      {/* Active indicator */}
                      {active && (
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-r"
                          style={{ background: '#8B5CF6', boxShadow: '0 0 6px #8B5CF6' }} />
                      )}
                      <span className="flex-shrink-0">{item.icon}</span>
                      {expanded && (
                        <span className="text-[13px] font-medium truncate">{item.label}</span>
                      )}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="flex-shrink-0 px-2 pb-2"
          style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <div className="pt-2">
            {!expanded && (
              <button onClick={() => setExpanded(true)}
                className="w-full flex items-center justify-center mt-1 py-1.5 rounded-lg text-slate-600 hover:text-slate-300 hover:bg-white/4 transition-all"
                aria-label="Expand sidebar">
                <ChevronRight size={14} />
              </button>
            )}
          </div>
        </div>
      </nav>

      {/* Spacer */}
      <div className="flex-shrink-0 transition-all duration-300" style={{ width: expanded ? 272 : 60 }} />
    </>
  );
}
