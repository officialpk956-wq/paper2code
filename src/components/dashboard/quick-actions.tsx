'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { BookOpen, Code2, Network, FileText, Zap, Trophy, Microscope, Map, Upload, Files } from 'lucide-react';

const ACTIONS = [
  { icon: BookOpen,   label: 'Continue Learning',  sub: 'Transformers track',     href: '/learn',          color: '#8B5CF6', glow: 'rgba(139,92,246,0.25)' },
  { icon: Code2,      label: 'Solve a Problem',    sub: 'Daily challenge',        href: '/problems',       color: '#06B6D4', glow: 'rgba(6,182,212,0.25)' },
  { icon: Network,    label: 'Explore Arch',       sub: 'ResNet deep-dive',       href: '/architectures',  color: '#10B981', glow: 'rgba(16,185,129,0.25)' },
  { icon: FileText,   label: 'Read a Paper',       sub: 'Attention is All You Need', href: '/papers',      color: '#F59E0B', glow: 'rgba(245,158,11,0.25)' },
  { icon: Zap,        label: 'Quick Lab',          sub: 'Forward pass lab',       href: '/labs',           color: '#EC4899', glow: 'rgba(236,72,153,0.25)' },
  { icon: Trophy,     label: 'Take a Quiz',        sub: 'Backprop basics',        href: '/dojo',           color: '#A78BFA', glow: 'rgba(167,139,250,0.25)' },
  { icon: Microscope, label: 'Research Lab',       sub: 'Reproduce a result',     href: '/paper-to-code',  color: '#34D399', glow: 'rgba(52,211,153,0.25)' },
  { icon: Map,        label: 'Roadmap',            sub: 'ML Engineer path',       href: '/roadmaps',       color: '#60A5FA', glow: 'rgba(96,165,250,0.25)' },
  { icon: Upload,     label: 'Upload Paper',       sub: 'Analyze a research paper', href: '/papers/upload', color: '#F97316', glow: 'rgba(249,115,22,0.25)' },
  { icon: Files,      label: 'My Papers',          sub: 'View uploaded papers',   href: '/papers',         color: '#22D3EE', glow: 'rgba(34,211,238,0.25)' },
];

export function QuickActionGrid() {
  return (
    <div>
      <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-3">Quick Actions</h2>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {ACTIONS.map((action, i) => {
          const Icon = action.icon;
          return (
            <motion.div
              key={action.label}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05, duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            >
              <Link
                href={action.href}
                className="flex flex-col gap-2.5 p-4 rounded-xl transition-all duration-200 group block"
                style={{
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.07)',
                }}
                onMouseEnter={e => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.background = `${action.color}0f`;
                  el.style.borderColor = `${action.color}35`;
                  el.style.boxShadow = `0 0 20px ${action.glow}`;
                  el.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={e => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.background = 'rgba(255,255,255,0.03)';
                  el.style.borderColor = 'rgba(255,255,255,0.07)';
                  el.style.boxShadow = 'none';
                  el.style.transform = 'translateY(0)';
                }}
              >
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: `${action.color}18`, color: action.color }}
                >
                  <Icon size={15} />
                </div>
                <div>
                  <div className="text-[13px] font-semibold text-white leading-tight">{action.label}</div>
                  <div className="text-[11px] text-slate-600 mt-0.5 truncate">{action.sub}</div>
                </div>
              </Link>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
