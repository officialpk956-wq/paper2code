'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { CheckCircle2, Circle, Lock } from 'lucide-react';

const ARCH_MILESTONES = [
  { label: 'LeNet',          year: '1998', status: 'done',    color: '#8B5CF6', href: '/architectures' },
  { label: 'AlexNet',        year: '2012', status: 'done',    color: '#8B5CF6', href: '/architectures' },
  { label: 'ResNet',         year: '2015', status: 'done',    color: '#8B5CF6', href: '/architectures' },
  { label: 'Transformer',    year: '2017', status: 'active',  color: '#06B6D4', href: '/architectures' },
  { label: 'BERT',           year: '2018', status: 'locked',  color: '#475569', href: '/architectures' },
  { label: 'GPT-3',          year: '2020', status: 'locked',  color: '#475569', href: '/architectures' },
  { label: 'ViT',            year: '2020', status: 'locked',  color: '#475569', href: '/architectures' },
  { label: 'Diffusion',      year: '2022', status: 'locked',  color: '#475569', href: '/architectures' },
  { label: 'LLaMA',          year: '2023', status: 'locked',  color: '#475569', href: '/architectures' },
];

export function ArchitectureJourney() {
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-3 flex items-center justify-between">
        <div>
          <div className="text-[13px] font-bold text-white">Architecture Journey</div>
          <div className="text-[11px] text-slate-600 mt-0.5">Neural architecture milestones</div>
        </div>
        <Link href="/architectures" className="text-[11px] font-semibold transition-colors" style={{ color: '#8B5CF6' }}
          onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = '#A78BFA'}
          onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = '#8B5CF6'}>
          View all →
        </Link>
      </div>

      <div className="px-4 pb-4 overflow-x-auto" style={{ scrollbarWidth: 'none' }}>
        <div className="flex items-center gap-0 min-w-max">
          {ARCH_MILESTONES.map((m, i) => (
            <div key={m.label} className="flex items-center">
              {/* Node */}
              <motion.div
                initial={{ opacity: 0, scale: 0.7 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.06, duration: 0.35 }}
              >
                <Link href={m.href} className="flex flex-col items-center gap-1.5 group px-1">
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200"
                    style={{
                      background: m.status === 'done' ? `${m.color}22` : m.status === 'active' ? `${m.color}20` : 'rgba(255,255,255,0.04)',
                      border: `1.5px solid ${m.status === 'locked' ? 'rgba(255,255,255,0.1)' : m.color}`,
                      boxShadow: m.status === 'active' ? `0 0 14px ${m.color}50` : 'none',
                    }}
                  >
                    {m.status === 'done'   && <CheckCircle2 size={14} style={{ color: m.color }} />}
                    {m.status === 'active' && <Circle       size={14} style={{ color: m.color }} className="fill-current" />}
                    {m.status === 'locked' && <Lock         size={11} style={{ color: '#334155' }} />}
                  </div>
                  <div className="text-center">
                    <div className="text-[11px] font-semibold leading-tight" style={{ color: m.status === 'locked' ? '#334155' : '#94A3B8' }}>{m.label}</div>
                    <div className="text-[9px] text-slate-700">{m.year}</div>
                  </div>
                </Link>
              </motion.div>

              {/* Connector */}
              {i < ARCH_MILESTONES.length - 1 && (
                <div className="w-8 h-px flex-shrink-0 mt-[-18px]"
                  style={{ background: i < 2 ? 'rgba(139,92,246,0.4)' : i === 3 ? 'rgba(6,182,212,0.3)' : 'rgba(255,255,255,0.07)' }} />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
