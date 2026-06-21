'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { CheckCircle2, Circle, Lock } from 'lucide-react';

const SD_MILESTONES = [
  { label: 'Load Balancer',    status: 'done',   color: '#10B981', href: '/system-design' },
  { label: 'Caching',          status: 'done',   color: '#10B981', href: '/system-design' },
  { label: 'Databases',        status: 'active', color: '#06B6D4', href: '/system-design' },
  { label: 'Message Queue',    status: 'locked', color: '#475569', href: '/system-design' },
  { label: 'CDN',              status: 'locked', color: '#475569', href: '/system-design' },
  { label: 'Microservices',    status: 'locked', color: '#475569', href: '/system-design' },
  { label: 'ML Serving',       status: 'locked', color: '#475569', href: '/system-design' },
  { label: 'Inference Infra',  status: 'locked', color: '#475569', href: '/system-design' },
];

export function SystemDesignJourney() {
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-3 flex items-center justify-between">
        <div>
          <div className="text-[13px] font-bold text-white">System Design Journey</div>
          <div className="text-[11px] text-slate-600 mt-0.5">Production infrastructure patterns</div>
        </div>
        <Link href="/system-design" className="text-[11px] font-semibold transition-colors" style={{ color: '#10B981' }}
          onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = '#34D399'}
          onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = '#10B981'}>
          View all →
        </Link>
      </div>

      <div className="px-4 pb-4 overflow-x-auto" style={{ scrollbarWidth: 'none' }}>
        <div className="flex items-center gap-0 min-w-max">
          {SD_MILESTONES.map((m, i) => (
            <div key={m.label} className="flex items-center">
              <motion.div
                initial={{ opacity: 0, scale: 0.7 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.06, duration: 0.35 }}
              >
                <Link href={m.href} className="flex flex-col items-center gap-1.5 px-1">
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
                    <div className="text-[10px] font-semibold leading-tight w-14 text-center" style={{ color: m.status === 'locked' ? '#334155' : '#94A3B8' }}>{m.label}</div>
                  </div>
                </Link>
              </motion.div>
              {i < SD_MILESTONES.length - 1 && (
                <div className="w-6 h-px flex-shrink-0 mt-[-18px]"
                  style={{ background: i < 1 ? 'rgba(16,185,129,0.4)' : i === 2 ? 'rgba(6,182,212,0.3)' : 'rgba(255,255,255,0.07)' }} />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
