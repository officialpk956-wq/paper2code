'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { FileText, BookOpen, Code2, FlaskConical, Rocket } from 'lucide-react';

const STAGES = [
  { icon: FileText,    label: 'Paper Read',         sub: '12 papers',         status: 'done',   color: '#8B5CF6', href: '/papers' },
  { icon: BookOpen,    label: 'Understood',          sub: '8 annotations',     status: 'done',   color: '#8B5CF6', href: '/papers' },
  { icon: Code2,       label: 'Implemented',         sub: '3 codebases',       status: 'active', color: '#06B6D4', href: '/paper-to-code' },
  { icon: FlaskConical,label: 'Reproduced',          sub: '1 result',          status: 'locked', color: '#475569', href: '/labs' },
  { icon: Rocket,      label: 'Productionized',      sub: '0 deployed',        status: 'locked', color: '#475569', href: '/paper-to-code' },
];

export function ResearchJourney() {
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-3 flex items-center justify-between">
        <div>
          <div className="text-[13px] font-bold text-white">Research Journey</div>
          <div className="text-[11px] text-slate-600 mt-0.5">From reading to production</div>
        </div>
        <Link href="/papers" className="text-[11px] font-semibold" style={{ color: '#8B5CF6' }}>View all →</Link>
      </div>

      <div className="px-4 pb-4">
        <div className="flex items-start">
          {STAGES.map((stage, i) => {
            const Icon = stage.icon;
            return (
              <div key={stage.label} className="flex items-start flex-1">
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1, duration: 0.4 }}
                  className="flex flex-col items-center w-full"
                >
                  <Link href={stage.href} className="flex flex-col items-center gap-1.5 group w-full">
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200"
                      style={{
                        background: stage.status === 'locked' ? 'rgba(255,255,255,0.04)' : `${stage.color}18`,
                        border: `1.5px solid ${stage.status === 'locked' ? 'rgba(255,255,255,0.1)' : stage.color}`,
                        boxShadow: stage.status === 'active' ? `0 0 14px ${stage.color}40` : 'none',
                        color: stage.status === 'locked' ? '#334155' : stage.color,
                      }}
                    >
                      <Icon size={14} />
                    </div>
                    <div className="text-center">
                      <div className="text-[11px] font-semibold" style={{ color: stage.status === 'locked' ? '#334155' : '#94A3B8' }}>{stage.label}</div>
                      <div className="text-[10px]" style={{ color: stage.status === 'locked' ? '#1E293B' : '#475569' }}>{stage.sub}</div>
                    </div>
                  </Link>
                </motion.div>

                {/* Connector line between stages */}
                {i < STAGES.length - 1 && (
                  <div className="flex-none w-6 mt-5 h-px"
                    style={{ background: i < 1 ? 'rgba(139,92,246,0.4)' : i === 2 ? 'rgba(6,182,212,0.3)' : 'rgba(255,255,255,0.07)' }} />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
