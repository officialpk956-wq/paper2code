'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Sparkles, Clock, FlaskConical, BookOpen, FileText, ArrowRight } from 'lucide-react';
import { RECOMMENDATIONS } from './learn-data';

const DIFF_COLORS: Record<string, string> = {
  Beginner: '#10B981', Intermediate: '#F59E0B', Advanced: '#EF4444',
};
const TYPE_ICONS: Record<string, React.ElementType> = {
  lesson: BookOpen, track: Sparkles, paper: FileText, lab: FlaskConical,
};
const TYPE_COLORS: Record<string, string> = {
  lesson: '#8B5CF6', track: '#A78BFA', paper: '#34D399', lab: '#06B6D4',
};

/* Confidence arc indicator */
function ConfidenceArc({ value, color }: { value: number; color: string }) {
  const r = 16, circ = 2 * Math.PI * r;
  return (
    <div className="relative flex items-center justify-center" style={{ width: 40, height: 40 }}>
      <svg width="40" height="40" className="absolute inset-0" aria-hidden="true">
        <circle cx="20" cy="20" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="2.5" />
        <motion.circle cx="20" cy="20" r={r} fill="none"
          stroke={color} strokeWidth="2.5"
          strokeLinecap="round"
          strokeDasharray={circ}
          transform="rotate(-90 20 20)"
          initial={{ strokeDashoffset: circ }}
          whileInView={{ strokeDashoffset: circ - (circ * value / 100) }}
          viewport={{ once: true }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
      </svg>
      <span className="text-[9px] font-black" style={{ color }}>{value}%</span>
    </div>
  );
}

export function Recommended() {
  return (
    <section className="mb-8">
      <div className="flex items-end justify-between mb-5">
        <div>
          <h2 className="text-lg font-black text-white flex items-center gap-2">
            <Sparkles size={16} style={{ color: '#A78BFA' }} />
            Recommended for You
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">AI-powered · based on your progress and interests</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
        {RECOMMENDATIONS.map((rec, i) => {
          const Icon = TYPE_ICONS[rec.type];
          const typeColor = TYPE_COLORS[rec.type];
          return (
            <motion.div
              key={rec.id}
              initial={{ opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.08, duration: 0.4 }}
            >
              <Link
                href={rec.href}
                className="group flex items-start gap-3 p-4 rounded-2xl transition-all duration-200 block h-full"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.background = `${typeColor}08`;
                  (e.currentTarget as HTMLElement).style.borderColor = `${typeColor}30`;
                  (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.07)';
                  (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
                }}
              >
                {/* Confidence arc */}
                <ConfidenceArc value={rec.confidence} color={typeColor} />

                <div className="flex-1 min-w-0">
                  {/* Type + domain */}
                  <div className="flex items-center gap-1.5 mb-1">
                    <div className="w-5 h-5 rounded flex items-center justify-center"
                      style={{ background: `${typeColor}18`, color: typeColor }}>
                      <Icon size={10} />
                    </div>
                    <span className="text-[10px] font-semibold capitalize" style={{ color: typeColor }}>{rec.type}</span>
                    <span className="text-slate-700">·</span>
                    <span className="text-[10px]" style={{ color: rec.domainColor }}>{rec.domain}</span>
                  </div>

                  <div className="text-[13px] font-semibold text-white leading-snug mb-1">{rec.title}</div>
                  <div className="text-[11px] text-slate-600 leading-snug mb-2">{rec.reason}</div>

                  <div className="flex items-center gap-3 text-[10px]">
                    <div className="flex items-center gap-1 text-slate-600">
                      <Clock size={9} /><span>{rec.estimatedMinutes}m</span>
                    </div>
                    <span style={{ color: DIFF_COLORS[rec.difficulty] }}>{rec.difficulty}</span>
                  </div>
                </div>

                <ArrowRight size={13} style={{ color: '#334155' }}
                  className="flex-none mt-1 group-hover:translate-x-0.5 transition-transform" />
              </Link>
            </motion.div>
          );
        })}
      </div>
    </section>
  );
}
