'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Sparkles, ArrowRight, Route } from 'lucide-react';
import { useEffect, useState } from 'react';
import { hasProfile, loadProfile } from '@/lib/assessment';
import { generateYourPath } from '@/lib/assessment/path';
import { getRecommendations, CATEGORY_COLORS } from '@/lib/recommendations';
import type { Recommendation } from '@/lib/recommendations';

export function RecommendedSteps() {
  const [mounted, setMounted] = useState(false);
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [isPath, setIsPath] = useState(false);

  useEffect(() => {
    setMounted(true);
    if (hasProfile()) {
      const p = loadProfile();
      if (p) {
        setRecs(generateYourPath(p, 4));
        setIsPath(true);
      }
    } else {
      setRecs(getRecommendations(4));
    }
  }, []);

  if (!mounted) return null;

  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-3 flex items-center gap-2">
        {isPath ? (
          <Route size={14} style={{ color: '#06B6D4' }} />
        ) : (
          <Sparkles size={13} style={{ color: '#A78BFA' }} />
        )}
        <div>
          <div className="text-[13px] font-bold text-white">{isPath ? 'Your Path' : 'AI Recommendations'}</div>
          <div className="text-[11px] text-slate-600">{isPath ? 'Next steps in your roadmap' : 'Personalized next steps'}</div>
        </div>
      </div>

      <div className="px-4 pb-4 space-y-2">
        {recs.map((rec, i) => {
          const color = CATEGORY_COLORS[rec.category] || '#8B5CF6';
          return (
            <motion.div
              key={rec.href + i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.09, duration: 0.4 }}
            >
              <Link
                href={rec.href}
                className="group flex items-start gap-3 p-3 rounded-xl transition-all duration-200 block"
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.background = `${color}08`;
                  (e.currentTarget as HTMLElement).style.borderColor = `${color}28`;
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
                }}
              >
                {/* Priority badge */}
                <div className="flex-shrink-0 mt-0.5">
                  <span className="text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full"
                    style={{ background: `${color}18`, color: color }}>
                    {rec.category.replace('_', ' ')}
                  </span>
                </div>

                <div className="flex-1 min-w-0">
                  <div className="text-[12px] font-semibold text-white leading-snug">{rec.title}</div>
                  <div className="text-[11px] text-slate-600 mt-0.5 leading-snug">{rec.reason}</div>
                </div>

                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className="text-[10px] font-bold" style={{ color: '#F59E0B' }}>
                    {rec.score ? `+${rec.score}` : ''}
                  </span>
                  <ArrowRight size={11} style={{ color: '#334155' }}
                    className="group-hover:translate-x-0.5 transition-transform" />
                </div>
              </Link>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
