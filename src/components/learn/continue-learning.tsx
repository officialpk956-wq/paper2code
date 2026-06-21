'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Play, Clock, ChevronRight } from 'lucide-react';
import { CURRENT_LESSON } from './learn-data';

export function ContinueLearning() {
  const lesson = CURRENT_LESSON;
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="relative rounded-2xl overflow-hidden mb-6"
      style={{
        background: 'linear-gradient(135deg, rgba(6,182,212,0.1) 0%, rgba(7,11,24,0.9) 60%, rgba(139,92,246,0.08) 100%)',
        border: '1px solid rgba(6,182,212,0.25)',
      }}
    >
      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(to right, transparent, #06B6D4, #8B5CF6, transparent)' }} />

      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 p-5">
        {/* Play button */}
        <Link href="/learn/attention"
          className="flex-none w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 group"
          style={{ background: 'rgba(6,182,212,0.15)', border: '1px solid rgba(6,182,212,0.3)' }}
          onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.25)'}
          onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.15)'}
        >
          <Play size={18} style={{ color: '#06B6D4' }} className="ml-0.5 group-hover:scale-110 transition-transform" />
        </Link>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded"
              style={{ background: `${lesson.domainColor}18`, color: lesson.domainColor }}>
              {lesson.domain}
            </span>
            <span className="text-[10px] text-slate-600">{lesson.trackTitle}</span>
          </div>
          <div className="text-[13px] font-bold text-white truncate">{lesson.lesson}</div>
          <div className="text-[11px] text-slate-500 mt-0.5 truncate">{lesson.module}</div>

          {/* Progress bar */}
          <div className="mt-2.5 flex items-center gap-3">
            <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.08)' }}>
              <motion.div
                className="h-full rounded-full"
                style={{ background: `linear-gradient(to right, ${lesson.domainColor}, #8B5CF6)` }}
                initial={{ width: 0 }}
                animate={{ width: `${lesson.progress}%` }}
                transition={{ duration: 1, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
            <span className="text-[11px] font-semibold flex-none" style={{ color: lesson.domainColor }}>{lesson.progress}%</span>
          </div>
        </div>

        {/* Right meta */}
        <div className="flex sm:flex-col items-center sm:items-end gap-3 sm:gap-1 flex-none">
          <div className="flex items-center gap-1 text-[11px] text-slate-500">
            <Clock size={11} />
            <span>{lesson.timeRemainingMinutes}m left</span>
          </div>
          <Link
            href="/learn/attention"
            className="inline-flex items-center gap-1 px-4 py-2 rounded-lg text-[12px] font-bold text-white transition-all duration-200"
            style={{ background: 'linear-gradient(135deg, #0891B2, #06B6D4)', boxShadow: '0 0 16px rgba(6,182,212,0.3)' }}
            onMouseEnter={e => (e.currentTarget as HTMLElement).style.boxShadow = '0 0 22px rgba(6,182,212,0.5)'}
            onMouseLeave={e => (e.currentTarget as HTMLElement).style.boxShadow = '0 0 16px rgba(6,182,212,0.3)'}
          >
            Resume <ChevronRight size={12} />
          </Link>
        </div>
      </div>
    </motion.div>
  );
}
