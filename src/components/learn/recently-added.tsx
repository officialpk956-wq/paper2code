'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Clock, FlaskConical, BookOpen, Code2, FileText } from 'lucide-react';
import { RECENT_LESSONS } from './learn-data';
import type { Lesson } from './learn-types';

const DIFF_COLORS: Record<string, string> = {
  Beginner: '#10B981', Intermediate: '#F59E0B', Advanced: '#EF4444',
};
const TYPE_ICONS: Record<string, React.ElementType> = {
  lesson: BookOpen, lab: FlaskConical, paper: FileText, project: Code2,
};
const TYPE_COLORS: Record<string, string> = {
  lesson: '#8B5CF6', lab: '#06B6D4', paper: '#34D399', project: '#F59E0B',
};

function LessonCard({ lesson, index }: { lesson: Lesson; index: number }) {
  const Icon = TYPE_ICONS[lesson.type];
  const typeColor = TYPE_COLORS[lesson.type];
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.06, duration: 0.4 }}
    >
      <Link
        href={lesson.href}
        className="group flex items-start gap-3 p-4 rounded-xl transition-all duration-200 block"
        style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLElement).style.background = 'rgba(139,92,246,0.06)';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(139,92,246,0.25)';
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
          (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
        }}
      >
        {/* Type icon */}
        <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
          style={{ background: `${typeColor}15`, color: typeColor }}>
          <Icon size={13} />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start gap-2 mb-1">
            <span className="text-[13px] font-semibold text-white leading-snug truncate">{lesson.title}</span>
            {lesson.isNew && (
              <span className="flex-none text-[9px] font-black uppercase tracking-wider px-1.5 py-0.5 rounded text-white"
                style={{ background: 'linear-gradient(135deg, #7C3AED, #06B6D4)' }}>
                NEW
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 text-[10px]">
            <span style={{ color: lesson.domainColor }}>{lesson.domain}</span>
            <span className="text-slate-700">·</span>
            <div className="flex items-center gap-1 text-slate-600">
              <Clock size={9} /><span>{lesson.durationMinutes}m</span>
            </div>
            <span className="text-slate-700">·</span>
            <span style={{ color: DIFF_COLORS[lesson.difficulty] }}>{lesson.difficulty}</span>
            <span className="text-slate-700">·</span>
            <span className="text-slate-700">
              {lesson.addedDaysAgo === 1 ? 'Yesterday' : `${lesson.addedDaysAgo}d ago`}
            </span>
          </div>
        </div>

        {/* Type badge */}
        <span className="flex-none text-[9px] font-bold uppercase tracking-wider px-2 py-1 rounded-lg capitalize"
          style={{ background: `${typeColor}15`, color: typeColor }}>
          {lesson.type}
        </span>
      </Link>
    </motion.div>
  );
}

export function RecentlyAdded() {
  return (
    <section className="mb-8">
      <div className="flex items-end justify-between mb-5">
        <div>
          <h2 className="text-lg font-black text-white">Recently Added</h2>
          <p className="text-xs text-slate-500 mt-0.5">Freshest content across all domains</p>
        </div>
        <Link href="/learn/new" className="text-[11px] font-semibold"
          style={{ color: '#8B5CF6' }}>View all →</Link>
      </div>

      <div className="space-y-2">
        {RECENT_LESSONS.map((lesson, i) => (
          <LessonCard key={lesson.id} lesson={lesson} index={i} />
        ))}
      </div>
    </section>
  );
}
