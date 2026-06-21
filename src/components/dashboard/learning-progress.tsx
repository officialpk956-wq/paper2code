'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';

const TRACKS = [
  { label: 'Transformers',       progress: 34, color: '#8B5CF6', modules: 12, done: 4,  href: '/learn/deep-learning' },
  { label: 'Reinforcement Learn', progress: 12, color: '#06B6D4', modules: 9,  done: 1,  href: '/learn/reinforcement-learning' },
  { label: 'Computer Vision',    progress: 60, color: '#10B981', modules: 8,  done: 5,  href: '/learn/computer-vision' },
  { label: 'System Design',      progress: 22, color: '#F59E0B', modules: 15, done: 3,  href: '/system-design' },
];

function CircleProgress({ value, color, size = 52 }: { value: number; color: string; size?: number }) {
  const r = (size / 2) - 5;
  const circ = 2 * Math.PI * r;
  const dash = circ * (value / 100);
  return (
    <svg width={size} height={size} className="flex-shrink-0" aria-hidden="true">
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3.5" />
      <motion.circle
        cx={size/2} cy={size/2} r={r}
        fill="none" stroke={color} strokeWidth="3.5"
        strokeLinecap="round"
        strokeDasharray={circ}
        transform={`rotate(-90 ${size/2} ${size/2})`}
        initial={{ strokeDashoffset: circ }}
        animate={{ strokeDashoffset: circ - dash }}
        transition={{ duration: 1, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
      />
      <text x={size/2} y={size/2 + 4} textAnchor="middle" fontSize="9" fontWeight="700"
        fill={color} fontFamily="sans-serif">{value}%</text>
    </svg>
  );
}

export function LearningProgress() {
  return (
    <div className="rounded-xl overflow-hidden" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <div className="px-4 pt-4 pb-2">
        <div className="text-[13px] font-bold text-white">Learning Progress</div>
        <div className="text-[11px] text-slate-600 mt-0.5">Active learning tracks</div>
      </div>

      <div className="px-4 pb-4 space-y-3">
        {TRACKS.map((track, i) => (
          <motion.div
            key={track.label}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08, duration: 0.4 }}
          >
            <Link
              href={track.href}
              className="flex items-center gap-3 p-2.5 rounded-lg transition-all duration-200 block"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid transparent' }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = `${track.color}0a`;
                (e.currentTarget as HTMLElement).style.borderColor = `${track.color}25`;
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)';
                (e.currentTarget as HTMLElement).style.borderColor = 'transparent';
              }}
            >
              <CircleProgress value={track.progress} color={track.color} />
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-semibold text-white truncate">{track.label}</div>
                <div className="text-[11px] text-slate-600 mt-0.5">{track.done}/{track.modules} modules complete</div>
                <div className="mt-1.5 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
                  <motion.div
                    className="h-full rounded-full"
                    style={{ background: track.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${track.progress}%` }}
                    transition={{ duration: 0.9, delay: 0.4 + i * 0.1, ease: [0.22, 1, 0.36, 1] }}
                  />
                </div>
              </div>
            </Link>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
