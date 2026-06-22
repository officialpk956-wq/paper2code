'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Users, Clock, ChevronRight, Star } from 'lucide-react';
import { TRACKS } from './learn-data';
import { DOMAINS } from './learn-data';

const DIFF_COLORS: Record<string, string> = {
  Beginner: '#10B981', Intermediate: '#F59E0B', Advanced: '#EF4444', Expert: '#A78BFA',
};

/* Mini roadmap preview: show domain dots linked */
function TrackPreview({ domainIds, color }: { domainIds: string[]; color: string }) {
  const domains = domainIds.map(id => DOMAINS.find(d => d.id === id)).filter(Boolean);
  return (
    <div className="flex items-center gap-0 mt-3 overflow-x-auto" style={{ scrollbarWidth: 'none' }}>
      {domains.map((d, i) => (
        <div key={d!.id} className="flex items-center flex-none">
          <div className="w-6 h-6 rounded-full flex items-center justify-center text-[9px] font-black"
            style={{ background: `${d!.color}20`, border: `1.5px solid ${d!.color}`, color: d!.color }}>
            {d!.label.slice(0, 2).toUpperCase()}
          </div>
          {i < domains.length - 1 && (
            <div className="w-4 h-px" style={{ background: `${color}30` }} />
          )}
        </div>
      ))}
    </div>
  );
}

export function FeaturedTracks() {
  return (
    <section className="mb-8">
      <div className="flex items-end justify-between mb-5">
        <div>
          <h2 className="text-lg font-black text-white">Featured Learning Tracks</h2>
          <p className="text-xs text-slate-500 mt-0.5">Structured paths to your dream role</p>
        </div>
        <Link href="/roadmaps" className="text-[11px] font-semibold"
          style={{ color: '#8B5CF6' }}>View all →</Link>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        {TRACKS.map((track, i) => (
          <motion.div
            key={track.id}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ delay: i * 0.07, duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          >
            <Link
              href={track.href}
              className="group flex flex-col p-5 rounded-2xl h-full transition-all duration-200 relative overflow-hidden block"
              style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
              onMouseEnter={e => {
                const el = e.currentTarget as HTMLElement;
                el.style.background = `${track.color}08`;
                el.style.borderColor = `${track.color}35`;
                el.style.boxShadow = `0 0 28px ${track.glow}`;
                el.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLElement;
                el.style.background = 'rgba(255,255,255,0.02)';
                el.style.borderColor = 'rgba(255,255,255,0.07)';
                el.style.boxShadow = 'none';
                el.style.transform = 'translateY(0)';
              }}
            >
              {/* Top accent bar */}
              <div className="absolute top-0 left-0 right-0 h-0.5 rounded-t-2xl"
                style={{ background: `linear-gradient(to right, ${track.color}, transparent)` }} />

              {/* Badge */}
              {track.badge && (
                <div className="flex items-center gap-1 mb-3">
                  <Star size={10} style={{ color: track.color }} className="fill-current" />
                  <span className="text-[10px] font-bold uppercase tracking-wider" style={{ color: track.color }}>
                    {track.badge}
                  </span>
                </div>
              )}

              {/* Title + role */}
              <div className="mb-1">
                <div className="text-[14px] font-black text-white leading-tight">{track.title}</div>
                <div className="text-[11px] font-medium mt-0.5" style={{ color: track.color }}>{track.role}</div>
              </div>

              {/* Stats row */}
              <div className="flex items-center gap-3 mt-3 mb-1 text-[11px] text-slate-500">
                <span>{track.modules} modules</span>
                <span className="text-slate-700">·</span>
                <div className="flex items-center gap-1">
                  <Clock size={10} />
                  <span>{track.estimatedHours}h</span>
                </div>
                <span className="text-slate-700">·</span>
                <span style={{ color: DIFF_COLORS[track.difficulty] }}>{track.difficulty}</span>
              </div>

              {/* Learners */}
              <div className="flex items-center gap-1 text-[10px] text-slate-600 mb-3">
                <Users size={10} />
                <span>{track.enrolled.toLocaleString()} learners</span>
              </div>

              {/* Domain preview */}
              <TrackPreview domainIds={track.domains} color={track.color} />

              {/* Progress */}
              <div className="mt-4">
                {track.progress > 0 ? (
                  <>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-[10px] text-slate-600">Your progress</span>
                      <span className="text-[10px] font-semibold" style={{ color: track.color }}>{track.progress}%</span>
                    </div>
                    <div className="h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.07)' }}>
                      <motion.div className="h-full rounded-full" style={{ background: track.color }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${track.progress}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.9, delay: 0.4 + i * 0.08 }}
                      />
                    </div>
                  </>
                ) : (
                  <div className="flex items-center justify-between mt-1">
                    <span className="text-[10px] text-slate-600">Start track</span>
                    <ChevronRight size={13} style={{ color: track.color }}
                      className="group-hover:translate-x-0.5 transition-transform" />
                  </div>
                )}
              </div>
            </Link>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
