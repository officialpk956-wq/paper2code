'use client';

import { motion, useInView, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useEffect, useRef } from 'react';

import type { RealStats } from '@/components/landing/hero-section';

function Counter({ value, suffix }: { value: number; suffix: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const count = useMotionValue(0);
  const springCount = useSpring(count, { stiffness: 60, damping: 20, restDelta: 0.5 });
  const display = useTransform(springCount, (v) => Math.round(v).toLocaleString());

  useEffect(() => {
    if (inView) count.set(value);
  }, [inView, count, value]);

  return (
    <span ref={ref}>
      <motion.span>{display}</motion.span>
      {suffix}
    </span>
  );
}

export function PlatformStats({ stats }: { stats: RealStats }) {
  const STATS = [
    { value: stats.topicCount, suffix: '', label: 'Topics', sub: 'Deep technical chapters', color: '#8B5CF6' },
    { value: stats.paperCount, suffix: '', label: 'Research Papers', sub: 'With implementations', color: '#EC4899' },
    { value: stats.archCount, suffix: '', label: 'Architectures', sub: 'Interactive visualizations', color: '#06B6D4' },
    { value: stats.problemCount, suffix: '', label: 'Practice Problems', sub: 'Across all ML domains', color: '#F59E0B' },
    { value: stats.sysDesignCount, suffix: '', label: 'System Designs', sub: 'Production case studies', color: '#10B981' },
  ];

  return (
    <section className="py-28 relative overflow-hidden" style={{ background: '#050816' }}>
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />

      {/* Ambient glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[300px] rounded-full opacity-10"
          style={{ background: 'radial-gradient(ellipse, rgba(139,92,246,0.5), transparent 70%)' }}
        />
      </div>

      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="text-center mb-16"
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-6"
            style={{ background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.3)', color: '#FCD34D' }}
          >
            Platform scale
          </div>
          <h2 className="text-4xl md:text-5xl font-heading font-black text-white mb-4">
            Built for serious learners
          </h2>
          <p className="text-lg text-slate-400 max-w-xl mx-auto">
            Comprehensive coverage across every dimension of AI engineering.
          </p>
        </motion.div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
          {STATS.map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, duration: 0.6 }}
              className="flex flex-col items-center text-center p-6 rounded-2xl"
              style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
            >
              <div
                className="text-4xl md:text-5xl font-black mb-2 font-heading"
                style={{
                  background: `linear-gradient(135deg, ${stat.color}, rgba(255,255,255,0.7))`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                <Counter value={stat.value} suffix={stat.suffix} />
              </div>
              <div className="text-sm font-bold text-white mb-1">{stat.label}</div>
              <div className="text-xs text-slate-600">{stat.sub}</div>

              {/* Bottom accent */}
              <div
                className="mt-4 w-8 h-0.5 rounded-full"
                style={{ background: stat.color, opacity: 0.6 }}
              />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
