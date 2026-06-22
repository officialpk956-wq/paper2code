'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

export function FinalCTA() {
  return (
    <section className="relative py-36 overflow-hidden" style={{ background: '#050816' }}>
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-violet-500/40 to-transparent" />

      {/* Portal / vortex background */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        {/* Outer rings */}
        {[600, 500, 400, 320, 240, 160, 80].map((size, i) => (
          <motion.div
            key={size}
            className="absolute rounded-full border"
            style={{
              width: size,
              height: size,
              borderColor: i % 2 === 0
                ? `rgba(139,92,246,${0.04 + i * 0.01})`
                : `rgba(6,182,212,${0.03 + i * 0.01})`,
            }}
            animate={{ scale: [1, 1.04, 1], opacity: [0.6, 1, 0.6] }}
            transition={{ duration: 4 + i * 0.8, repeat: Infinity, ease: 'easeInOut', delay: i * 0.3 }}
          />
        ))}

        {/* Center glow */}
        <div
          className="absolute w-64 h-64 rounded-full"
          style={{ background: 'radial-gradient(circle, rgba(139,92,246,0.25) 0%, rgba(6,182,212,0.1) 40%, transparent 70%)', filter: 'blur(20px)' }}
        />
      </div>

      {/* Gradient top/bottom fades */}
      <div className="absolute inset-x-0 top-0 h-32 pointer-events-none"
        style={{ background: 'linear-gradient(to bottom, #050816, transparent)' }} />
      <div className="absolute inset-x-0 bottom-0 h-32 pointer-events-none"
        style={{ background: 'linear-gradient(to top, #050816, transparent)' }} />

      {/* Content */}
      <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-8"
            style={{ background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.3)', color: '#C4B5FD' }}
          >
            Begin your journey
          </div>

          <h2 className="text-5xl md:text-6xl lg:text-7xl font-heading font-black text-white mb-8 leading-tight">
            Start Your AI Journey
          </h2>

          <div className="space-y-2 mb-10">
            {['Learn.', 'Practice.', 'Build.', 'Research.', 'Deploy.'].map((word, i) => (
              <motion.p
                key={word}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 + i * 0.08 }}
                className="text-xl font-semibold"
                style={{
                  background: `linear-gradient(135deg, rgba(167,139,250,${1 - i * 0.12}), rgba(34,211,238,${0.6 + i * 0.08}))`,
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                {word}
              </motion.p>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.6 }}
          >
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-3 px-10 py-5 rounded-2xl text-base font-bold text-white transition-all duration-300 group"
              style={{
                background: 'linear-gradient(135deg, #7C3AED 0%, #8B5CF6 50%, #06B6D4 100%)',
                boxShadow: '0 0 50px rgba(139,92,246,0.5), 0 8px 32px rgba(0,0,0,0.5)',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.boxShadow = '0 0 70px rgba(139,92,246,0.7), 0 8px 40px rgba(0,0,0,0.5)';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.boxShadow = '0 0 50px rgba(139,92,246,0.5), 0 8px 32px rgba(0,0,0,0.5)';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
              }}
            >
              Enter Paper2Code
              <ArrowRight size={18} className="group-hover:translate-x-1 transition-transform" />
            </Link>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
