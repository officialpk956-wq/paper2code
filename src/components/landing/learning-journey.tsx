'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { BookOpen, Code2, FlaskConical, Wrench, FileText, Mic2, Rocket } from 'lucide-react';

const STEPS = [
  { icon: <BookOpen size={20} />, label: 'Learn', desc: 'Core concepts', color: '#8B5CF6' },
  { icon: <Code2 size={20} />, label: 'Practice', desc: 'Solve problems', color: '#06B6D4' },
  { icon: <FlaskConical size={20} />, label: 'Experiment', desc: 'Interactive labs', color: '#10B981' },
  { icon: <Wrench size={20} />, label: 'Build', desc: 'Real projects', color: '#F59E0B' },
  { icon: <FileText size={20} />, label: 'Research', desc: 'Read papers', color: '#EC4899' },
  { icon: <Mic2 size={20} />, label: 'Interview', desc: 'Prep & practice', color: '#22D3EE' },
  { icon: <Rocket size={20} />, label: 'Career', desc: 'Land the role', color: '#8B5CF6' },
];

export function LearningJourney() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });

  return (
    <section className="py-28 relative overflow-hidden" style={{ background: '#060912' }}>
      {/* Background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0 opacity-[0.12]"
          style={{
            backgroundImage: 'linear-gradient(rgba(139,92,246,0.15) 1px, transparent 1px), linear-gradient(90deg, rgba(139,92,246,0.15) 1px, transparent 1px)',
            backgroundSize: '80px 80px',
          }}
        />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] rounded-full opacity-20"
          style={{ background: 'radial-gradient(ellipse, rgba(139,92,246,0.3), transparent 70%)' }}
        />
      </div>

      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="text-center mb-20"
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-6"
            style={{ background: 'rgba(6,182,212,0.12)', border: '1px solid rgba(6,182,212,0.3)', color: '#67E8F9' }}
          >
            Your learning path
          </div>
          <h2 className="text-4xl md:text-5xl font-heading font-black text-white mb-4">
            The complete learning journey
          </h2>
          <p className="text-lg text-slate-400 max-w-xl mx-auto">
            A structured path from first principles to production-ready AI engineering.
          </p>
        </motion.div>

        {/* Timeline */}
        <div ref={ref} className="relative">
          {/* Connector line — desktop */}
          <div className="hidden lg:block absolute top-[52px] left-0 right-0 h-px">
            <motion.div
              className="h-full origin-left"
              style={{ background: 'linear-gradient(to right, #7C3AED, #06B6D4, #10B981, #F59E0B, #EC4899, #22D3EE, #7C3AED)' }}
              initial={{ scaleX: 0 }}
              animate={inView ? { scaleX: 1 } : { scaleX: 0 }}
              transition={{ duration: 1.4, ease: [0.22, 1, 0.36, 1], delay: 0.3 }}
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-7 gap-6 lg:gap-4">
            {STEPS.map((step, i) => (
              <motion.div
                key={step.label}
                initial={{ opacity: 0, y: 30 }}
                animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
                transition={{ duration: 0.55, delay: 0.2 + i * 0.1, ease: [0.22, 1, 0.36, 1] }}
                className="flex flex-col items-center text-center"
              >
                {/* Node */}
                <div
                  className="relative w-[104px] h-[104px] rounded-2xl flex items-center justify-center mb-4 z-10"
                  style={{
                    background: `linear-gradient(135deg, ${step.color}20, ${step.color}08)`,
                    border: `1px solid ${step.color}40`,
                    boxShadow: `0 0 20px ${step.color}20`,
                  }}
                >
                  {/* Step number */}
                  <span
                    className="absolute -top-3 -right-3 w-6 h-6 rounded-full text-[10px] font-black flex items-center justify-center"
                    style={{ background: step.color, color: '#050816' }}
                  >
                    {i + 1}
                  </span>
                  {/* Icon */}
                  <div style={{ color: step.color }}>
                    {step.icon}
                  </div>
                </div>

                <h3 className="text-sm font-bold text-white mb-1">{step.label}</h3>
                <p className="text-xs text-slate-500">{step.desc}</p>

                {/* Arrow — only for non-last items on mobile */}
                {i < STEPS.length - 1 && (
                  <div className="lg:hidden mt-4 text-slate-700 text-lg">↓</div>
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Bottom note */}
        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.8 }}
          className="text-center mt-16 text-sm text-slate-600"
        >
          Each stage is supported by interactive content, practice problems, and real-world projects.
        </motion.p>
      </div>
    </section>
  );
}
