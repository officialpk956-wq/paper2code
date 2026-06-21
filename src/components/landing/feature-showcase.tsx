'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  BookOpen, Layers3, Share2,
  Code2, FileText, Map, ArrowRight,
} from 'lucide-react';

const FEATURES = [
  {
    icon: <BookOpen className="w-6 h-6" />,
    title: 'Learn',
    desc: 'Interactive visual lessons with animations, equations, and code.',
    href: '/learn',
    color: '#8B5CF6',
    glow: 'rgba(139,92,246,0.25)',
  },
  {
    icon: <Layers3 className="w-6 h-6" />,
    title: 'Architectures',
    desc: 'Explore neural network architectures with interactive visualizations.',
    href: '/architectures',
    color: '#06B6D4',
    glow: 'rgba(6,182,212,0.25)',
  },
  {
    icon: <Share2 className="w-6 h-6" />,
    title: 'System Design',
    desc: 'ML Systems, RAG, LLMOps, MLOps and production architecture deep-dives.',
    href: '/system-design',
    color: '#10B981',
    glow: 'rgba(16,185,129,0.25)',
  },

  {
    icon: <Code2 className="w-6 h-6" />,
    title: 'Practice',
    desc: 'Hands-on coding challenges across ML/DL domains.',
    href: '/problems',
    color: '#8B5CF6',
    glow: 'rgba(139,92,246,0.25)',
  },
  {
    icon: <FileText className="w-6 h-6" />,
    title: 'Research',
    desc: 'Read, visualize, and implement landmark AI research papers.',
    href: '/papers',
    color: '#EC4899',
    glow: 'rgba(236,72,153,0.25)',
  },
  {
    icon: <Map className="w-6 h-6" />,
    title: 'Roadmaps',
    desc: 'Guided learning paths from beginner to expert in every AI domain.',
    href: '/roadmaps',
    color: '#22D3EE',
    glow: 'rgba(34,211,238,0.25)',
  },

];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
};

const item = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
};

function FeatureCard({ feat }: { feat: typeof FEATURES[number] }) {
  return (
    <motion.div variants={item}>
      <Link
        href={feat.href}
        className="group relative flex flex-col h-full p-6 rounded-2xl transition-all duration-300 overflow-hidden"
        style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.07)',
        }}
        onMouseEnter={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.background = `rgba(${feat.color === '#8B5CF6' ? '139,92,246' : feat.color === '#06B6D4' ? '6,182,212' : feat.color === '#10B981' ? '16,185,129' : feat.color === '#F59E0B' ? '245,158,11' : feat.color === '#EC4899' ? '236,72,153' : feat.color === '#22D3EE' ? '34,211,238' : '255,255,255'},0.06)`;
          el.style.borderColor = `${feat.color}44`;
          el.style.boxShadow = `0 0 30px ${feat.glow}, 0 20px 60px rgba(0,0,0,0.3)`;
          el.style.transform = 'translateY(-3px)';
        }}
        onMouseLeave={e => {
          const el = e.currentTarget as HTMLElement;
          el.style.background = 'rgba(255,255,255,0.03)';
          el.style.borderColor = 'rgba(255,255,255,0.07)';
          el.style.boxShadow = '';
          el.style.transform = 'translateY(0)';
        }}
      >
        {/* Icon */}
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center mb-4 transition-all duration-300 group-hover:scale-110"
          style={{ background: `${feat.color}18`, color: feat.color }}
        >
          {feat.icon}
        </div>

        {/* Text */}
        <h3 className="text-base font-bold text-white mb-2">{feat.title}</h3>
        <p className="text-sm text-slate-400 leading-relaxed flex-1">{feat.desc}</p>

        {/* Arrow */}
        <div
          className="mt-4 flex items-center gap-1 text-xs font-semibold opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ color: feat.color }}
        >
          Explore <ArrowRight size={12} />
        </div>

        {/* Corner accent */}
        <div
          className="absolute top-0 right-0 w-20 h-20 rounded-bl-full opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ background: `radial-gradient(circle at top right, ${feat.color}15, transparent 70%)` }}
        />
      </Link>
    </motion.div>
  );
}

export function FeatureShowcase() {
  return (
    <section className="py-28 relative overflow-hidden" style={{ background: '#050816' }}>
      {/* Section glow */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-violet-500/40 to-transparent" />

      <div className="max-w-7xl mx-auto px-6 md:px-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7 }}
          className="text-center mb-16"
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-6"
            style={{ background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.3)', color: '#C4B5FD' }}
          >
            Everything in one platform
          </div>
          <h2 className="text-4xl md:text-5xl font-heading font-black text-white mb-4">
            Your complete AI learning ecosystem
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Eight integrated modules that take you from theory to deployment — structured, visual, and built for engineers.
          </p>
        </motion.div>

        {/* Cards grid */}
        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: '-60px' }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          {FEATURES.map((feat) => (
            <FeatureCard key={feat.title} feat={feat} />
          ))}
        </motion.div>
      </div>

      {/* Bottom fade */}
      <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent" />
    </section>
  );
}
