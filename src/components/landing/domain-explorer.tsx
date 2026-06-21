'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

const DOMAINS = [
  {
    title: 'Machine Learning',
    emoji: '🤖',
    color: '#8B5CF6',
    glow: 'rgba(139,92,246,0.2)',
    topics: ['Regression', 'Classification', 'Clustering', 'Ensembles'],
    hours: 80,
    level: 'Beginner → Advanced',
    modules: 24,
    href: '/learn',
  },
  {
    title: 'Deep Learning',
    emoji: '🧠',
    color: '#06B6D4',
    glow: 'rgba(6,182,212,0.2)',
    topics: ['CNNs', 'RNNs', 'Attention', 'Backprop'],
    hours: 120,
    level: 'Intermediate',
    modules: 38,
    href: '/architectures',
  },
  {
    title: 'Computer Vision',
    emoji: '👁️',
    color: '#10B981',
    glow: 'rgba(16,185,129,0.2)',
    topics: ['Object Detection', 'Segmentation', 'ViT', 'Diffusion'],
    hours: 96,
    level: 'Intermediate',
    modules: 31,
    href: '/papers',
  },
  {
    title: 'NLP & Language',
    emoji: '📝',
    color: '#F59E0B',
    glow: 'rgba(245,158,11,0.2)',
    topics: ['Tokenization', 'Embeddings', 'BERT', 'GPT'],
    hours: 110,
    level: 'Intermediate → Expert',
    modules: 35,
    href: '/learn',
  },
  {
    title: 'LLMs',
    emoji: '⚡',
    color: '#EC4899',
    glow: 'rgba(236,72,153,0.2)',
    topics: ['Fine-tuning', 'RLHF', 'RAG', 'Agents'],
    hours: 140,
    level: 'Advanced',
    modules: 42,
    href: '/system-design',
  },
  {
    title: 'Generative AI',
    emoji: '🎨',
    color: '#22D3EE',
    glow: 'rgba(34,211,238,0.2)',
    topics: ['GANs', 'VAEs', 'Diffusion', 'Flow'],
    hours: 88,
    level: 'Advanced',
    modules: 28,
    href: '/architectures',
  },
  {
    title: 'Data Engineering',
    emoji: '🔧',
    color: '#8B5CF6',
    glow: 'rgba(139,92,246,0.2)',
    topics: ['Pipelines', 'Feature Stores', 'Streaming', 'Warehouses'],
    hours: 72,
    level: 'Intermediate',
    modules: 22,
    href: '/system-design',
  },
  {
    title: 'MLOps',
    emoji: '🚀',
    color: '#06B6D4',
    glow: 'rgba(6,182,212,0.2)',
    topics: ['CI/CD', 'Monitoring', 'Deployment', 'Infra'],
    hours: 90,
    level: 'Advanced',
    modules: 30,
    href: '/system-design',
  },
  {
    title: 'System Design',
    emoji: '🏗️',
    color: '#10B981',
    glow: 'rgba(16,185,129,0.2)',
    topics: ['Scalability', 'Inference', 'Vector DBs', 'APIs'],
    hours: 100,
    level: 'Expert',
    modules: 40,
    href: '/system-design',
  },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06 } },
};

const item = {
  hidden: { opacity: 0, y: 28 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
};

export function DomainExplorer() {
  return (
    <section className="py-28 relative overflow-hidden" style={{ background: '#050816' }}>
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent" />

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
            style={{ background: 'rgba(16,185,129,0.12)', border: '1px solid rgba(16,185,129,0.3)', color: '#6EE7B7' }}
          >
            9 domains covered
          </div>
          <h2 className="text-4xl md:text-5xl font-heading font-black text-white mb-4">
            Explore every AI domain
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Structured courses across the entire data science and AI landscape, from foundations to cutting-edge research.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
        >
          {DOMAINS.map((domain) => (
            <motion.div key={domain.title} variants={item}>
              <Link
                href={domain.href}
                className="group relative flex flex-col p-6 rounded-2xl h-full overflow-hidden transition-all duration-300"
                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
                onMouseEnter={e => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.borderColor = `${domain.color}50`;
                  el.style.boxShadow = `0 0 40px ${domain.glow}, 0 20px 60px rgba(0,0,0,0.3)`;
                  el.style.transform = 'translateY(-4px)';
                }}
                onMouseLeave={e => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.borderColor = 'rgba(255,255,255,0.07)';
                  el.style.boxShadow = '';
                  el.style.transform = 'translateY(0)';
                }}
              >
                {/* Top accent */}
                <div
                  className="absolute top-0 left-0 right-0 h-px opacity-60 group-hover:opacity-100 transition-opacity"
                  style={{ background: `linear-gradient(to right, transparent, ${domain.color}, transparent)` }}
                />

                {/* Header */}
                <div className="flex items-start gap-4 mb-5">
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl flex-shrink-0"
                    style={{ background: `${domain.color}18` }}
                  >
                    {domain.emoji}
                  </div>
                  <div>
                    <h3 className="text-base font-bold text-white mb-0.5">{domain.title}</h3>
                    <span className="text-xs text-slate-500">{domain.level}</span>
                  </div>
                </div>

                {/* Topics */}
                <div className="flex flex-wrap gap-2 mb-5 flex-1">
                  {domain.topics.map((t) => (
                    <span
                      key={t}
                      className="px-2.5 py-1 rounded-full text-[11px] font-medium"
                      style={{ background: `${domain.color}14`, color: domain.color, border: `1px solid ${domain.color}25` }}
                    >
                      {t}
                    </span>
                  ))}
                </div>

                {/* Stats */}
                <div className="flex items-center justify-between pt-4 border-t border-white/5">
                  <div className="text-xs text-slate-500">
                    <span className="text-white font-semibold">{domain.modules}</span> modules
                  </div>
                  <div className="text-xs text-slate-500">
                    <span className="text-white font-semibold">{domain.hours}h</span> content
                  </div>
                  <div
                    className="flex items-center gap-1 text-xs font-semibold opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ color: domain.color }}
                  >
                    Start <ArrowRight size={11} />
                  </div>
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
