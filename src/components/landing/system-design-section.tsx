'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

const SYSTEMS = [
  { title: 'RAG Systems', emoji: '🔍', desc: 'Retrieval-augmented generation architectures', color: '#8B5CF6' },
  { title: 'LLM Serving', emoji: '⚡', desc: 'High-throughput inference infrastructure', color: '#06B6D4' },
  { title: 'Inference Systems', emoji: '🧮', desc: 'Batching, quantization, speculative decoding', color: '#10B981' },
  { title: 'Recommendation Systems', emoji: '🎯', desc: 'Two-tower models, retrieval + ranking', color: '#F59E0B' },
  { title: 'MLOps Pipelines', emoji: '🔄', desc: 'Training, versioning, deployment automation', color: '#EC4899' },
  { title: 'Vector Databases', emoji: '🗄️', desc: 'ANN search, HNSW, IVF, PQ compression', color: '#22D3EE' },
  { title: 'Agent Systems', emoji: '🤖', desc: 'Tool use, planning, memory architectures', color: '#8B5CF6' },
  { title: 'Multi-Agent Architectures', emoji: '🕸️', desc: 'Orchestration, communication, coordination', color: '#06B6D4' },
  { title: 'Model Deployment', emoji: '🚀', desc: 'Containers, serving APIs, A/B testing', color: '#10B981' },
  { title: 'Feature Stores', emoji: '📦', desc: 'Online/offline feature pipelines at scale', color: '#F59E0B' },
  { title: 'Streaming Pipelines', emoji: '🌊', desc: 'Real-time ML with Kafka & Flink', color: '#EC4899' },
  { title: 'Embeddings at Scale', emoji: '🔢', desc: 'Serving billions of embeddings efficiently', color: '#22D3EE' },
];

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05 } },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] } },
};

export function SystemDesignSection() {
  return (
    <section className="py-28 relative overflow-hidden" style={{ background: '#050816' }}>
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent" />

      {/* Grid background */}
      <div className="absolute inset-0 pointer-events-none opacity-[0.08]"
        style={{
          backgroundImage: 'linear-gradient(rgba(6,182,212,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(6,182,212,0.3) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

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
            style={{ background: 'rgba(6,182,212,0.12)', border: '1px solid rgba(6,182,212,0.3)', color: '#67E8F9' }}
          >
            Production systems
          </div>
          <h2 className="text-4xl md:text-5xl font-heading font-black text-white mb-4">
            Master ML System Design
          </h2>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto">
            Deep dives into how top AI companies build, deploy, and scale machine learning systems.
          </p>
        </motion.div>

        <motion.div
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: '-50px' }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4"
        >
          {SYSTEMS.map((sys) => (
            <motion.div key={sys.title} variants={item}>
              <Link
                href="/system-design"
                className="group flex flex-col p-5 rounded-xl h-full transition-all duration-250"
                style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}
                onMouseEnter={e => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.borderColor = `${sys.color}40`;
                  el.style.background = `${sys.color}08`;
                  el.style.boxShadow = `0 0 24px ${sys.color}18`;
                  el.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={e => {
                  const el = e.currentTarget as HTMLElement;
                  el.style.borderColor = 'rgba(255,255,255,0.07)';
                  el.style.background = 'rgba(255,255,255,0.03)';
                  el.style.boxShadow = '';
                  el.style.transform = 'translateY(0)';
                }}
              >
                <div className="text-2xl mb-3">{sys.emoji}</div>
                <h3 className="text-sm font-bold text-white mb-1.5">{sys.title}</h3>
                <p className="text-xs text-slate-500 leading-relaxed flex-1">{sys.desc}</p>
                <div
                  className="mt-3 flex items-center gap-1 text-[11px] font-semibold opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ color: sys.color }}
                >
                  Explore <ArrowRight size={10} />
                </div>
              </Link>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
          className="text-center mt-12"
        >
          <Link
            href="/system-design"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold text-white transition-all"
            style={{
              background: 'rgba(6,182,212,0.12)',
              border: '1px solid rgba(6,182,212,0.3)',
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.2)';
              (e.currentTarget as HTMLElement).style.borderColor = 'rgba(6,182,212,0.5)';
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLElement).style.background = 'rgba(6,182,212,0.12)';
              (e.currentTarget as HTMLElement).style.borderColor = 'rgba(6,182,212,0.3)';
            }}
          >
            View all system designs
            <ArrowRight size={14} />
          </Link>
        </motion.div>
      </div>
    </section>
  );
}
