'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { FileText, Sigma, Layers3, Code2, FlaskConical, Rocket, ArrowDown } from 'lucide-react';

const STAGES = [
  {
    icon: <FileText className="w-7 h-7" />,
    title: 'Research Paper',
    desc: 'Start with a landmark paper — attention mechanisms, diffusion, transformers.',
    color: '#8B5CF6',
    glow: 'rgba(139,92,246,0.3)',
  },
  {
    icon: <Sigma className="w-7 h-7" />,
    title: 'Mathematics',
    desc: 'Understand every equation — derivations, proofs, intuitions explained visually.',
    color: '#06B6D4',
    glow: 'rgba(6,182,212,0.3)',
  },
  {
    icon: <Layers3 className="w-7 h-7" />,
    title: 'Architecture',
    desc: 'Interactive diagrams showing how each component connects and flows.',
    color: '#10B981',
    glow: 'rgba(16,185,129,0.3)',
  },
  {
    icon: <Code2 className="w-7 h-7" />,
    title: 'Implementation',
    desc: 'Clean PyTorch code with line-by-line annotations and shape tracking.',
    color: '#F59E0B',
    glow: 'rgba(245,158,11,0.3)',
  },
  {
    icon: <FlaskConical className="w-7 h-7" />,
    title: 'Experiments',
    desc: 'Reproduce results, ablation studies, and hyperparameter analysis.',
    color: '#EC4899',
    glow: 'rgba(236,72,153,0.3)',
  },
  {
    icon: <Rocket className="w-7 h-7" />,
    title: 'Production',
    desc: 'Deployment, optimization, serving infrastructure, and monitoring.',
    color: '#22D3EE',
    glow: 'rgba(34,211,238,0.3)',
  },
];

export function ResearchPipeline() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });

  return (
    <section className="py-28 relative overflow-hidden" style={{ background: '#060912' }}>
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-violet-500/40 to-transparent" />

      {/* Background decoration */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div
          className="absolute left-1/2 -translate-x-1/2 top-0 w-px h-full opacity-20"
          style={{ background: 'linear-gradient(to bottom, transparent, #8B5CF6, #06B6D4, #10B981, #F59E0B, #EC4899, #22D3EE, transparent)' }}
        />
      </div>

      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">

          {/* Left: copy */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <div
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold mb-6"
              style={{ background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.3)', color: '#C4B5FD' }}
            >
              Signature feature
            </div>
            <h2 className="text-4xl md:text-5xl font-heading font-black text-white mb-6 leading-tight">
              Turn Research Papers
              <br />
              <span style={{ background: 'linear-gradient(135deg, #A78BFA, #22D3EE)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
                Into Real Systems
              </span>
            </h2>
            <p className="text-lg text-slate-400 leading-relaxed mb-8">
              Understand every equation, every architectural choice, every implementation detail —
              from the original paper to a production-ready system.
            </p>

            {/* Paper examples */}
            <div className="space-y-3">
              {['Attention Is All You Need', 'Denoising Diffusion Probabilistic Models', 'Language Models are Few-Shot Learners'].map((paper) => (
                <div
                  key={paper}
                  className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm text-slate-300"
                  style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}
                >
                  <FileText size={14} className="text-violet-400 flex-shrink-0" />
                  {paper}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Right: pipeline stages */}
          <div ref={ref} className="relative">
            <div className="space-y-4">
              {STAGES.map((stage, i) => (
                <motion.div
                  key={stage.title}
                  initial={{ opacity: 0, x: 30 }}
                  animate={inView ? { opacity: 1, x: 0 } : { opacity: 0, x: 30 }}
                  transition={{ delay: 0.1 + i * 0.12, duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
                  className="relative flex items-start gap-4"
                >
                  {/* Connector */}
                  {i < STAGES.length - 1 && (
                    <motion.div
                      className="absolute left-[23px] top-[52px] w-px h-4"
                      style={{ background: stage.color, opacity: 0.4 }}
                      initial={{ scaleY: 0 }}
                      animate={inView ? { scaleY: 1 } : { scaleY: 0 }}
                      transition={{ delay: 0.3 + i * 0.12 }}
                    />
                  )}

                  {/* Icon box */}
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
                    style={{
                      background: `${stage.color}18`,
                      border: `1px solid ${stage.color}40`,
                      color: stage.color,
                      boxShadow: `0 0 16px ${stage.color}20`,
                    }}
                  >
                    {stage.icon}
                  </div>

                  {/* Text */}
                  <div
                    className="flex-1 p-4 rounded-xl"
                    style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}
                  >
                    <h3 className="text-sm font-bold text-white mb-1">{stage.title}</h3>
                    <p className="text-xs text-slate-500 leading-relaxed">{stage.desc}</p>
                  </div>

                  {/* Arrow down */}
                  {i < STAGES.length - 1 && (
                    <div className="absolute -bottom-2.5 left-[23px] -translate-x-1/2 z-10">
                      <ArrowDown size={10} style={{ color: stage.color, opacity: 0.5 }} />
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
