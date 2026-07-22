'use client';

import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useState, type ReactNode } from 'react';
import { usePrefersReducedMotion } from './usePrefersReducedMotion';

interface Pair {
  label: string;
  formula: ReactNode;
  code: string;
}

const PAIRS: Pair[] = [
  {
    label: 'Scaled Dot-Product Attention',
    formula: (
      <span style={{ fontFamily: "'Times New Roman', serif", fontSize: 26, letterSpacing: '0.02em' }}>
        Attention(<i>Q</i>,<i>K</i>,<i>V</i>) = softmax
        <span style={{ display: 'inline-block', verticalAlign: 'middle', margin: '0 6px' }}>
          <span style={{ display: 'block', borderBottom: '1.5px solid currentColor', padding: '0 6px', fontSize: 20 }}>
            <i>QK</i><sup>⊤</sup>
          </span>
          <span style={{ display: 'block', padding: '0 6px', fontSize: 20, textAlign: 'center' }}>√<i>d</i><sub>k</sub></span>
        </span>
        <i>V</i>
      </span>
    ),
    code: `def attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    weights = scores.softmax(dim=-1)
    return weights @ V`,
  },
  {
    label: 'Gradient Descent Update',
    formula: (
      <span style={{ fontFamily: "'Times New Roman', serif", fontSize: 28 }}>
        θ<sub>t+1</sub> = θ<sub>t</sub> − η · ∇<sub>θ</sub> ℒ(θ<sub>t</sub>)
      </span>
    ),
    code: `for step in range(num_steps):
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, params)
    for p, g in zip(params, grads):
        p.data -= lr * g`,
  },
  {
    label: 'Softmax',
    formula: (
      <span style={{ fontFamily: "'Times New Roman', serif", fontSize: 28 }}>
        σ(<i>z</i>)<sub>i</sub> ={' '}
        <span style={{ display: 'inline-block', verticalAlign: 'middle', margin: '0 6px' }}>
          <span style={{ display: 'block', borderBottom: '1.5px solid currentColor', padding: '0 8px', fontSize: 22 }}>
            <i>e</i><sup><i>z</i><sub>i</sub></sup>
          </span>
          <span style={{ display: 'block', padding: '0 8px', fontSize: 22, textAlign: 'center' }}>
            Σ<sub>j</sub> <i>e</i><sup><i>z</i><sub>j</sub></sup>
          </span>
        </span>
      </span>
    ),
    code: `def softmax(z):
    z = z - z.max(dim=-1, keepdim=True).values
    exp = torch.exp(z)
    return exp / exp.sum(dim=-1, keepdim=True)`,
  },
];

function highlight(code: string) {
  const KW = /\b(def|for|in|return|import|from|class|if|else|with|as|lambda|range)\b/g;
  const BUILTIN = /\b(torch|math|softmax|sqrt|transpose|zip|size)\b/g;
  const NUM = /\b(\d+(?:\.\d+)?)\b/g;
  const STR = /("[^"]*"|'[^']*')/g;
  const COMMENT = /(#.*$)/gm;

  return code
    .replace(COMMENT, `<span style="color:#5C6D8C">$1</span>`)
    .replace(STR, `<span style="color:#A3E635">$1</span>`)
    .replace(KW, `<span style="color:#00E5FF">$1</span>`)
    .replace(BUILTIN, `<span style="color:#7C5CFF">$1</span>`)
    .replace(NUM, `<span style="color:#66F5FF">$1</span>`);
}

export function FormulaCodeMorph() {
  const [i, setI] = useState(0);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    if (reduced) return;
    const id = window.setInterval(() => setI((v) => (v + 1) % PAIRS.length), 5000);
    return () => window.clearInterval(id);
  }, [reduced]);

  const pair = PAIRS[i];

  return (
    <div className="relative rounded-2xl border border-[#1A2744] bg-[#080C18] p-6 md:p-8">
      <div className="mb-5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-[#00E5FF]" />
          <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-[#8FA3C4]">
            Paper → Code
          </span>
        </div>
        <div className="flex gap-1.5">
          {PAIRS.map((_, k) => (
            <button
              key={k}
              type="button"
              aria-label={`Show pair ${k + 1}`}
              onClick={() => setI(k)}
              className="h-1.5 rounded-full transition-all"
              style={{ width: k === i ? 22 : 8, background: k === i ? '#00E5FF' : '#1A2744' }}
            />
          ))}
        </div>
      </div>

      <div className="mb-3 text-[13px] font-semibold text-white">{pair.label}</div>

      <div className="grid grid-cols-1 items-center gap-4 md:grid-cols-[1fr_auto_1fr]">
        {/* Formula */}
        <div className="min-h-[160px] rounded-xl border border-[#1A2744] bg-[#050810] p-6 flex items-center justify-center overflow-hidden">
          <AnimatePresence mode="wait">
            <motion.div
              key={`f-${i}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.4 }}
              className="text-white text-center"
            >
              {pair.formula}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Arrow */}
        <div className="hidden md:flex items-center justify-center px-2">
          <svg width="90" height="40" viewBox="0 0 90 40" fill="none">
            <defs>
              <linearGradient id="p2c-arrow" x1="0" y1="0" x2="90" y2="0" gradientUnits="userSpaceOnUse">
                <stop offset="0%" stopColor="#00E5FF" />
                <stop offset="100%" stopColor="#7C5CFF" />
              </linearGradient>
            </defs>
            <motion.path
              d="M 4 20 C 30 20, 55 20, 78 20"
              stroke="url(#p2c-arrow)"
              strokeWidth="2"
              strokeLinecap="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.9, ease: 'easeInOut' }}
              key={`arr-${i}`}
            />
            <motion.path
              d="M 72 12 L 84 20 L 72 28"
              stroke="url(#p2c-arrow)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7, duration: 0.3 }}
              key={`ah-${i}`}
            />
          </svg>
        </div>
        <div className="md:hidden flex justify-center text-[#00E5FF] text-2xl">↓</div>

        {/* Code */}
        <div className="min-h-[160px] rounded-xl border border-[#1A2744] bg-[#050810] overflow-hidden">
          <div className="flex items-center gap-1.5 border-b border-[#1A2744] px-3 py-2">
            <span className="h-2 w-2 rounded-full bg-[#F87171]" />
            <span className="h-2 w-2 rounded-full bg-[#FACC15]" />
            <span className="h-2 w-2 rounded-full bg-[#4ADE80]" />
            <span className="ml-2 text-[10px] text-[#5C6D8C] font-mono">attention.py</span>
          </div>
          <AnimatePresence mode="wait">
            <motion.pre
              key={`c-${i}`}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.4 }}
              className="p-4 text-[12.5px] leading-relaxed font-mono text-[#D6DCF0] overflow-auto"
              dangerouslySetInnerHTML={{ __html: highlight(pair.code) }}
            />
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
