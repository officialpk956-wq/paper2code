'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface ArchDiagramProps {
  slug: string;
}

export default function ArchDiagram({ slug }: ArchDiagramProps) {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: 'spring', stiffness: 300, damping: 24 },
    },
  };

  const pathVariants = {
    hidden: { pathLength: 0, opacity: 0 },
    visible: {
      pathLength: 1,
      opacity: 1,
      transition: { duration: 0.8, ease: 'easeOut' },
    },
  };

  const drawArrow = (x1: number, y1: number, x2: number, y2: number) => {
    const pathData = `M ${x1} ${y1} L ${x2} ${y2}`;
    const markerId = `arrowhead-${x1}-${y1}-${x2}-${y2}`;
    return (
      <g key={`${x1}-${y1}-${x2}-${y2}`}>
        <defs>
          <marker
            id={markerId}
            markerWidth="6"
            markerHeight="6"
            refX="5"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L6,3 Z" fill="#525252" />
          </marker>
        </defs>
        <motion.path
          d={pathData}
          fill="none"
          stroke="#525252"
          strokeWidth="2"
          markerEnd={`url(#${markerId})`}
          variants={pathVariants}
        />
      </g>
    );
  };

  const drawSkipConnection = (x1: number, y1: number, x2: number, y2: number) => {
    const midX = (x1 + x2) / 2 + 40;
    const midY = (y1 + y2) / 2;
    const pathData = `M ${x1} ${y1} Q ${midX} ${midY} ${x2} ${y2}`;
    const markerId = `arrowhead-skip-${x1}-${y1}-${x2}-${y2}`;
    return (
      <g key={`${x1}-${y1}-${x2}-${y2}-skip`}>
        <defs>
          <marker
            id={markerId}
            markerWidth="6"
            markerHeight="6"
            refX="5"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L6,3 Z" fill="#A78BFA" />
          </marker>
        </defs>
        <motion.path
          d={pathData}
          fill="none"
          stroke="#A78BFA"
          strokeWidth="2"
          strokeDasharray="4 4"
          markerEnd={`url(#${markerId})`}
          variants={pathVariants}
        />
      </g>
    );
  };

  const renderTransformer = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(100, 100, 160, 100)}
        {drawArrow(260, 100, 320, 100)}
        {drawArrow(520, 100, 580, 100)}
        {drawArrow(680, 100, 720, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="80" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="60" y="105" fill="#A5B4FC" fontSize="13" fontWeight="bold" textAnchor="middle">Input</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="160" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="210" y="98" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Input</text>
          <text x="210" y="116" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Embedding</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="320" y="30" width="200" height="140" rx="12" fill="#1e1b4b" stroke="#A78BFA" strokeWidth="2" strokeDasharray="3 3" />
          <text x="420" y="55" fill="#C4B5FD" fontSize="11" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">N× ENCODER BLOCK</text>

          <rect x="340" y="75" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="420" y="97" fill="#A78BFA" fontSize="11" fontWeight="semibold" textAnchor="middle">Self-Attention</text>

          <rect x="340" y="120" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="420" y="142" fill="#F472B6" fontSize="11" fontWeight="semibold" textAnchor="middle">Feed-Forward (FFN)</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="580" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="630" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Linear + Softmax</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="720" y="70" width="60" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="750" y="105" fill="#34D399" fontSize="13" fontWeight="bold" textAnchor="middle">Output</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderResNet = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(90, 100, 140, 100)}
        {drawArrow(220, 100, 270, 100)}
        {drawArrow(490, 100, 540, 100)}
        {drawArrow(630, 100, 680, 100)}

        {drawSkipConnection(180, 70, 420, 70)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="70" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="55" y="105" fill="#A5B4FC" fontSize="13" fontWeight="bold" textAnchor="middle">Input</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="140" y="70" width="80" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="180" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Initial Conv</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="270" y="30" width="220" height="140" rx="12" fill="#1e293b" stroke="#60A5FA" strokeWidth="2" strokeDasharray="3 3" />
          <text x="380" y="55" fill="#93C5FD" fontSize="11" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">N× RESIDUAL BLOCK</text>

          <rect x="290" y="75" width="80" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="330" y="97" fill="#60A5FA" fontSize="11" fontWeight="semibold" textAnchor="middle">Conv 3x3</text>

          <rect x="390" y="75" width="80" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="430" y="97" fill="#60A5FA" fontSize="11" fontWeight="semibold" textAnchor="middle">Conv 3x3</text>

          <rect x="340" y="125" width="100" height="30" rx="6" fill="#1e1b4b" stroke="#A78BFA" strokeWidth="1.5" />
          <text x="390" y="144" fill="#C4B5FD" fontSize="10" fontWeight="bold" textAnchor="middle">Add (F(x) + x)</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="540" y="70" width="90" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="585" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Global Pool</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="680" y="70" width="100" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="730" y="105" fill="#34D399" fontSize="13" fontWeight="bold" textAnchor="middle">FC (Output)</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderBERT = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(120, 100, 180, 100)}
        {drawArrow(320, 100, 380, 100)}
        {drawArrow(580, 100, 640, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="60" width="100" height="80" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="70" y="95" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">[CLS] Tokens</text>
          <text x="70" y="115" fill="#6366F1" fontSize="11" textAnchor="middle">Inputs</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="180" y="60" width="140" height="80" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="250" y="90" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Token + Positional</text>
          <text x="250" y="110" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">+ Segment Embed</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="380" y="30" width="200" height="140" rx="12" fill="#311042" stroke="#D8B4FE" strokeWidth="2" strokeDasharray="3 3" />
          <text x="480" y="55" fill="#E9D5FF" fontSize="11" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">12× or 24× ENCODER</text>

          <rect x="400" y="75" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="480" y="97" fill="#D8B4FE" fontSize="11" fontWeight="semibold" textAnchor="middle">Multi-Head Attn</text>

          <rect x="400" y="120" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="480" y="142" fill="#D8B4FE" fontSize="11" fontWeight="semibold" textAnchor="middle">FFN Block</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="640" y="60" width="140" height="80" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="710" y="90" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Downstream Tasks</text>
          <text x="710" y="110" fill="#34D399" fontSize="11" textAnchor="middle">(NSP, NER, QA)</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderViT = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(100, 100, 160, 100)}
        {drawArrow(260, 100, 320, 100)}
        {drawArrow(520, 100, 580, 100)}
        {drawArrow(680, 100, 720, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="80" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="60" y="105" fill="#A5B4FC" fontSize="13" fontWeight="bold" textAnchor="middle">Image Input</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="160" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="210" y="98" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Flatten &</text>
          <text x="210" y="116" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Linear Proj.</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="320" y="30" width="200" height="140" rx="12" fill="#1e1b4b" stroke="#A78BFA" strokeWidth="2" strokeDasharray="3 3" />
          <text x="420" y="55" fill="#C4B5FD" fontSize="11" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">12× TRANSFORMER</text>

          <rect x="340" y="75" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="420" y="97" fill="#A78BFA" fontSize="11" fontWeight="semibold" textAnchor="middle">Self-Attention</text>

          <rect x="340" y="120" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="420" y="142" fill="#F472B6" fontSize="11" fontWeight="semibold" textAnchor="middle">FFN Block</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="580" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="630" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">MLP Head</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="720" y="70" width="60" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="750" y="105" fill="#34D399" fontSize="13" fontWeight="bold" textAnchor="middle">Class</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderGPT = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(100, 100, 160, 100)}
        {drawArrow(260, 100, 320, 100)}
        {drawArrow(520, 100, 580, 100)}
        {drawArrow(680, 100, 720, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="80" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="60" y="105" fill="#A5B4FC" fontSize="13" fontWeight="bold" textAnchor="middle">Tokens</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="160" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="210" y="98" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Token + Pos</text>
          <text x="210" y="116" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Embedding</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="320" y="30" width="200" height="140" rx="12" fill="#581c0c" stroke="#FB923C" strokeWidth="2" strokeDasharray="3 3" />
          <text x="420" y="55" fill="#FED7AA" fontSize="11" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">N× DECODER BLOCK</text>

          <rect x="340" y="75" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="420" y="97" fill="#FB923C" fontSize="11" fontWeight="semibold" textAnchor="middle">Masked Self-Attn</text>

          <rect x="340" y="120" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="420" y="142" fill="#FB923C" fontSize="11" fontWeight="semibold" textAnchor="middle">FFN Block</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="580" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="630" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Linear / Softmax</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="720" y="70" width="60" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="750" y="98" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Next</text>
          <text x="750" y="116" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Token</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderLLaMA = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(100, 100, 160, 100)}
        {drawArrow(260, 100, 320, 100)}
        {drawArrow(520, 100, 580, 100)}
        {drawArrow(680, 100, 720, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="80" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="60" y="105" fill="#A5B4FC" fontSize="13" fontWeight="bold" textAnchor="middle">Tokens</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="160" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="210" y="98" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">RMSNorm + RoPE</text>
          <text x="210" y="116" fill="#D1D5DB" fontSize="11" textAnchor="middle">Embed</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="320" y="30" width="200" height="140" rx="12" fill="#064e3b" stroke="#34D399" strokeWidth="2" strokeDasharray="3 3" />
          <text x="420" y="55" fill="#A7F3D0" fontSize="11" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">LLaMA BLOCK</text>

          <rect x="340" y="75" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="420" y="97" fill="#34D399" fontSize="11" fontWeight="semibold" textAnchor="middle">RoPE Self-Attn (GQA)</text>

          <rect x="340" y="120" width="160" height="35" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="420" y="142" fill="#34D399" fontSize="11" fontWeight="semibold" textAnchor="middle">SwiGLU FFN</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="580" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="630" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">RMSNorm + Linear</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="720" y="70" width="60" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="750" y="98" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Next</text>
          <text x="750" y="116" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Token</text>
        </motion.g>
      </motion.svg>
    );
  };

  switch (slug) {
    case 'transformer':
      return renderTransformer();
    case 'resnet':
      return renderResNet();
    case 'bert':
      return renderBERT();
    case 'vit':
      return renderViT();
    case 'gpt':
      return renderGPT();
    case 'llama':
      return renderLLaMA();
    default:
      return null;
  }
}
