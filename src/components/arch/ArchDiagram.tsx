'use client';

import React from 'react';
import { motion } from 'framer-motion';

import { GENERIC_FLOWS, type GenBlock } from './archFlows';

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

  const renderGAN = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(110, 55, 140, 55)}
        {drawArrow(260, 55, 290, 55)}
        {drawArrow(382, 55, 428, 88)}
        {drawArrow(382, 145, 428, 112)}
        {drawArrow(560, 100, 598, 100)}

        {/* adversarial feedback loop */}
        <motion.path
          d="M 480 130 Q 320 195 200 87"
          fill="none"
          stroke="#A78BFA"
          strokeWidth="1.5"
          strokeDasharray="4 4"
          variants={pathVariants}
        />
        <text x="330" y="188" fill="#A78BFA" fontSize="9" textAnchor="middle">adversarial loss</text>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="30" width="90" height="50" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="65" y="60" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Noise z</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="140" y="25" width="120" height="60" rx="8" fill="#1e1b4b" stroke="#A78BFA" strokeWidth="2" />
          <text x="200" y="60" fill="#C4B5FD" fontSize="12" fontWeight="bold" textAnchor="middle">Generator</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="290" y="30" width="90" height="50" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="335" y="52" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Fake</text>
          <text x="335" y="68" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Image</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="290" y="120" width="90" height="50" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="335" y="142" fill="#34D399" fontSize="11" fontWeight="bold" textAnchor="middle">Real</text>
          <text x="335" y="158" fill="#34D399" fontSize="11" fontWeight="bold" textAnchor="middle">Image</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="430" y="70" width="130" height="60" rx="8" fill="#581c0c" stroke="#FB923C" strokeWidth="2" />
          <text x="495" y="105" fill="#FED7AA" fontSize="12" fontWeight="bold" textAnchor="middle">Discriminator</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="600" y="75" width="100" height="50" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="650" y="105" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Real / Fake</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderDiffusion = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(100, 105, 140, 105)}
        {drawArrow(420, 105, 458, 105)}
        {drawArrow(548, 105, 586, 105)}
        {drawArrow(676, 105, 714, 105)}

        {/* ×T denoise loop */}
        <motion.path
          d="M 400 46 Q 280 14 160 46"
          fill="none"
          stroke="#A78BFA"
          strokeWidth="1.5"
          strokeDasharray="4 4"
          markerEnd=""
          variants={pathVariants}
        />
        <text x="280" y="20" fill="#C4B5FD" fontSize="10" fontWeight="bold" textAnchor="middle">×T denoise steps</text>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="75" width="80" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="60" y="100" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Noise</text>
          <text x="60" y="118" fill="#6366F1" fontSize="10" textAnchor="middle">x_T</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="140" y="52" width="280" height="118" rx="12" fill="#1e1b4b" stroke="#A78BFA" strokeWidth="2" strokeDasharray="3 3" />
          <text x="280" y="70" fill="#C4B5FD" fontSize="10" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">U-NET (ε-prediction)</text>
          <rect x="160" y="82" width="66" height="78" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="193" y="126" fill="#60A5FA" fontSize="10" fontWeight="semibold" textAnchor="middle">Down</text>
          <rect x="247" y="102" width="66" height="42" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="280" y="127" fill="#F472B6" fontSize="9" fontWeight="semibold" textAnchor="middle">Bottleneck</text>
          <rect x="334" y="82" width="66" height="78" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="367" y="126" fill="#60A5FA" fontSize="10" fontWeight="semibold" textAnchor="middle">Up</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="458" y="75" width="90" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="503" y="100" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Latent</text>
          <text x="503" y="118" fill="#9CA3AF" fontSize="10" textAnchor="middle">x_0</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="586" y="75" width="90" height="60" rx="8" fill="#311042" stroke="#D8B4FE" strokeWidth="1.5" />
          <text x="631" y="110" fill="#E9D5FF" fontSize="11" fontWeight="bold" textAnchor="middle">Decoder</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="714" y="75" width="66" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="747" y="110" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Image</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderLSTM = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(90, 100, 120, 100)}
        {drawArrow(210, 100, 240, 100)}
        {drawArrow(500, 100, 538, 100)}
        {drawArrow(620, 100, 648, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="70" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="55" y="105" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Input</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="120" y="70" width="90" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="165" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Embedding</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="240" y="35" width="260" height="130" rx="12" fill="#064e3b" stroke="#34D399" strokeWidth="2" strokeDasharray="3 3" />
          <text x="370" y="54" fill="#A7F3D0" fontSize="10" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">LSTM CELL · unrolled ×T</text>
          {/* cell-state line */}
          <line x1="255" y1="72" x2="485" y2="72" stroke="#FBBF24" strokeWidth="1.5" strokeDasharray="2 3" />
          <text x="270" y="68" fill="#FBBF24" fontSize="8" textAnchor="start">c_t</text>

          <rect x="258" y="92" width="66" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="291" y="113" fill="#F87171" fontSize="10" fontWeight="semibold" textAnchor="middle">Forget σ</text>
          <rect x="334" y="92" width="66" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="367" y="113" fill="#60A5FA" fontSize="10" fontWeight="semibold" textAnchor="middle">Input σ</text>
          <rect x="410" y="92" width="66" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="443" y="113" fill="#A78BFA" fontSize="10" fontWeight="semibold" textAnchor="middle">Output σ</text>
          <rect x="334" y="132" width="66" height="26" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="367" y="149" fill="#34D399" fontSize="10" fontWeight="semibold" textAnchor="middle">tanh</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="538" y="70" width="82" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="579" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Dense</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="648" y="70" width="82" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="689" y="105" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Output</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderMoE = () => {
    const experts = [
      { y: 24, on: true },
      { y: 62, on: true },
      { y: 104, on: false },
      { y: 146, on: false },
    ];
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(100, 100, 130, 100)}
        {drawArrow(230, 100, 268, 100)}
        {experts.map((e, i) => (
          <g key={`r-${i}`}>
            {drawArrow(268, 100, 300, e.y + 17)}
            {drawArrow(410, e.y + 17, 442, 100)}
          </g>
        ))}
        {drawArrow(542, 100, 578, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="75" width="80" height="50" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="60" y="105" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Input</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="130" y="70" width="100" height="60" rx="8" fill="#0c2740" stroke="#60A5FA" strokeWidth="2" />
          <text x="180" y="96" fill="#93C5FD" fontSize="12" fontWeight="bold" textAnchor="middle">Router</text>
          <text x="180" y="113" fill="#60A5FA" fontSize="9" textAnchor="middle">top-k gating</text>
        </motion.g>
        {experts.map((e, i) => (
          <motion.g key={`e-${i}`} variants={itemVariants} whileHover={{ scale: 1.04 }}>
            <rect
              x="300" y={e.y} width="110" height="34" rx="6"
              fill={e.on ? '#1e1b4b' : '#0A0A0A'}
              stroke={e.on ? '#A78BFA' : '#262626'}
              strokeWidth={e.on ? 2 : 1.5}
            />
            <text x="355" y={e.y + 21} fill={e.on ? '#C4B5FD' : '#525252'} fontSize="10" fontWeight="semibold" textAnchor="middle">
              Expert {i + 1}
            </text>
          </motion.g>
        ))}
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="442" y="70" width="100" height="60" rx="8" fill="#581c0c" stroke="#FB923C" strokeWidth="2" />
          <text x="492" y="96" fill="#FED7AA" fontSize="12" fontWeight="bold" textAnchor="middle">Combine</text>
          <text x="492" y="113" fill="#FB923C" fontSize="9" textAnchor="middle">weighted Σ</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="578" y="75" width="80" height="50" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="618" y="105" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Output</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderUNet = () => {
    // Encoder descends left→down, bottleneck, decoder ascends→up. Dashed skips.
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(70, 45, 85, 45)}
        {drawArrow(155, 55, 180, 78)}
        {drawArrow(245, 98, 270, 120)}
        {drawArrow(340, 148, 365, 165)}
        {drawArrow(445, 165, 470, 148)}
        {drawArrow(535, 120, 560, 98)}
        {drawArrow(625, 78, 650, 55)}
        {drawArrow(720, 45, 735, 45)}

        {/* skip connections (dashed) */}
        <motion.path d="M 155 45 L 645 45" fill="none" stroke="#60A5FA" strokeWidth="1.5" strokeDasharray="4 4" variants={pathVariants} />
        <motion.path d="M 245 88 L 555 88" fill="none" stroke="#60A5FA" strokeWidth="1.5" strokeDasharray="4 4" variants={pathVariants} />
        <motion.path d="M 335 130 L 465 130" fill="none" stroke="#60A5FA" strokeWidth="1.5" strokeDasharray="4 4" variants={pathVariants} />
        <text x="400" y="40" fill="#60A5FA" fontSize="9" textAnchor="middle">skip connections</text>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="15" y="25" width="55" height="40" rx="7" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="42" y="49" fill="#A5B4FC" fontSize="10" fontWeight="bold" textAnchor="middle">Image</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="85" y="25" width="70" height="40" rx="7" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="120" y="49" fill="#D1D5DB" fontSize="10" fontWeight="bold" textAnchor="middle">Enc 1</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="180" y="68" width="70" height="40" rx="7" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="215" y="92" fill="#D1D5DB" fontSize="10" fontWeight="bold" textAnchor="middle">Enc 2</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="270" y="110" width="70" height="40" rx="7" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="305" y="134" fill="#D1D5DB" fontSize="10" fontWeight="bold" textAnchor="middle">Enc 3</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="360" y="150" width="80" height="38" rx="7" fill="#311042" stroke="#D8B4FE" strokeWidth="2" />
          <text x="400" y="173" fill="#E9D5FF" fontSize="10" fontWeight="bold" textAnchor="middle">Bottleneck</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="460" y="110" width="70" height="40" rx="7" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="495" y="134" fill="#D1D5DB" fontSize="10" fontWeight="bold" textAnchor="middle">Dec 3</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="550" y="68" width="70" height="40" rx="7" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="585" y="92" fill="#D1D5DB" fontSize="10" fontWeight="bold" textAnchor="middle">Dec 2</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="645" y="25" width="70" height="40" rx="7" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="680" y="49" fill="#D1D5DB" fontSize="10" fontWeight="bold" textAnchor="middle">Dec 1</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="735" y="25" width="55" height="40" rx="7" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="762" y="49" fill="#34D399" fontSize="10" fontWeight="bold" textAnchor="middle">Mask</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderMamba = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(90, 100, 115, 100)}
        {drawArrow(200, 100, 225, 100)}
        {drawArrow(525, 100, 545, 100)}
        {drawArrow(635, 100, 655, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="70" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="55" y="105" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Input</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="115" y="70" width="85" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="157" y="105" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Embedding</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="225" y="35" width="300" height="130" rx="12" fill="#064e3b" stroke="#34D399" strokeWidth="2" strokeDasharray="3 3" />
          <text x="375" y="54" fill="#A7F3D0" fontSize="10" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">N× MAMBA BLOCK</text>
          <rect x="243" y="88" width="62" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="274" y="109" fill="#60A5FA" fontSize="10" fontWeight="semibold" textAnchor="middle">Linear ↑</text>
          <rect x="313" y="88" width="60" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="343" y="109" fill="#60A5FA" fontSize="10" fontWeight="semibold" textAnchor="middle">Conv1d</text>
          <rect x="381" y="88" width="126" height="34" rx="6" fill="#0A0A0A" stroke="#34D399" strokeWidth="1.5" />
          <text x="444" y="109" fill="#34D399" fontSize="10" fontWeight="semibold" textAnchor="middle">SSM (selective scan)</text>
          <rect x="313" y="128" width="194" height="26" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="410" y="145" fill="#F472B6" fontSize="10" fontWeight="semibold" textAnchor="middle">Gate ⊗ SiLU</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="545" y="70" width="90" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="590" y="98" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">RMSNorm</text>
          <text x="590" y="115" fill="#9CA3AF" fontSize="10" textAnchor="middle">+ Linear</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="655" y="70" width="80" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="695" y="105" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Output</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderGNN = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(130, 100, 158, 100)}
        {drawArrow(438, 100, 466, 100)}
        {drawArrow(566, 100, 594, 100)}

        {/* Input graph glyph */}
        <motion.g variants={itemVariants} whileHover={{ scale: 1.04 }}>
          <rect x="15" y="45" width="115" height="110" rx="10" fill="#0A0A0A" stroke="#312E81" strokeWidth="1.5" />
          <line x1="45" y1="75" x2="95" y2="70" stroke="#3730A3" strokeWidth="1.5" />
          <line x1="45" y1="75" x2="55" y2="125" stroke="#3730A3" strokeWidth="1.5" />
          <line x1="95" y1="70" x2="100" y2="120" stroke="#3730A3" strokeWidth="1.5" />
          <line x1="55" y1="125" x2="100" y2="120" stroke="#3730A3" strokeWidth="1.5" />
          <circle cx="45" cy="75" r="9" fill="#1E1B4B" stroke="#A5B4FC" strokeWidth="1.5" />
          <circle cx="95" cy="70" r="9" fill="#1E1B4B" stroke="#A5B4FC" strokeWidth="1.5" />
          <circle cx="55" cy="125" r="9" fill="#1E1B4B" stroke="#A5B4FC" strokeWidth="1.5" />
          <circle cx="100" cy="120" r="9" fill="#1E1B4B" stroke="#A5B4FC" strokeWidth="1.5" />
          <text x="72" y="168" fill="#A5B4FC" fontSize="10" fontWeight="bold" textAnchor="middle">Graph</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="158" y="40" width="280" height="120" rx="12" fill="#1e1b4b" stroke="#A78BFA" strokeWidth="2" strokeDasharray="3 3" />
          <text x="298" y="60" fill="#C4B5FD" fontSize="10" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">K× GNN LAYER</text>
          <rect x="176" y="80" width="76" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="214" y="101" fill="#A78BFA" fontSize="10" fontWeight="semibold" textAnchor="middle">Message</text>
          <rect x="262" y="80" width="76" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="300" y="101" fill="#60A5FA" fontSize="10" fontWeight="semibold" textAnchor="middle">Aggregate Σ</text>
          <rect x="348" y="80" width="76" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="386" y="101" fill="#F472B6" fontSize="10" fontWeight="semibold" textAnchor="middle">Update</text>
          <text x="298" y="140" fill="#525252" fontSize="9" textAnchor="middle">message passing</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="466" y="70" width="100" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="516" y="96" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Readout</text>
          <text x="516" y="113" fill="#9CA3AF" fontSize="9" textAnchor="middle">pool</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="594" y="75" width="80" height="50" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="634" y="105" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Output</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderGRU = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(90, 100, 120, 100)}
        {drawArrow(210, 100, 240, 100)}
        {drawArrow(500, 100, 538, 100)}
        {drawArrow(620, 100, 648, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="70" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="55" y="105" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Input</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="120" y="70" width="90" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="165" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Embedding</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="240" y="35" width="260" height="130" rx="12" fill="#1e293b" stroke="#60A5FA" strokeWidth="2" strokeDasharray="3 3" />
          <text x="370" y="54" fill="#93C5FD" fontSize="10" fontWeight="bold" textAnchor="middle" letterSpacing="0.1em">GRU CELL · unrolled ×T</text>
          <line x1="255" y1="72" x2="485" y2="72" stroke="#FBBF24" strokeWidth="1.5" strokeDasharray="2 3" />
          <text x="270" y="68" fill="#FBBF24" fontSize="8" textAnchor="start">h_t</text>
          <rect x="270" y="92" width="66" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="303" y="113" fill="#F87171" fontSize="10" fontWeight="semibold" textAnchor="middle">Reset σ</text>
          <rect x="346" y="92" width="66" height="34" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="379" y="113" fill="#60A5FA" fontSize="10" fontWeight="semibold" textAnchor="middle">Update σ</text>
          <rect x="308" y="132" width="104" height="26" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="360" y="149" fill="#34D399" fontSize="10" fontWeight="semibold" textAnchor="middle">Candidate tanh</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="538" y="70" width="82" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="579" y="105" fill="#D1D5DB" fontSize="12" fontWeight="bold" textAnchor="middle">Dense</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="648" y="70" width="82" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="689" y="105" fill="#34D399" fontSize="12" fontWeight="bold" textAnchor="middle">Output</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderVAE = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(90, 100, 118, 100)}
        {drawArrow(208, 88, 236, 76)}
        {drawArrow(208, 112, 236, 124)}
        {drawArrow(326, 100, 354, 100)}
        {drawArrow(444, 100, 472, 100)}
        {drawArrow(562, 100, 590, 100)}

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="70" width="70" height="60" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="55" y="105" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Input</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="118" y="70" width="90" height="60" rx="8" fill="#0c2740" stroke="#60A5FA" strokeWidth="1.5" />
          <text x="163" y="105" fill="#93C5FD" fontSize="12" fontWeight="bold" textAnchor="middle">Encoder</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="236" y="58" width="66" height="34" rx="6" fill="#0A0A0A" stroke="#A78BFA" strokeWidth="1.5" />
          <text x="269" y="79" fill="#C4B5FD" fontSize="11" fontWeight="bold" textAnchor="middle">μ</text>
          <rect x="236" y="108" width="66" height="34" rx="6" fill="#0A0A0A" stroke="#A78BFA" strokeWidth="1.5" />
          <text x="269" y="129" fill="#C4B5FD" fontSize="11" fontWeight="bold" textAnchor="middle">σ</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="354" y="70" width="90" height="60" rx="8" fill="#1e1b4b" stroke="#A78BFA" strokeWidth="2" />
          <text x="399" y="96" fill="#C4B5FD" fontSize="11" fontWeight="bold" textAnchor="middle">z ~ N(μ,σ²)</text>
          <text x="399" y="113" fill="#8B7FD6" fontSize="8" textAnchor="middle">reparameterize</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="472" y="70" width="90" height="60" rx="8" fill="#311042" stroke="#D8B4FE" strokeWidth="1.5" />
          <text x="517" y="105" fill="#E9D5FF" fontSize="12" fontWeight="bold" textAnchor="middle">Decoder</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="590" y="70" width="100" height="60" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="640" y="98" fill="#34D399" fontSize="11" fontWeight="bold" textAnchor="middle">Recon-</text>
          <text x="640" y="115" fill="#34D399" fontSize="11" fontWeight="bold" textAnchor="middle">struction</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderRLHF = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(100, 100, 128, 100)}
        {drawArrow(242, 100, 270, 100)}
        {drawArrow(362, 100, 390, 100)}
        {drawArrow(502, 100, 530, 100)}

        {/* PPO feedback loop back to policy */}
        <motion.path d="M 575 128 Q 400 195 185 132" fill="none" stroke="#34D399" strokeWidth="1.5" strokeDasharray="4 4" variants={pathVariants} />
        <text x="380" y="188" fill="#34D399" fontSize="9" textAnchor="middle">PPO policy update ← reward</text>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="20" y="75" width="80" height="50" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="60" y="105" fill="#A5B4FC" fontSize="12" fontWeight="bold" textAnchor="middle">Prompt</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="128" y="70" width="114" height="60" rx="8" fill="#1e1b4b" stroke="#A78BFA" strokeWidth="2" />
          <text x="185" y="96" fill="#C4B5FD" fontSize="12" fontWeight="bold" textAnchor="middle">Policy</text>
          <text x="185" y="113" fill="#8B7FD6" fontSize="9" textAnchor="middle">LLM πθ</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="270" y="75" width="92" height="50" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="316" y="105" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Response</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="390" y="70" width="112" height="60" rx="8" fill="#581c0c" stroke="#FB923C" strokeWidth="2" />
          <text x="446" y="96" fill="#FED7AA" fontSize="11" fontWeight="bold" textAnchor="middle">Reward</text>
          <text x="446" y="113" fill="#FB923C" fontSize="9" textAnchor="middle">Model</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="530" y="75" width="90" height="50" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="575" y="105" fill="#34D399" fontSize="11" fontWeight="bold" textAnchor="middle">Reward r</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderEncDec = () => {
    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {drawArrow(75, 100, 95, 100)}
        {drawArrow(520, 100, 548, 100)}
        {drawArrow(660, 100, 688, 100)}
        {/* memory / cross-attention */}
        <motion.path d="M 250 100 L 335 100" fill="none" stroke="#A78BFA" strokeWidth="2" strokeDasharray="4 4" variants={pathVariants} />
        <text x="292" y="92" fill="#C4B5FD" fontSize="8" textAnchor="middle">memory</text>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="15" y="75" width="60" height="50" rx="8" fill="#1E1B4B" stroke="#312E81" strokeWidth="1.5" />
          <text x="45" y="104" fill="#A5B4FC" fontSize="10" fontWeight="bold" textAnchor="middle">Input</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="95" y="35" width="155" height="130" rx="12" fill="#1e293b" stroke="#60A5FA" strokeWidth="2" strokeDasharray="3 3" />
          <text x="172" y="54" fill="#93C5FD" fontSize="10" fontWeight="bold" textAnchor="middle">N× ENCODER</text>
          <rect x="112" y="72" width="120" height="30" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="172" y="91" fill="#60A5FA" fontSize="10" fontWeight="semibold" textAnchor="middle">Self-Attention</text>
          <rect x="112" y="112" width="120" height="30" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="172" y="131" fill="#F472B6" fontSize="10" fontWeight="semibold" textAnchor="middle">Feed-Forward</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.02 }}>
          <rect x="335" y="35" width="185" height="130" rx="12" fill="#581c0c" stroke="#FB923C" strokeWidth="2" strokeDasharray="3 3" />
          <text x="427" y="54" fill="#FED7AA" fontSize="10" fontWeight="bold" textAnchor="middle">N× DECODER</text>
          <rect x="352" y="66" width="150" height="26" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="427" y="84" fill="#FB923C" fontSize="9" fontWeight="semibold" textAnchor="middle">Masked Self-Attn</text>
          <rect x="352" y="98" width="150" height="26" rx="6" fill="#0A0A0A" stroke="#A78BFA" strokeWidth="1.5" />
          <text x="427" y="116" fill="#C4B5FD" fontSize="9" fontWeight="semibold" textAnchor="middle">Cross-Attention</text>
          <rect x="352" y="130" width="150" height="26" rx="6" fill="#0A0A0A" stroke="#262626" strokeWidth="1.5" />
          <text x="427" y="148" fill="#F472B6" fontSize="9" fontWeight="semibold" textAnchor="middle">Feed-Forward</text>
        </motion.g>

        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="548" y="70" width="112" height="60" rx="8" fill="#111827" stroke="#374151" strokeWidth="1.5" />
          <text x="604" y="98" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Linear +</text>
          <text x="604" y="115" fill="#D1D5DB" fontSize="11" fontWeight="bold" textAnchor="middle">Softmax</text>
        </motion.g>
        <motion.g variants={itemVariants} whileHover={{ scale: 1.05 }}>
          <rect x="688" y="75" width="66" height="50" rx="8" fill="#064E3B" stroke="#065F46" strokeWidth="1.5" />
          <text x="721" y="104" fill="#34D399" fontSize="10" fontWeight="bold" textAnchor="middle">Output</text>
        </motion.g>
      </motion.svg>
    );
  };

  const renderGeneric = (blocks: GenBlock[]) => {
    const n = blocks.length;
    const gap = 14;
    const boxW = Math.max(70, Math.min(130, Math.floor((800 - 40 - gap * (n - 1)) / n)));
    const boxH = 66;
    const step = boxW + gap;
    const totalW = step * n - gap;
    const startX = Math.max(20, (800 - totalW) / 2);
    const y = 67;
    const cy = y + boxH / 2;

    return (
      <motion.svg
        viewBox="0 0 800 200"
        className="w-full h-[200px] bg-[#0A0A0A] border border-[#262626] rounded-xl p-4"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {blocks.map((_, i) =>
          i < n - 1 ? drawArrow(startX + i * step + boxW, cy, startX + (i + 1) * step, cy) : null,
        )}
        {blocks.map((b, i) => {
          const x = startX + i * step;
          const accent = b.accent || '#A78BFA';
          return (
            <motion.g key={i} variants={itemVariants} whileHover={{ scale: 1.06 }}>
              <rect x={x} y={y} width={boxW} height={boxH} rx="8" fill="#0A0A0A" stroke={accent} strokeWidth="1.5" />
              <text
                x={x + boxW / 2}
                y={b.sub ? y + 30 : cy + 4}
                fill={accent}
                fontSize="11"
                fontWeight="bold"
                textAnchor="middle"
              >
                {b.label}
              </text>
              {b.sub && (
                <text x={x + boxW / 2} y={y + 46} fill="#A3A3A3" fontSize="9" textAnchor="middle">
                  {b.sub}
                </text>
              )}
            </motion.g>
          );
        })}
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
    case 'gan':
      return renderGAN();
    case 'diffusion':
      return renderDiffusion();
    case 'lstm':
      return renderLSTM();
    case 'moe':
      return renderMoE();
    case 'unet':
      return renderUNet();
    case 'mamba':
      return renderMamba();
    case 'gnn':
      return renderGNN();
    case 'gru':
      return renderGRU();
    case 'vae':
      return renderVAE();
    case 'rlhf':
      return renderRLHF();
    case 'encdec':
      return renderEncDec();
    default:
      if (GENERIC_FLOWS[slug]) return renderGeneric(GENERIC_FLOWS[slug]);
      return null;
  }
}
