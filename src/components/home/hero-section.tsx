'use client';

import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

export function HeroSection() {
  return (
    <section className="relative py-20 md:py-32 overflow-hidden">
      {/* Aurora Background */}
      <div
        className="absolute inset-0 opacity-40 pointer-events-none"
        style={{
          background:
            'radial-gradient(circle at 20% 50%, rgba(124, 58, 237, 0.2) 0%, transparent 50%), radial-gradient(circle at 80% 50%, rgba(6, 182, 212, 0.2) 0%, transparent 50%)',
          animation: 'aurora 8s ease infinite',
        }}
      />

      <div className="relative z-10 max-w-4xl mx-auto px-6">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[--bg-panel] border border-[--color-border] mb-8 hover:border-[--accent-primary] transition-colors">
          <span className="w-2 h-2 rounded-full bg-[--accent-primary] animate-pulse" />
          <span className="text-xs font-semibold text-[--accent-light]">
            Now with 110+ LeetCode problems
          </span>
        </div>

        {/* Main Title */}
        <h1 className="text-5xl md:text-6xl lg:text-7xl font-heading font-bold mb-6 leading-tight">
          <span className="gradient-text">Master AI Engineering</span>
          <br />
          <span className="text-[--color-text-primary]">Through Deep Work</span>
        </h1>

        {/* Subtitle */}
        <p className="text-lg md:text-xl text-[--color-text-secondary] mb-8 leading-relaxed max-w-2xl">
          Learn transformers, LLMs, and deep learning through 110+ problems, 50+ papers,
          interactive architectures, and guided roadmaps. Professional tool, not a website.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 mb-16">
          <Link
            href="/dashboard"
            className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-lg
            bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan]
            text-white font-semibold text-sm hover:opacity-90 transition-opacity"
          >
            Launch Dashboard
            <ArrowRight size={18} />
          </Link>
          <Link
            href="/problems"
            className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-lg
            bg-[--bg-panel] border border-[--color-border] text-[--color-text-primary]
            font-semibold text-sm hover:border-[--accent-primary] hover:text-[--accent-primary]
            transition-colors"
          >
            Solve Problems
            <ArrowRight size={18} />
          </Link>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-3 gap-6 md:gap-12">
          <div>
            <div className="text-3xl md:text-4xl font-bold gradient-text mb-2">110+</div>
            <p className="text-sm text-[--color-text-secondary]">Coding Problems</p>
          </div>
          <div>
            <div className="text-3xl md:text-4xl font-bold gradient-text mb-2">50+</div>
            <p className="text-sm text-[--color-text-secondary]">Research Papers</p>
          </div>
          <div>
            <div className="text-3xl md:text-4xl font-bold gradient-text mb-2">6</div>
            <p className="text-sm text-[--color-text-secondary]">Learning Tracks</p>
          </div>
        </div>
      </div>
    </section>
  );
}
