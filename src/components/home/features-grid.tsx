'use client';

import { Code2, Network, BookOpen, Zap } from 'lucide-react';

interface Feature {
  icon: React.ReactNode;
  title: string;
  description: string;
}

const features: Feature[] = [
  {
    icon: <Code2 size={32} />,
    title: 'Paper-to-Code',
    description: 'Transform research papers into production-ready implementations with step-by-step guidance.',
  },
  {
    icon: <Network size={32} />,
    title: 'Interactive Architectures',
    description: 'Explore transformer, diffusion, and attention mechanisms with animated visualizations.',
  },
  {
    icon: <BookOpen size={32} />,
    title: '110+ Coding Problems',
    description: 'LeetCode-style problems covering algorithms, data structures, and AI engineering fundamentals.',
  },
  {
    icon: <Zap size={32} />,
    title: 'Guided Roadmaps',
    description: 'Structured learning paths from foundations to advanced topics with clear progression.',
  },
];

export function FeaturesGrid() {
  return (
    <section className="relative py-20 md:py-32 overflow-hidden">
      <div className="max-w-6xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-heading font-bold mb-4">
            <span className="gradient-text">Everything You Need</span>
            <br />
            <span className="text-[--color-text-primary]">for Deep Learning</span>
          </h2>
          <p className="text-lg text-[--color-text-secondary] max-w-2xl mx-auto leading-relaxed">
            A complete suite of tools and resources for mastering AI engineering through hands-on practice.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="workspace-card group cursor-pointer"
            >
              {/* Icon */}
              <div className="mb-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-[--accent-primary] to-[--accent-cyan]
                              flex items-center justify-center text-white group-hover:scale-110 transition-transform duration-300">
                  {feature.icon}
                </div>
              </div>

              {/* Content */}
              <h3 className="text-xl font-bold text-[--color-text-primary] mb-3 group-hover:text-[--accent-primary] transition-colors">
                {feature.title}
              </h3>
              <p className="text-[--color-text-secondary] leading-relaxed">
                {feature.description}
              </p>

              {/* Accent line on hover */}
              <div className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan]
                            w-0 group-hover:w-full transition-all duration-300 rounded-b-lg" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
