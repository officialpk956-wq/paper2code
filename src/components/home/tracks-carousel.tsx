'use client';

import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

interface Track {
  id: string;
  title: string;
  description: string;
  icon: string;
  problems: number;
  color: 'purple' | 'cyan' | 'emerald' | 'orange' | 'pink' | 'blue';
}

const tracks: Track[] = [
  {
    id: 'foundations',
    title: 'AI Foundations',
    description: 'Core concepts of machine learning, neural networks, and linear algebra.',
    icon: '📚',
    problems: 18,
    color: 'purple',
  },
  {
    id: 'transformers',
    title: 'Transformers & LLMs',
    description: 'Attention mechanisms, BERT, GPT, and large language model architectures.',
    icon: '🤖',
    problems: 22,
    color: 'cyan',
  },
  {
    id: 'diffusion',
    title: 'Diffusion Models',
    description: 'Generative models, DDPM, latent diffusion, and image synthesis.',
    icon: '✨',
    problems: 16,
    color: 'emerald',
  },
  {
    id: 'systems',
    title: 'System Design',
    description: 'Distributed training, inference optimization, and production deployment.',
    icon: '⚙️',
    problems: 20,
    color: 'orange',
  },
  {
    id: 'reinforcement',
    title: 'Reinforcement Learning',
    description: 'Policy gradients, actor-critic methods, and multi-agent systems.',
    icon: '🎮',
    problems: 19,
    color: 'pink',
  },
  {
    id: 'specialized',
    title: 'Specialized Topics',
    description: 'Vision transformers, multi-modal models, and cutting-edge research.',
    icon: '🔬',
    problems: 15,
    color: 'blue',
  },
];

const colorGradients = {
  purple: 'from-purple-500/20 to-purple-600/10 border-purple-500/30 hover:border-purple-500/50',
  cyan: 'from-cyan-500/20 to-cyan-600/10 border-cyan-500/30 hover:border-cyan-500/50',
  emerald: 'from-emerald-500/20 to-emerald-600/10 border-emerald-500/30 hover:border-emerald-500/50',
  orange: 'from-orange-500/20 to-orange-600/10 border-orange-500/30 hover:border-orange-500/50',
  pink: 'from-pink-500/20 to-pink-600/10 border-pink-500/30 hover:border-pink-500/50',
  blue: 'from-blue-500/20 to-blue-600/10 border-blue-500/30 hover:border-blue-500/50',
};

const accentColors = {
  purple: 'text-purple-400',
  cyan: 'text-cyan-400',
  emerald: 'text-emerald-400',
  orange: 'text-orange-400',
  pink: 'text-pink-400',
  blue: 'text-blue-400',
};

export function TracksCarousel() {
  return (
    <section className="relative py-20 md:py-32 overflow-hidden">
      <div className="max-w-7xl mx-auto px-6">
        {/* Section Header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-heading font-bold mb-4">
            <span className="text-[--color-text-primary]">6 Comprehensive</span>
            <br />
            <span className="gradient-text">Learning Tracks</span>
          </h2>
          <p className="text-lg text-[--color-text-secondary] max-w-2xl mx-auto leading-relaxed">
            Structured curricula spanning 110+ problems, from fundamentals to cutting-edge research implementation.
          </p>
        </div>

        {/* Tracks Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tracks.map((track) => (
            <Link
              key={track.id}
              href={`/academy/${track.id}`}
              className={`group relative overflow-hidden rounded-lg border transition-all duration-300
                         bg-gradient-to-br ${colorGradients[track.color]} p-6
                         hover:shadow-lg hover:scale-105`}
            >
              {/* Background accent */}
              <div className="absolute top-0 right-0 w-32 h-32 opacity-0 group-hover:opacity-10 transition-opacity
                            bg-gradient-to-br from-white blur-3xl" />

              {/* Content */}
              <div className="relative z-10">
                {/* Icon and Title */}
                <div className="flex items-start justify-between mb-4">
                  <div className="text-5xl">{track.icon}</div>
                  <ArrowRight
                    size={20}
                    className="text-[--color-text-secondary] group-hover:text-[--accent-primary]
                             transform group-hover:translate-x-1 transition-all duration-300"
                  />
                </div>

                {/* Title */}
                <h3 className="text-xl font-bold text-[--color-text-primary] mb-2">
                  {track.title}
                </h3>

                {/* Description */}
                <p className="text-sm text-[--color-text-secondary] mb-4 line-clamp-2">
                  {track.description}
                </p>

                {/* Problem count */}
                <div className="flex items-center gap-2 text-sm">
                  <span className={`font-semibold ${accentColors[track.color]}`}>
                    {track.problems} problems
                  </span>
                  <span className="text-[--color-text-tertiary]">•</span>
                  <span className="text-[--color-text-tertiary]">Self-paced</span>
                </div>
              </div>

              {/* Bottom accent bar */}
              <div className={`absolute bottom-0 left-0 h-1 bg-gradient-to-r
                            w-0 group-hover:w-full transition-all duration-300
                            ${track.color === 'purple' && 'from-purple-500 to-purple-600'}
                            ${track.color === 'cyan' && 'from-cyan-500 to-cyan-600'}
                            ${track.color === 'emerald' && 'from-emerald-500 to-emerald-600'}
                            ${track.color === 'orange' && 'from-orange-500 to-orange-600'}
                            ${track.color === 'pink' && 'from-pink-500 to-pink-600'}
                            ${track.color === 'blue' && 'from-blue-500 to-blue-600'}`} />
            </Link>
          ))}
        </div>

        {/* CTA */}
        <div className="text-center mt-16">
          <Link
            href="/academy"
            className="inline-flex items-center justify-center gap-2 px-8 py-4 rounded-lg
                     bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan]
                     text-white font-semibold text-sm hover:opacity-90 transition-opacity
                     transition-opacity"
          >
            Explore All Tracks
            <ArrowRight size={18} />
          </Link>
        </div>
      </div>
    </section>
  );
}
