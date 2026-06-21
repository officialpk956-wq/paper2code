import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DomainHero } from '@/components/domain/DomainHero';
import type { DomainData } from '@/data/domains/types';

vi.mock('framer-motion', () => ({
  motion: new Proxy({}, {
    get: (_t, tag) => {
      const React = require('react');
      return ({ children, ...rest }: React.HTMLAttributes<HTMLElement>) =>
        React.createElement(tag as string, rest, children);
    },
  }),
  AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
}));

const mockData: DomainData = {
  slug: 'deep-learning',
  name: 'Deep Learning',
  tagline: 'Master neural networks and modern AI systems.',
  description: 'Full description here.',
  color: '#0EA5E9',
  glowColor: 'rgba(14,165,233,0.25)',
  icon: 'Layers3',
  stats: { lessons: 72, tracks: 8, problems: 145, projects: 23 },
  progression: [
    { level: 'beginner', label: 'Foundations', description: 'Basics', topicSamples: ['Perceptron'] },
    { level: 'intermediate', label: 'Core', description: 'Intermediate', topicSamples: ['CNN'] },
    { level: 'advanced', label: 'Modern', description: 'Advanced', topicSamples: ['ViT'] },
  ],
  progress: { lessonsCompleted: 24, totalLessons: 72, currentStreak: 5, estimatedHoursRemaining: 44, masteryPercent: 34 },
  roadmap: [],
  topicClusters: [],
  featuredLessons: [],
  kgNodes: [],
  kgEdges: [],
  projects: [],
  papers: [],
};

describe('DomainHero', () => {
  it('renders domain name as heading', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByRole('heading', { name: /deep learning/i })).toBeInTheDocument();
  });

  it('renders tagline', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByText(/master neural networks/i)).toBeInTheDocument();
  });

  it('renders all 4 stats', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByText('72')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();
    expect(screen.getByText('145')).toBeInTheDocument();
    expect(screen.getByText('23')).toBeInTheDocument();
  });

  it('renders stat labels', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByText('Lessons')).toBeInTheDocument();
    expect(screen.getByText('Tracks')).toBeInTheDocument();
    expect(screen.getByText('Problems')).toBeInTheDocument();
    expect(screen.getByText('Projects')).toBeInTheDocument();
  });

  it('renders all 3 progression levels', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByText('Foundations')).toBeInTheDocument();
    expect(screen.getByText('Core')).toBeInTheDocument();
    expect(screen.getByText('Modern')).toBeInTheDocument();
  });

  it('renders progression level labels (beginner/intermediate/advanced)', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByText('beginner')).toBeInTheDocument();
    expect(screen.getByText('intermediate')).toBeInTheDocument();
    expect(screen.getByText('advanced')).toBeInTheDocument();
  });

  it('renders topic samples', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByText('Perceptron')).toBeInTheDocument();
    expect(screen.getByText('CNN')).toBeInTheDocument();
    expect(screen.getByText('ViT')).toBeInTheDocument();
  });

  it('renders breadcrumb back to Learn', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByRole('link', { name: /back to learn home/i })).toHaveAttribute('href', '/learn');
  });

  it('renders Learning Progression section header', () => {
    render(<DomainHero data={mockData} />);
    expect(screen.getByText(/learning progression/i)).toBeInTheDocument();
  });

  it('renders the domain badge with name', () => {
    render(<DomainHero data={mockData} />);
    const badges = screen.getAllByText('Deep Learning');
    expect(badges.length).toBeGreaterThanOrEqual(1);
  });
});
