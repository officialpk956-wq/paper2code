import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TopicHero } from '@/components/topic/TopicHero';
import type { TopicMeta } from '@/data/topics/types';

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

const meta: TopicMeta = {
  slug: 'attention',
  title: 'Attention Mechanism',
  subtitle: 'The core idea behind Transformers.',
  difficulty: 'intermediate',
  durationMinutes: 45,
  prerequisites: ['Linear Algebra', 'Neural Networks'],
  domain: 'Deep Learning',
  domainSlug: 'deep-learning',
  domainColor: '#0EA5E9',
  tags: ['transformer', 'nlp'],
  completionPercent: 30,
};

describe('TopicHero', () => {
  it('renders topic title', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getByRole('heading', { name: 'Attention Mechanism' })).toBeInTheDocument();
  });

  it('renders subtitle', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getByText('The core idea behind Transformers.')).toBeInTheDocument();
  });

  it('renders domain badge', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getAllByText('Deep Learning').length).toBeGreaterThan(0);
  });

  it('renders difficulty badge', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getByText('Intermediate')).toBeInTheDocument();
  });

  it('renders duration', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getByText(/45 min/i)).toBeInTheDocument();
  });

  it('renders prerequisite count', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getByText(/2 prerequisites/i)).toBeInTheDocument();
  });

  it('renders tags', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getByText('transformer')).toBeInTheDocument();
  });

  it('renders breadcrumb links', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getByRole('link', { name: 'Learn' })).toBeInTheDocument();
  });

  it('renders back to domain link', () => {
    render(<TopicHero meta={meta} />);
    expect(screen.getByRole('link', { name: /back to deep learning/i })).toBeInTheDocument();
  });

  it('back link points to domain page', () => {
    render(<TopicHero meta={meta} />);
    const link = screen.getByRole('link', { name: /back to deep learning/i });
    expect(link.getAttribute('href')).toBe('/learn/deep-learning');
  });
});
