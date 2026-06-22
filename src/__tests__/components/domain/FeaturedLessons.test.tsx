import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FeaturedLessons } from '@/components/domain/FeaturedLessons';
import type { DomainLesson } from '@/data/domains/types';

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

vi.mock('@/lib/progress', () => ({
  isCompleted: vi.fn((_type, slug) => slug === 'attention'),
}));

vi.mock('@/data/topics', () => ({
  resolveTopicPath: vi.fn((slug) => `/learn/deep-learning/${slug}`),
}));

const lessons: DomainLesson[] = [
  { id: 'l1', title: 'Attention Mechanism', description: 'Learn self-attention', difficulty: 'advanced', durationMinutes: 45, completionPercent: 100, topicSlug: 'attention', isNew: false },
  { id: 'l2', title: 'Backpropagation', description: 'Chain rule in practice', difficulty: 'intermediate', durationMinutes: 30, completionPercent: 60, topicSlug: 'backprop', isNew: false },
  { id: 'l3', title: 'Diffusion Models', description: 'Score-based generative models', difficulty: 'advanced', durationMinutes: 60, completionPercent: 0, topicSlug: 'diffusion', isNew: true },
];

describe('FeaturedLessons', () => {
  it('renders section heading', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByRole('heading', { name: /featured lessons/i })).toBeInTheDocument();
  });

  it('renders lesson titles', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText('Attention Mechanism')).toBeInTheDocument();
    expect(screen.getByText('Backpropagation')).toBeInTheDocument();
    expect(screen.getByText('Diffusion Models')).toBeInTheDocument();
  });

  it('renders lesson descriptions', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText('Learn self-attention')).toBeInTheDocument();
  });

  it('renders duration for each lesson', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText('45 min')).toBeInTheDocument();
    expect(screen.getByText('30 min')).toBeInTheDocument();
    expect(screen.getByText('60 min')).toBeInTheDocument();
  });

  it('shows "Review" CTA for completed lesson', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText('Review')).toBeInTheDocument();
  });

  it('shows "Start" CTA for not-started lesson', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getAllByText('Start').length).toBe(2);
  });

  it('renders "New" badge for new lesson', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText(/new/i)).toBeInTheDocument();
  });

  it('links each lesson to the correct topic page', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    const attentionLink = screen.getByRole('link', { name: /attention mechanism/i });
    expect(attentionLink.getAttribute('href')).toBe('/learn/deep-learning/attention');
  });

  it('renders "View all" link', () => {
    render(<FeaturedLessons lessons={lessons} domainSlug="deep-learning" color="#0EA5E9" />);
    const viewAll = screen.getByRole('link', { name: /view all/i });
    expect(viewAll.getAttribute('href')).toBe('/learn/deep-learning/lessons');
  });
});
