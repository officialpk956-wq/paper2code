import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TopicClusters } from '@/components/domain/TopicClusters';
import type { TopicCluster } from '@/data/domains/types';

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

vi.mock('@/data/topics', () => ({
  resolveTopicPath: vi.fn((slug) => `/learn/deep-learning/${slug}`),
}));

const clusters: TopicCluster[] = [
  {
    id: 'nn',
    title: 'Neural Networks',
    description: 'Fundamental building blocks',
    color: '#0EA5E9',
    topics: [
      { id: 't1', title: 'Perceptron', status: 'completed', lessonCount: 4, difficulty: 'beginner', slug: 'perceptron' },
      { id: 't2', title: 'CNNs', status: 'in_progress', lessonCount: 6, difficulty: 'intermediate', slug: 'cnns' },
      { id: 't3', title: 'Transformers', status: 'locked', lessonCount: 8, difficulty: 'advanced', slug: 'transformers' },
    ],
  },
  {
    id: 'training',
    title: 'Training Techniques',
    description: 'Regularization and optimization',
    color: '#7C3AED',
    topics: [
      { id: 't4', title: 'Batch Norm', status: 'available', lessonCount: 3, difficulty: 'intermediate', slug: 'batch-norm' },
    ],
  },
];

describe('TopicClusters', () => {
  it('renders section heading', () => {
    render(<TopicClusters clusters={clusters} />);
    expect(screen.getByRole('heading', { name: /topic clusters/i })).toBeInTheDocument();
  });

  it('renders cluster titles', () => {
    render(<TopicClusters clusters={clusters} />);
    expect(screen.getByText('Neural Networks')).toBeInTheDocument();
    expect(screen.getByText('Training Techniques')).toBeInTheDocument();
  });

  it('renders cluster descriptions', () => {
    render(<TopicClusters clusters={clusters} />);
    expect(screen.getByText('Fundamental building blocks')).toBeInTheDocument();
    expect(screen.getByText('Regularization and optimization')).toBeInTheDocument();
  });

  it('renders topic titles', () => {
    render(<TopicClusters clusters={clusters} />);
    expect(screen.getByText('Perceptron')).toBeInTheDocument();
    expect(screen.getByText('CNNs')).toBeInTheDocument();
  });

  it('navigates to correct href for available topics', () => {
    render(<TopicClusters clusters={clusters} />);
    const link = screen.getByRole('link', { name: /cnns/i });
    expect(link).toHaveAttribute('href', '/learn/deep-learning/cnns');
  });

  it('locked topics are rendered as disabled divs', () => {
    render(<TopicClusters clusters={clusters} />);
    // Since it's a div, we query by text or aria-label. It should have the text "Transformers"
    const lockedDiv = screen.getByText('Transformers').closest('div[aria-label]');
    expect(lockedDiv).toBeInTheDocument();
    expect(lockedDiv).toHaveStyle('cursor: not-allowed');
  });

  it('renders lesson count for topics', () => {
    render(<TopicClusters clusters={clusters} />);
    expect(screen.getByText('4 lessons')).toBeInTheDocument();
    expect(screen.getByText('6 lessons')).toBeInTheDocument();
  });

  it('renders progress percentage for each cluster', () => {
    render(<TopicClusters clusters={clusters} />);
    // 1 completed out of 3 = 33%
    expect(screen.getByText('33%')).toBeInTheDocument();
  });

  it('renders empty state without crashing', () => {
    expect(() => render(<TopicClusters clusters={[]} />)).not.toThrow();
  });
});
