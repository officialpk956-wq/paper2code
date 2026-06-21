import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ContinueLearningCard } from '@/components/learn/ContinueLearningCard';
import type { ContinueLearningData } from '@/data/learn/recommendations';

vi.mock('framer-motion', () => ({
  motion: {
    section: ({ children, ...props }: React.HTMLAttributes<HTMLElement>) => <section {...props}>{children}</section>,
    div:     ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
  },
}));

const MOCK_DATA: ContinueLearningData = {
  id: 'test-lesson',
  title: 'Multi-Head Attention',
  subtitle: 'Deep Learning · Transformers',
  domain: 'Deep Learning',
  progress: 62,
  minutesRemaining: 18,
  totalMinutes: 45,
  href: '/learn/deep-learning/mha',
  color: '#06B6D4',
  lastAccessedAt: '2026-06-19T14:32:00Z',
};

describe('ContinueLearningCard', () => {
  it('renders the lesson title', () => {
    render(<ContinueLearningCard data={MOCK_DATA} />);
    expect(screen.getByText('Multi-Head Attention')).toBeInTheDocument();
  });

  it('renders the subtitle', () => {
    render(<ContinueLearningCard data={MOCK_DATA} />);
    expect(screen.getByText('Deep Learning · Transformers')).toBeInTheDocument();
  });

  it('renders the progress percentage', () => {
    render(<ContinueLearningCard data={MOCK_DATA} />);
    expect(screen.getByText('62%')).toBeInTheDocument();
  });

  it('renders minutes remaining', () => {
    render(<ContinueLearningCard data={MOCK_DATA} />);
    expect(screen.getByText('18 min left')).toBeInTheDocument();
  });

  it('renders the Resume link with correct href', () => {
    render(<ContinueLearningCard data={MOCK_DATA} />);
    const link = screen.getByRole('link', { name: /resume/i });
    expect(link).toHaveAttribute('href', '/learn/deep-learning/mha');
  });

  it('renders the section label', () => {
    render(<ContinueLearningCard data={MOCK_DATA} />);
    expect(screen.getByText('Continue Learning')).toBeInTheDocument();
  });

  it('has accessible aria-label on the section', () => {
    render(<ContinueLearningCard data={MOCK_DATA} />);
    const section = screen.getByRole('region', { name: /continue learning.*multi-head attention/i });
    expect(section).toBeInTheDocument();
  });

  it('clamps progress to 100 for over-limit values', () => {
    render(<ContinueLearningCard data={{ ...MOCK_DATA, progress: 150 }} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('clamps progress to 0 for negative values', () => {
    render(<ContinueLearningCard data={{ ...MOCK_DATA, progress: -10 }} />);
    expect(screen.getByText('0%')).toBeInTheDocument();
  });
});
