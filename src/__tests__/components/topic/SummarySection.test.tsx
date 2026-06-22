import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SummarySection } from '@/components/topic/SummarySection';
import type { SummaryPoint } from '@/data/topics/types';

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

const points: SummaryPoint[] = [
  { icon: '🎯', text: 'compares every token to every other token.', emphasis: 'Attention' },
  { icon: '⚡', text: 'Scale is sqrt(d_k) to prevent gradient issues.', emphasis: '' },
  { icon: '🔀', text: 'let the model attend to different subspaces.', emphasis: 'Multiple heads' },
  { icon: '🚀', text: 'Transformers replaced RNNs at scale.', emphasis: '' },
];

describe('SummarySection', () => {
  it('renders section heading', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getByRole('heading', { name: /summary/i })).toBeInTheDocument();
  });

  it('renders section with id="summary"', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(document.querySelector('#summary')).not.toBeNull();
  });

  it('renders all icons', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getByText('🎯')).toBeInTheDocument();
    expect(screen.getByText('⚡')).toBeInTheDocument();
  });

  it('renders all body text', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getByText(/compares every token to every other token/i)).toBeInTheDocument();
  });

  it('renders emphasis text in bold', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getByText('Attention')).toBeInTheDocument();
    expect(screen.getByText('Multiple heads')).toBeInTheDocument();
  });

  it('renders "Chapter Complete" banner', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getByText('Chapter Complete')).toBeInTheDocument();
  });

  it('renders list with aria-label', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getByRole('list', { name: /key takeaways/i })).toBeInTheDocument();
  });

  it('renders correct number of list items', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getAllByRole('listitem').length).toBe(4);
  });

  it('renders subtitle text', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getByText(/key takeaways from this chapter/i)).toBeInTheDocument();
  });

  it('renders banner description text', () => {
    render(<SummarySection points={points} color="#0EA5E9" />);
    expect(screen.getByText(/you've covered motivation/i)).toBeInTheDocument();
  });
});
