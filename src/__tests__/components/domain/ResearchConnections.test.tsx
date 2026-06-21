import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ResearchConnections } from '@/components/domain/ResearchConnections';
import type { DomainPaper } from '@/data/domains/types';

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

const papers: DomainPaper[] = [
  {
    id: 'p1',
    title: 'Attention Is All You Need',
    authors: 'Vaswani et al.',
    year: 2017,
    citations: 85000,
    impact: 'landmark',
    description: 'Introduced the Transformer architecture.',
    color: '#0EA5E9',
    href: '/papers/attention-is-all-you-need',
  },
  {
    id: 'p2',
    title: 'Deep Residual Learning',
    authors: 'He et al.',
    year: 2015,
    citations: 120000,
    impact: 'high',
    description: 'Introduced skip connections in deep networks.',
    color: '#10B981',
    href: '/papers/deep-residual',
  },
];

describe('ResearchConnections', () => {
  it('renders section heading', () => {
    render(<ResearchConnections papers={papers} />);
    expect(screen.getByRole('heading', { name: /research connections/i })).toBeInTheDocument();
  });

  it('renders all paper titles', () => {
    render(<ResearchConnections papers={papers} />);
    expect(screen.getByText('Attention Is All You Need')).toBeInTheDocument();
    expect(screen.getByText('Deep Residual Learning')).toBeInTheDocument();
  });

  it('renders author names', () => {
    render(<ResearchConnections papers={papers} />);
    expect(screen.getByText('Vaswani et al.')).toBeInTheDocument();
    expect(screen.getByText('He et al.')).toBeInTheDocument();
  });

  it('renders publication years', () => {
    render(<ResearchConnections papers={papers} />);
    expect(screen.getByText('2017')).toBeInTheDocument();
    expect(screen.getByText('2015')).toBeInTheDocument();
  });

  it('renders impact badges', () => {
    render(<ResearchConnections papers={papers} />);
    expect(screen.getByText('Landmark')).toBeInTheDocument();
    expect(screen.getByText('High Impact')).toBeInTheDocument();
  });

  it('renders citation counts formatted', () => {
    render(<ResearchConnections papers={papers} />);
    expect(screen.getByText(/85k citations/i)).toBeInTheDocument();
    expect(screen.getByText(/120k citations/i)).toBeInTheDocument();
  });

  it('renders "Study Paper" CTA for each paper', () => {
    render(<ResearchConnections papers={papers} />);
    const studyLinks = screen.getAllByRole('link', { name: /study paper/i });
    expect(studyLinks.length).toBe(2);
  });

  it('"Study Paper" links point to correct href', () => {
    render(<ResearchConnections papers={papers} />);
    const attentionLink = screen.getByRole('link', { name: /study paper: attention is all you need/i });
    expect(attentionLink.getAttribute('href')).toBe('/papers/attention-is-all-you-need');
  });

  it('renders "Browse all papers" link', () => {
    render(<ResearchConnections papers={papers} />);
    expect(screen.getByRole('link', { name: /browse all papers/i })).toBeInTheDocument();
  });

  it('renders empty state when no papers', () => {
    render(<ResearchConnections papers={[]} />);
    expect(screen.getByText(/no papers linked/i)).toBeInTheDocument();
  });
});
