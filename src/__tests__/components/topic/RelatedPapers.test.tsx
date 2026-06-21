import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RelatedPapers } from '@/components/topic/RelatedPapers';
import type { TopicPaper } from '@/data/topics/types';

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

const papers: TopicPaper[] = [
  {
    id: 'attention-all-you-need',
    title: 'Attention Is All You Need',
    authors: 'Vaswani et al.',
    year: 2017,
    citations: 85000,
    impact: 'landmark',
    description: 'The original Transformer paper.',
    color: '#7C3AED',
    href: 'https://arxiv.org/abs/1706.03762',
  },
  {
    id: 'bert',
    title: 'BERT',
    authors: 'Devlin et al.',
    year: 2019,
    citations: 55000,
    impact: 'high',
    description: 'Bidirectional pre-training for NLU.',
    color: '#0EA5E9',
    href: 'https://arxiv.org/abs/1810.04805',
  },
];

describe('RelatedPapers', () => {
  it('renders section heading', () => {
    render(<RelatedPapers papers={papers} />);
    expect(screen.getByRole('heading', { name: /related papers/i })).toBeInTheDocument();
  });

  it('renders section with id="papers"', () => {
    render(<RelatedPapers papers={papers} />);
    expect(document.querySelector('#papers')).not.toBeNull();
  });

  it('renders paper titles', () => {
    render(<RelatedPapers papers={papers} />);
    expect(screen.getByText('Attention Is All You Need')).toBeInTheDocument();
    expect(screen.getByText('BERT')).toBeInTheDocument();
  });

  it('renders authors', () => {
    render(<RelatedPapers papers={papers} />);
    expect(screen.getByText('Vaswani et al.')).toBeInTheDocument();
    expect(screen.getByText('Devlin et al.')).toBeInTheDocument();
  });

  it('renders years', () => {
    render(<RelatedPapers papers={papers} />);
    expect(screen.getByText('2017')).toBeInTheDocument();
    expect(screen.getByText('2019')).toBeInTheDocument();
  });

  it('renders impact badges', () => {
    render(<RelatedPapers papers={papers} />);
    expect(screen.getByText('Landmark')).toBeInTheDocument();
    expect(screen.getByText('High Impact')).toBeInTheDocument();
  });

  it('renders formatted citation counts', () => {
    render(<RelatedPapers papers={papers} />);
    expect(screen.getByText(/85k/)).toBeInTheDocument();
  });

  it('renders study links', () => {
    render(<RelatedPapers papers={papers} />);
    const links = screen.getAllByRole('link', { name: /study paper/i });
    expect(links.length).toBe(2);
  });

  it('study link has correct href', () => {
    render(<RelatedPapers papers={papers} />);
    const link = screen.getByRole('link', { name: /study paper: attention is all you need/i });
    expect(link.getAttribute('href')).toBe('https://arxiv.org/abs/1706.03762');
  });

  it('renders paper list with aria-label', () => {
    render(<RelatedPapers papers={papers} />);
    expect(screen.getByRole('list', { name: /related research papers/i })).toBeInTheDocument();
  });
});
