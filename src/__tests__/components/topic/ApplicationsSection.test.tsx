import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ApplicationsSection } from '@/components/topic/ApplicationsSection';
import type { ApplicationCard } from '@/data/topics/types';

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

const applications: ApplicationCard[] = [
  {
    name: 'GPT',
    description: 'Language model by OpenAI.',
    whereUsed: 'Text generation and chat.',
    whyUseful: 'Scales attention to long context.',
    color: '#10B981',
    icon: '🤖',
  },
  {
    name: 'BERT',
    description: 'Bidirectional encoder model.',
    whereUsed: 'Search and NLU tasks.',
    whyUseful: 'Full context understanding.',
    color: '#0EA5E9',
    icon: '🔍',
  },
];

describe('ApplicationsSection', () => {
  it('renders section heading', () => {
    render(<ApplicationsSection applications={applications} />);
    expect(screen.getByRole('heading', { name: /real-world applications/i })).toBeInTheDocument();
  });

  it('renders section with id="applications"', () => {
    render(<ApplicationsSection applications={applications} />);
    expect(document.querySelector('#applications')).not.toBeNull();
  });

  it('renders application names', () => {
    render(<ApplicationsSection applications={applications} />);
    expect(screen.getByText('GPT')).toBeInTheDocument();
    expect(screen.getByText('BERT')).toBeInTheDocument();
  });

  it('renders descriptions', () => {
    render(<ApplicationsSection applications={applications} />);
    expect(screen.getByText('Language model by OpenAI.')).toBeInTheDocument();
  });

  it('renders "Where" labels', () => {
    render(<ApplicationsSection applications={applications} />);
    const whereLabels = screen.getAllByText('Where:');
    expect(whereLabels.length).toBe(2);
  });

  it('renders "Why" labels', () => {
    render(<ApplicationsSection applications={applications} />);
    const whyLabels = screen.getAllByText('Why:');
    expect(whyLabels.length).toBe(2);
  });

  it('renders whereUsed text', () => {
    render(<ApplicationsSection applications={applications} />);
    expect(screen.getByText('Text generation and chat.')).toBeInTheDocument();
  });

  it('renders whyUseful text', () => {
    render(<ApplicationsSection applications={applications} />);
    expect(screen.getByText('Scales attention to long context.')).toBeInTheDocument();
  });

  it('renders icons', () => {
    render(<ApplicationsSection applications={applications} />);
    expect(screen.getByText('🤖')).toBeInTheDocument();
    expect(screen.getByText('🔍')).toBeInTheDocument();
  });

  it('renders list with aria-label', () => {
    render(<ApplicationsSection applications={applications} />);
    expect(screen.getByRole('list', { name: /application examples/i })).toBeInTheDocument();
  });
});
