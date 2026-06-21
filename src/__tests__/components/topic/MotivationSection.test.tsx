import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MotivationSection } from '@/components/topic/MotivationSection';
import type { MotivationCard } from '@/data/topics/types';

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

const cards: MotivationCard[] = [
  { type: 'problem',    title: 'Sequential Bottleneck', body: 'RNNs process tokens one by one.', icon: 'AlertTriangle' },
  { type: 'limitation', title: 'Vanishing Context',     body: 'Long sequences lose context.',   icon: 'TrendingDown' },
  { type: 'solution',   title: 'Direct Connections',    body: 'Attention connects any two tokens.', icon: 'Zap' },
];

describe('MotivationSection', () => {
  it('renders section heading', () => {
    render(<MotivationSection cards={cards} />);
    expect(screen.getByRole('heading', { name: /motivation/i })).toBeInTheDocument();
  });

  it('renders all card titles', () => {
    render(<MotivationSection cards={cards} />);
    expect(screen.getByText('Sequential Bottleneck')).toBeInTheDocument();
    expect(screen.getByText('Vanishing Context')).toBeInTheDocument();
    expect(screen.getByText('Direct Connections')).toBeInTheDocument();
  });

  it('renders all card bodies', () => {
    render(<MotivationSection cards={cards} />);
    expect(screen.getByText(/RNNs process tokens one by one/i)).toBeInTheDocument();
  });

  it('renders type labels', () => {
    render(<MotivationSection cards={cards} />);
    expect(screen.getByText('Problem')).toBeInTheDocument();
    expect(screen.getByText('Limitation')).toBeInTheDocument();
    expect(screen.getByText('Solution')).toBeInTheDocument();
  });

  it('renders with role="list"', () => {
    render(<MotivationSection cards={cards} />);
    expect(screen.getByRole('list', { name: /problem, limitation/i })).toBeInTheDocument();
  });

  it('renders section with correct id', () => {
    render(<MotivationSection cards={cards} />);
    const section = document.querySelector('#motivation');
    expect(section).not.toBeNull();
  });

  it('renders 3 list items', () => {
    render(<MotivationSection cards={cards} />);
    expect(screen.getAllByRole('listitem').length).toBe(3);
  });

  it('renders subtitle text', () => {
    render(<MotivationSection cards={cards} />);
    expect(screen.getByText(/why was this invented/i)).toBeInTheDocument();
  });

  it('renders empty list gracefully', () => {
    render(<MotivationSection cards={[]} />);
    expect(screen.getByRole('heading', { name: /motivation/i })).toBeInTheDocument();
  });

  it('renders solution card body text', () => {
    render(<MotivationSection cards={cards} />);
    expect(screen.getByText(/Attention connects any two tokens/i)).toBeInTheDocument();
  });
});
