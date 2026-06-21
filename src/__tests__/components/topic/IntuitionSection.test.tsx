import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { IntuitionSection } from '@/components/topic/IntuitionSection';
import type { IntuitionData } from '@/data/topics/types';

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

const intuition: IntuitionData = {
  sentence: 'It ate the apple because it was hungry.',
  tokens: [
    { id: 0, text: 'It' },
    { id: 1, text: 'ate' },
    { id: 2, text: 'the' },
    { id: 3, text: 'apple' },
  ],
  connections: [
    { fromId: 0, toId: 3, weight: 0.85, label: 'It → apple' },
  ],
  explanation: 'Each token attends to other tokens with learned weights.',
};

describe('IntuitionSection', () => {
  it('renders section heading', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    expect(screen.getByRole('heading', { name: /intuition/i })).toBeInTheDocument();
  });

  it('renders all token texts', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    expect(screen.getByText('It')).toBeInTheDocument();
    expect(screen.getByText('ate')).toBeInTheDocument();
    expect(screen.getByText('the')).toBeInTheDocument();
    expect(screen.getByText('apple')).toBeInTheDocument();
  });

  it('renders explanation text', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    expect(screen.getByText(/Each token attends to other tokens/i)).toBeInTheDocument();
  });

  it('renders animate button', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    expect(screen.getByText(/animate attention flow/i)).toBeInTheDocument();
  });

  it('toggles animation on button click', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    const animBtn = screen.getByText(/animate attention flow/i);
    fireEvent.click(animBtn);
    expect(screen.getByText(/stop animation/i)).toBeInTheDocument();
  });

  it('renders section with correct id', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    expect(document.querySelector('#intuition')).not.toBeNull();
  });

  it('hovering token with connection shows connection label', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    const itButton = screen.getByRole('button', { name: /Token: It/i });
    fireEvent.mouseEnter(itButton);
    expect(screen.getByText('It → apple')).toBeInTheDocument();
  });

  it('connection labels are not shown initially', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    // Before any hover, no connection label should be visible
    expect(screen.queryByText('It → apple')).not.toBeInTheDocument();
  });

  it('renders subtitle instruction text', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    expect(screen.getByText(/hover any word/i)).toBeInTheDocument();
  });

  it('renders "How it works:" label', () => {
    render(<IntuitionSection intuition={intuition} color="#0EA5E9" />);
    expect(screen.getByText(/how it works:/i)).toBeInTheDocument();
  });
});
