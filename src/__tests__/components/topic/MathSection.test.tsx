import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MathSection } from '@/components/topic/MathSection';
import type { Formula } from '@/data/topics/types';

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

vi.mock('katex', () => ({
  renderToString: (latex: string) => `<span data-testid="katex">${latex}</span>`,
}));

const formulas: Formula[] = [
  {
    id: 'scaled-dot',
    name: 'Scaled Dot-Product Attention',
    latex: '\\text{Attention}(Q,K,V)',
    intuition: 'Compute attention weights from queries and keys.',
    variables: [
      { symbol: 'Q', meaning: 'Queries matrix' },
      { symbol: 'K', meaning: 'Keys matrix' },
    ],
    derivationSteps: ['Step 1: compute QK^T', 'Step 2: scale by sqrt(d_k)'],
  },
  {
    id: 'softmax',
    name: 'Softmax',
    latex: '\\text{softmax}(x)',
    intuition: 'Normalizes scores to probabilities.',
    variables: [{ symbol: 'x', meaning: 'Input vector' }],
  },
];

describe('MathSection', () => {
  it('renders section heading', () => {
    render(<MathSection formulas={formulas} />);
    expect(screen.getByRole('heading', { name: /mathematics/i })).toBeInTheDocument();
  });

  it('renders all formula names', () => {
    render(<MathSection formulas={formulas} />);
    expect(screen.getByText('Scaled Dot-Product Attention')).toBeInTheDocument();
    expect(screen.getByText('Softmax')).toBeInTheDocument();
  });

  it('renders section with id="mathematics"', () => {
    render(<MathSection formulas={formulas} />);
    expect(document.querySelector('#mathematics')).not.toBeNull();
  });

  it('expands formula on click — shows variable list', () => {
    render(<MathSection formulas={formulas} />);
    const toggle = screen.getByRole('button', { name: /scaled dot-product/i });
    fireEvent.click(toggle);
    expect(screen.getByText('Queries matrix')).toBeInTheDocument();
  });

  it('shows variable symbols in code elements when expanded', () => {
    render(<MathSection formulas={formulas} />);
    fireEvent.click(screen.getByRole('button', { name: /scaled dot-product/i }));
    // The code elements with symbols should be present
    expect(screen.getByLabelText('Variable: Q')).toBeInTheDocument();
    expect(screen.getByLabelText('Variable: K')).toBeInTheDocument();
  });

  it('shows expanded content below the toggle — variable breakdown', () => {
    render(<MathSection formulas={formulas} />);
    fireEvent.click(screen.getByRole('button', { name: /scaled dot-product/i }));
    expect(screen.getByRole('list', { name: /formula variables/i })).toBeInTheDocument();
  });

  it('shows derivation toggle when formula has steps', () => {
    render(<MathSection formulas={formulas} />);
    fireEvent.click(screen.getByRole('button', { name: /scaled dot-product/i }));
    expect(screen.getByText(/derivation steps/i)).toBeInTheDocument();
  });

  it('does not show derivation toggle when no steps', () => {
    render(<MathSection formulas={formulas} />);
    fireEvent.click(screen.getByRole('button', { name: /softmax/i }));
    expect(screen.queryByText(/derivation steps/i)).not.toBeInTheDocument();
  });

  it('toggle button has aria-expanded=false initially', () => {
    render(<MathSection formulas={formulas} />);
    const toggle = screen.getByRole('button', { name: /scaled dot-product/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });

  it('renders formula list with aria-label', () => {
    render(<MathSection formulas={formulas} />);
    expect(screen.getByRole('list', { name: /mathematical formulas/i })).toBeInTheDocument();
  });
});
