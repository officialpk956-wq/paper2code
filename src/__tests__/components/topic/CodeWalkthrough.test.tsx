import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CodeWalkthrough } from '@/components/topic/CodeWalkthrough';
import type { CodeSnippet } from '@/data/topics/types';

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

const snippets: CodeSnippet[] = [
  {
    language: 'pytorch',
    label: 'PyTorch',
    code: 'import torch\nclass Attn(torch.nn.Module):\n    pass',
    lines: [
      { lineNumbers: '1', explanation: 'Import PyTorch library.' },
      { lineNumbers: '2', explanation: 'Define attention module.' },
    ],
  },
  {
    language: 'pseudocode',
    label: 'Pseudocode',
    code: 'function attention(Q, K, V):\n  return softmax(Q @ K.T) @ V',
    lines: [
      { lineNumbers: '1', explanation: 'Function signature.' },
      { lineNumbers: '2', explanation: 'Compute scaled dot-product.' },
    ],
  },
];

describe('CodeWalkthrough', () => {
  it('renders section heading', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    expect(screen.getByRole('heading', { name: /code walkthrough/i })).toBeInTheDocument();
  });

  it('renders section with id="code"', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    expect(document.querySelector('#code')).not.toBeNull();
  });

  it('renders language tabs', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    expect(screen.getByRole('tab', { name: 'PyTorch' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Pseudocode' })).toBeInTheDocument();
  });

  it('first tab is selected by default', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    expect(screen.getByRole('tab', { name: 'PyTorch' })).toHaveAttribute('aria-selected', 'true');
  });

  it('switching tab updates selected state', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Pseudocode' }));
    expect(screen.getByRole('tab', { name: 'Pseudocode' })).toHaveAttribute('aria-selected', 'true');
  });

  it('renders line explanations', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    expect(screen.getByText('Import PyTorch library.')).toBeInTheDocument();
    expect(screen.getByText('Define attention module.')).toBeInTheDocument();
  });

  it('renders copy button', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    expect(screen.getByRole('button', { name: /copy code/i })).toBeInTheDocument();
  });

  it('renders tablist with accessible label', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    expect(screen.getByRole('tablist', { name: /code language/i })).toBeInTheDocument();
  });

  it('renders line explanation list', () => {
    render(<CodeWalkthrough snippets={snippets} />);
    expect(screen.getByRole('list', { name: /code line explanations/i })).toBeInTheDocument();
  });

  it('returns null when no snippets', () => {
    const { container } = render(<CodeWalkthrough snippets={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
