import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DomainKnowledgeGraph } from '@/components/domain/DomainKnowledgeGraph';
import type { DomainKGNode, DomainKGEdge } from '@/data/domains/types';

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

const nodes: DomainKGNode[] = [
  { id: 'n1', label: 'Neurons', x: 100, y: 100, r: 16, color: '#0EA5E9' },
  { id: 'n2', label: 'Backprop', x: 250, y: 100, r: 14, color: '#7C3AED' },
  { id: 'n3', label: 'CNNs', x: 175, y: 180, r: 14, color: '#10B981' },
];

const edges: DomainKGEdge[] = [
  { from: 'n1', to: 'n2', label: 'enables' },
  { from: 'n2', to: 'n3', label: 'used in' },
];

describe('DomainKnowledgeGraph', () => {
  it('renders section heading', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByRole('heading', { name: /knowledge graph/i })).toBeInTheDocument();
  });

  it('renders the SVG graph area', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByRole('img')).toBeInTheDocument();
  });

  it('renders all node labels', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText('Neurons')).toBeInTheDocument();
    expect(screen.getByText('Backprop')).toBeInTheDocument();
    expect(screen.getByText('CNNs')).toBeInTheDocument();
  });

  it('renders nodes as interactive buttons', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    const btns = screen.getAllByRole('button');
    expect(btns.length).toBe(3);
  });

  it('nodes respond to hover (mouse enter)', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    const btn = screen.getByRole('button', { name: /neurons/i });
    expect(() => fireEvent.mouseEnter(btn)).not.toThrow();
  });

  it('nodes respond to focus', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    const btn = screen.getByRole('button', { name: /neurons/i });
    fireEvent.focus(btn);
    fireEvent.blur(btn);
  });

  it('renders "Full Graph" link to /explorer', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    const links = screen.getAllByRole('link');
    const explorerLinks = links.filter(l => l.getAttribute('href') === '/explorer');
    expect(explorerLinks.length).toBeGreaterThan(0);
  });

  it('renders concept relationships subtitle', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText(/concept relationships/i)).toBeInTheDocument();
  });

  it('nodes support keyboard Enter/Space', () => {
    render(<DomainKnowledgeGraph nodes={nodes} edges={edges} domainSlug="deep-learning" color="#0EA5E9" />);
    const btn = screen.getByRole('button', { name: /neurons/i });
    expect(() => fireEvent.keyDown(btn, { key: 'Enter' })).not.toThrow();
    expect(() => fireEvent.keyDown(btn, { key: ' ' })).not.toThrow();
  });

  it('renders empty nodes without crashing', () => {
    expect(() => render(<DomainKnowledgeGraph nodes={[]} edges={[]} domainSlug="deep-learning" color="#0EA5E9" />)).not.toThrow();
  });
});
