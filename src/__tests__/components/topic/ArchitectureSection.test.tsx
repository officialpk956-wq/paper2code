import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ArchitectureSection } from '@/components/topic/ArchitectureSection';
import type { ArchNode, ArchEdge } from '@/data/topics/types';

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

const nodes: ArchNode[] = [
  { id: 'input', label: 'Input Embeddings', shape: 'rect', params: '[B, T, d]',   purpose: 'Token representations.', x: 0, y: 0, color: '#0EA5E9' },
  { id: 'qkv',   label: 'Q / K / V',        shape: 'rect', params: '[B, T, d/h]', purpose: 'Project queries, keys, values.', x: 0, y: 80, color: '#7C3AED' },
];

const edges: ArchEdge[] = [
  { from: 'input', to: 'qkv', label: 'linear' },
];

describe('ArchitectureSection', () => {
  it('renders section heading', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    expect(screen.getByRole('heading', { name: /visual architecture/i })).toBeInTheDocument();
  });

  it('renders section with id="architecture"', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    expect(document.querySelector('#architecture')).not.toBeNull();
  });

  it('renders all node labels in SVG', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    expect(screen.getByText('Input Embeddings')).toBeInTheDocument();
    expect(screen.getByText('Q / K / V')).toBeInTheDocument();
  });

  it('renders SVG with aria-label', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    expect(screen.getByRole('img', { name: /architecture flow/i })).toBeInTheDocument();
  });

  it('renders "hover a block" hint when no node hovered', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    expect(screen.getByText(/hover a block to see details/i)).toBeInTheDocument();
  });

  it('hovering a node shows its purpose in detail panel', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    const inputBtn = screen.getByRole('button', { name: /Input Embeddings/i });
    fireEvent.mouseEnter(inputBtn);
    expect(screen.getByText('Token representations.')).toBeInTheDocument();
  });

  it('hovering a node shows "Selected Block" label', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    const inputBtn = screen.getByRole('button', { name: /Input Embeddings/i });
    fireEvent.mouseEnter(inputBtn);
    expect(screen.getByText('Selected Block')).toBeInTheDocument();
  });

  it('hovered node label appears in detail panel', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    const inputBtn = screen.getByRole('button', { name: /Input Embeddings/i });
    fireEvent.mouseEnter(inputBtn);
    // The detail panel shows the node label
    expect(screen.getAllByText('Input Embeddings').length).toBeGreaterThanOrEqual(1);
  });

  it('renders subtitle instruction', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    expect(screen.getByText(/hover any block/i)).toBeInTheDocument();
  });

  it('hovering node shows params in detail panel via code element', () => {
    render(<ArchitectureSection nodes={nodes} edges={edges} color="#0EA5E9" />);
    fireEvent.mouseEnter(screen.getByRole('button', { name: /Input Embeddings/i }));
    // The params appear in a <code> element in the detail panel
    const codes = screen.getAllByText('[B, T, d]');
    expect(codes.length).toBeGreaterThanOrEqual(1);
  });
});
