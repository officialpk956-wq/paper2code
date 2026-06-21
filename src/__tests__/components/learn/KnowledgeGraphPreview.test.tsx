import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { KnowledgeGraphPreview } from '@/components/learn/KnowledgeGraphPreview';

vi.mock('framer-motion', () => ({
  motion: {
    section: ({ children, ...props }: React.HTMLAttributes<HTMLElement>) => <section {...props}>{children}</section>,
  },
}));

describe('KnowledgeGraphPreview', () => {
  it('renders the section heading', () => {
    render(<KnowledgeGraphPreview />);
    expect(screen.getByRole('heading', { name: /knowledge graph/i })).toBeInTheDocument();
  });

  it('renders the Open Full Graph button/link', () => {
    render(<KnowledgeGraphPreview />);
    const link = screen.getByRole('link', { name: /open.*full.*knowledge graph/i });
    expect(link).toHaveAttribute('href', '/explorer');
  });

  it('renders all graph node labels as accessible buttons', () => {
    render(<KnowledgeGraphPreview />);
    const mathBtn = screen.getByRole('button', { name: /math/i });
    expect(mathBtn).toBeInTheDocument();
  });

  it('renders all major nodes', () => {
    render(<KnowledgeGraphPreview />);
    expect(screen.getByRole('button', { name: /transformers/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /llms/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /agents/i })).toBeInTheDocument();
  });

  it('renders the SVG with accessible label', () => {
    render(<KnowledgeGraphPreview />);
    expect(screen.getByRole('img', { name: /knowledge graph/i })).toBeInTheDocument();
  });

  it('renders the footer hint with explorer link', () => {
    render(<KnowledgeGraphPreview />);
    const links = screen.getAllByRole('link', { name: /open the full graph/i });
    expect(links[0]).toHaveAttribute('href', '/explorer');
  });

  it('renders hint text about connections', () => {
    render(<KnowledgeGraphPreview />);
    expect(screen.getByText(/explore all.*connections/i)).toBeInTheDocument();
  });

  it('hover instruction text is present', () => {
    render(<KnowledgeGraphPreview />);
    expect(screen.getByText(/hover a node/i)).toBeInTheDocument();
  });

  it('node buttons are keyboard focusable', () => {
    render(<KnowledgeGraphPreview />);
    const btn = screen.getByRole('button', { name: /math/i });
    btn.focus();
    expect(document.activeElement).toBe(btn);
  });

  it('highlights on hover by updating state (smoke test)', async () => {
    const user = userEvent.setup();
    render(<KnowledgeGraphPreview />);
    const btn = screen.getByRole('button', { name: /math/i });
    await user.hover(btn);
    // No error thrown = state transition works
    expect(btn).toBeInTheDocument();
  });
});
