import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { StudyAssistant } from '@/components/topic/StudyAssistant';
import type { TopicData } from '@/data/topics/types';

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

const stubData: Partial<TopicData> = {
  meta: {
    title: 'Test Topic',
    slug: 'test-topic',
    domainSlug: 'deep-learning',
    difficulty: 'intermediate',
    durationMinutes: 45,
    description: 'Test',
  },
  prerequisites: ['Linear Algebra', 'Neural Networks', 'Calculus'],
  relatedTopics: [
    { title: 'Multi-Head Attention', slug: 'multi-head-attention', domainSlug: 'deep-learning', color: '#7C3AED' },
    { title: 'Transformer Architecture', slug: 'transformer', domainSlug: 'deep-learning', color: '#0EA5E9' },
  ],
  quickRevision: [
    'Attention computes similarity between queries and keys.',
    'Scale factor prevents vanishing gradients.',
  ],
};

describe('StudyAssistant', () => {
  it('renders prerequisites heading', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    expect(screen.getByRole('heading', { name: /prerequisites/i })).toBeInTheDocument();
  });

  it('renders all prerequisites', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    expect(screen.getByText('Linear Algebra')).toBeInTheDocument();
    expect(screen.getByText('Neural Networks')).toBeInTheDocument();
    expect(screen.getByText('Calculus')).toBeInTheDocument();
  });

  it('renders related topics heading', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    expect(screen.getByRole('heading', { name: /related topics/i })).toBeInTheDocument();
  });

  it('renders related topic links', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    expect(screen.getByRole('link', { name: /multi-head attention/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /transformer architecture/i })).toBeInTheDocument();
  });

  it('related topic link has correct href', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    const link = screen.getByRole('link', { name: /multi-head attention/i });
    expect(link.getAttribute('href')).toBe('/learn/deep-learning/multi-head-attention');
  });

  it('renders quick revision heading', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    expect(screen.getByRole('heading', { name: /quick revision/i })).toBeInTheDocument();
  });

  it('renders revision points', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    expect(screen.getByText('Attention computes similarity between queries and keys.')).toBeInTheDocument();
  });

  it('renders learning actions heading', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    expect(screen.getByRole('heading', { name: /learning actions/i })).toBeInTheDocument();
  });

  it('renders Mark Complete button', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    expect(screen.getByRole('button', { name: /mark complete/i })).toBeInTheDocument();
  });

  it('Mark Complete button toggles aria-pressed', () => {
    render(<StudyAssistant data={stubData as TopicData} />);
    const btn = screen.getByRole('button', { name: /mark complete/i });
    expect(btn).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(btn);
    expect(btn).toHaveAttribute('aria-pressed', 'true');
  });
});
