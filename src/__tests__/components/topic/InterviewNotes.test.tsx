import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { InterviewNotes } from '@/components/topic/InterviewNotes';
import type { InterviewQuestion } from '@/data/topics/types';

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

const questions: InterviewQuestion[] = [
  {
    id: 'q1',
    question: 'Why is attention better than RNNs?',
    answer: 'Attention provides direct token-to-token connections without sequential processing.',
    tags: ['architecture', 'scalability'],
  },
  {
    id: 'q2',
    question: 'What is the complexity of self-attention?',
    answer: 'O(n²·d) due to computing pairwise attention scores for all token pairs.',
    tags: ['complexity'],
  },
];

describe('InterviewNotes', () => {
  it('renders section heading', () => {
    render(<InterviewNotes questions={questions} />);
    expect(screen.getByRole('heading', { name: /interview notes/i })).toBeInTheDocument();
  });

  it('renders section with id="interview"', () => {
    render(<InterviewNotes questions={questions} />);
    expect(document.querySelector('#interview')).not.toBeNull();
  });

  it('renders all question texts', () => {
    render(<InterviewNotes questions={questions} />);
    expect(screen.getByText('Why is attention better than RNNs?')).toBeInTheDocument();
    expect(screen.getByText('What is the complexity of self-attention?')).toBeInTheDocument();
  });

  it('answers are hidden by default', () => {
    render(<InterviewNotes questions={questions} />);
    expect(screen.queryByText(/direct token-to-token/i)).not.toBeInTheDocument();
  });

  it('clicking a question reveals the answer', () => {
    render(<InterviewNotes questions={questions} />);
    fireEvent.click(screen.getByRole('button', { name: /why is attention better/i }));
    expect(screen.getByText(/direct token-to-token/i)).toBeInTheDocument();
  });

  it('renders aria-expanded=false initially', () => {
    render(<InterviewNotes questions={questions} />);
    const btn = screen.getByRole('button', { name: /why is attention better/i });
    expect(btn).toHaveAttribute('aria-expanded', 'false');
  });

  it('re-queried button has aria-expanded=true after click', () => {
    render(<InterviewNotes questions={questions} />);
    fireEvent.click(screen.getByRole('button', { name: /why is attention better/i }));
    // Re-query to get fresh DOM reference
    const btnAfter = screen.getByRole('button', { name: /why is attention better/i });
    expect(btnAfter).toHaveAttribute('aria-expanded', 'true');
  });

  it('renders answer tags when expanded', () => {
    render(<InterviewNotes questions={questions} />);
    fireEvent.click(screen.getByRole('button', { name: /why is attention better/i }));
    expect(screen.getByText('architecture')).toBeInTheDocument();
    expect(screen.getByText('scalability')).toBeInTheDocument();
  });

  it('renders list with aria-label', () => {
    render(<InterviewNotes questions={questions} />);
    expect(screen.getByRole('list', { name: /interview questions/i })).toBeInTheDocument();
  });

  it('two questions have separate open states', () => {
    render(<InterviewNotes questions={questions} />);
    // Open q1
    fireEvent.click(screen.getByRole('button', { name: /why is attention better/i }));
    // q2 answer not visible
    expect(screen.queryByText(/pairwise attention scores/i)).not.toBeInTheDocument();
  });
});
