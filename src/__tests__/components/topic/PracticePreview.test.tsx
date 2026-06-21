import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PracticePreview } from '@/components/topic/PracticePreview';
import type { PracticeQuestion } from '@/data/topics/types';

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

const questions: PracticeQuestion[] = [
  {
    id: 'pq1',
    type: 'mcq',
    prompt: 'What is the time complexity of self-attention?',
    options: [
      { id: 'a', text: 'O(n)' },
      { id: 'b', text: 'O(n squared)' },
      { id: 'c', text: 'O(n log n)' },
    ],
    correctAnswer: 'b',
    hint: 'Think about pairwise comparisons.',
  },
  {
    id: 'pq2',
    type: 'math',
    prompt: 'If d_model=512 and h=8, what is d_k?',
    hint: 'd_k = d_model / h',
  },
];

describe('PracticePreview', () => {
  it('renders section heading', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    expect(screen.getByRole('heading', { name: /practice/i })).toBeInTheDocument();
  });

  it('renders section with id="practice"', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    expect(document.querySelector('#practice')).not.toBeNull();
  });

  it('renders MCQ prompt', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    expect(screen.getByText('What is the time complexity of self-attention?')).toBeInTheDocument();
  });

  it('renders MCQ options', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    expect(screen.getByText('O(n squared)')).toBeInTheDocument();
    expect(screen.getByText('O(n)')).toBeInTheDocument();
  });

  it('renders MCQ type badge', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    expect(screen.getByText('MCQ')).toBeInTheDocument();
  });

  it('renders Math type badge', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    expect(screen.getByText('Math')).toBeInTheDocument();
  });

  it('renders MCQ options with radio role', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    const radios = screen.getAllByRole('radio');
    expect(radios.length).toBe(3);
  });

  it('renders "Full Practice" link', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    expect(screen.getByRole('link', { name: /full practice/i })).toBeInTheDocument();
  });

  it('Full Practice link has dojo href', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    const link = screen.getByRole('link', { name: /full practice/i });
    expect(link.getAttribute('href')).toContain('/dojo');
  });

  it('renders practice list with aria-label', () => {
    render(<PracticePreview questions={questions} domainSlug="deep-learning" topicSlug="attention" />);
    expect(screen.getByRole('list', { name: /practice questions preview/i })).toBeInTheDocument();
  });
});
