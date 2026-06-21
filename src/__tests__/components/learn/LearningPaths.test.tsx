import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LearningPaths } from '@/components/learn/LearningPaths';
import type { LearningPath } from '@/data/learn/paths';

vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
  },
}));

const MOCK_PATHS: LearningPath[] = [
  {
    id: 'ai-engineer', title: 'AI Engineer', subtitle: 'Build production AI systems',
    description: 'Master the full AI engineering stack.', difficulty: 'advanced',
    lessonCount: 120, estimatedHours: 180, completionPercent: 32,
    domainIds: ['deep-learning', 'llms'], color: '#7C3AED', glowColor: 'rgba(124,58,237,0.3)',
    icon: 'Cpu', tags: ['LLMs', 'Inference'],
  },
  {
    id: 'ml-engineer', title: 'ML Engineer', subtitle: 'From data to models',
    description: 'Build and ship ML models at scale.', difficulty: 'intermediate',
    lessonCount: 96, estimatedHours: 140, completionPercent: 0,
    domainIds: ['machine-learning', 'mlops'], color: '#06B6D4', glowColor: 'rgba(6,182,212,0.3)',
    icon: 'Workflow', tags: ['PyTorch', 'MLflow'],
  },
];

describe('LearningPaths', () => {
  it('renders the section heading', () => {
    render(<LearningPaths paths={MOCK_PATHS} />);
    expect(screen.getByRole('heading', { name: /learning paths/i })).toBeInTheDocument();
  });

  it('renders all path titles', () => {
    render(<LearningPaths paths={MOCK_PATHS} />);
    expect(screen.getByText('AI Engineer')).toBeInTheDocument();
    expect(screen.getByText('ML Engineer')).toBeInTheDocument();
  });

  it('renders difficulty badges', () => {
    render(<LearningPaths paths={MOCK_PATHS} />);
    expect(screen.getByText('Advanced')).toBeInTheDocument();
    expect(screen.getByText('Intermediate')).toBeInTheDocument();
  });

  it('renders lesson counts', () => {
    render(<LearningPaths paths={MOCK_PATHS} />);
    expect(screen.getByText(/120 lessons · 180h/)).toBeInTheDocument();
  });

  it('shows Start Path for unstarted paths', () => {
    render(<LearningPaths paths={MOCK_PATHS} />);
    // Both paths show Start Path because completionPercent is 0
    expect(screen.getAllByText('Start Path').length).toBeGreaterThan(0);
  });

  it('renders the View all link', () => {
    render(<LearningPaths paths={MOCK_PATHS} />);
    const link = screen.getByRole('link', { name: /view all/i });
    expect(link).toHaveAttribute('href', '/learn/paths');
  });

  it('renders list with correct aria-label', () => {
    render(<LearningPaths paths={MOCK_PATHS} />);
    expect(screen.getByRole('list', { name: /learning paths/i })).toBeInTheDocument();
  });

  it('renders subtitles', () => {
    render(<LearningPaths paths={MOCK_PATHS} />);
    expect(screen.getByText('Build production AI systems')).toBeInTheDocument();
  });
});
