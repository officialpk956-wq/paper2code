import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ProjectShowcase } from '@/components/domain/ProjectShowcase';
import type { DomainProject } from '@/data/domains/types';

vi.mock('framer-motion', () => ({
  motion: new Proxy({}, {
    get: (_t, tag) => {
      const React = require('react');
      return ({ children, ...rest }: React.HTMLAttributes<HTMLElement>) =>
        React.createElement(tag as string, rest, children);
    },
  }),
  AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
  useMotionValue: () => ({ set: vi.fn(), get: vi.fn(() => 0) }),
  useSpring: () => ({ set: vi.fn(), get: vi.fn(() => 0) }),
}));

const projects: DomainProject[] = [
  {
    id: 'p1',
    title: 'CIFAR-10 Classifier',
    description: 'Train a CNN on CIFAR-10',
    skills: ['PyTorch', 'CNNs', 'Data Augmentation'],
    difficulty: 'beginner',
    estimatedHours: 8,
    color: '#0EA5E9',
    href: '/projects/cifar-10',
  },
  {
    id: 'p2',
    title: 'GPT-2 from Scratch',
    description: 'Implement GPT-2 architecture',
    skills: ['Transformers', 'PyTorch', 'Tokenization'],
    difficulty: 'advanced',
    estimatedHours: 24,
    color: '#7C3AED',
    href: '/projects/gpt2',
  },
];

describe('ProjectShowcase', () => {
  it('renders section heading', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    expect(screen.getByRole('heading', { name: /projects/i })).toBeInTheDocument();
  });

  it('renders all project titles', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    expect(screen.getByText('CIFAR-10 Classifier')).toBeInTheDocument();
    expect(screen.getByText('GPT-2 from Scratch')).toBeInTheDocument();
  });

  it('renders project descriptions', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    expect(screen.getByText('Train a CNN on CIFAR-10')).toBeInTheDocument();
  });

  it('renders difficulty badges', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    expect(screen.getByText('Beginner')).toBeInTheDocument();
    expect(screen.getByText('Advanced')).toBeInTheDocument();
  });

  it('renders estimated hours', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    expect(screen.getByText('8h')).toBeInTheDocument();
    expect(screen.getByText('24h')).toBeInTheDocument();
  });

  it('renders skill tags', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    expect(screen.getAllByText('PyTorch').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('CNNs')).toBeInTheDocument();
    expect(screen.getByText('Transformers')).toBeInTheDocument();
  });

  it('links each project card to its href', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    const cifar = screen.getByRole('link', { name: /cifar-10 classifier/i });
    expect(cifar.getAttribute('href')).toBe('/projects/cifar-10');
  });

  it('renders "View all" link with correct domain', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    const viewAll = screen.getByRole('link', { name: /view all/i });
    expect(viewAll.getAttribute('href')).toBe('/learn/deep-learning/projects');
  });

  it('renders empty state when no projects', () => {
    render(<ProjectShowcase projects={[]} domainSlug="deep-learning" />);
    expect(screen.getByText(/no projects available/i)).toBeInTheDocument();
  });

  it('renders tilt container for each project card', () => {
    render(<ProjectShowcase projects={projects} domainSlug="deep-learning" />);
    const listItems = screen.getAllByRole('listitem');
    expect(listItems.length).toBe(2);
  });
});
