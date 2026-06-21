import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProgressOverview } from '@/components/domain/ProgressOverview';
import type { DomainProgress } from '@/data/domains/types';

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

const mockProgress: DomainProgress = {
  lessonsCompleted: 24,
  totalLessons: 72,
  currentStreak: 5,
  estimatedHoursRemaining: 44,
  masteryPercent: 34,
};

describe('ProgressOverview', () => {
  it('renders the section heading', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByRole('heading', { name: /your progress/i })).toBeInTheDocument();
  });

  it('renders lessons completed / total', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByText(/72/)).toBeInTheDocument();
  });

  it('renders streak label', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByText(/streak/i)).toBeInTheDocument();
  });

  it('renders hours remaining label', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByText(/hours remaining/i)).toBeInTheDocument();
  });

  it('renders mastery label', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByText('Domain Mastery')).toBeInTheDocument();
  });

  it('renders lessons completed label', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByText(/lessons completed/i)).toBeInTheDocument();
  });

  it('renders days as unit next to streak value', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByText(/days/i)).toBeInTheDocument();
  });

  it('renders overall progress label', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByLabelText(/overall progress/i)).toBeInTheDocument();
  });

  it('renders total lessons subtitle', () => {
    render(<ProgressOverview progress={mockProgress} color="#0EA5E9" />);
    expect(screen.getByText(/of 72 total/i)).toBeInTheDocument();
  });

  it('accepts zero progress without crashing', () => {
    const zero: DomainProgress = { lessonsCompleted: 0, totalLessons: 50, currentStreak: 0, estimatedHoursRemaining: 100, masteryPercent: 0 };
    expect(() => render(<ProgressOverview progress={zero} color="#7C3AED" />)).not.toThrow();
  });
});
