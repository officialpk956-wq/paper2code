import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LearnHero } from '@/components/learn/LearnHero';

vi.mock('framer-motion', () => ({
  motion: {
    section: ({ children, ...props }: React.HTMLAttributes<HTMLElement>) => <section {...props}>{children}</section>,
    div:     ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
    h1:      ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => <h1 {...props}>{children}</h1>,
    p:       ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement>) => <p {...props}>{children}</p>,
    g:       ({ children, ...props }: React.SVGAttributes<SVGGElement>) => <g {...props}>{children}</g>,
    circle:  ({ children, ...props }: React.SVGAttributes<SVGCircleElement>) => <circle {...props}>{children}</circle>,
  },
  useMotionValue: () => ({ set: vi.fn(), get: vi.fn() }),
  useMotionTemplate: () => '',
  useSpring: () => ({ set: vi.fn() }),
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const mockStats = { paperCount: 24, archCount: 21, problemCount: 29, topicCount: 12 };

describe('LearnHero', () => {
  it('renders the main headline', () => {
    render(<LearnHero stats={mockStats} />);
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
    expect(screen.getByText('Learn AI')).toBeInTheDocument();
  });

  it('renders the hero section with accessible label', () => {
    render(<LearnHero stats={mockStats} />);
    expect(screen.getByRole('region', { name: /learn home hero/i })).toBeInTheDocument();
  });

  it('renders all 4 stats', () => {
    render(<LearnHero stats={mockStats} />);
    expect(screen.getByText('12')).toBeInTheDocument();
    expect(screen.getByText('21')).toBeInTheDocument();
    expect(screen.getByText('29')).toBeInTheDocument();
    expect(screen.getByText('24')).toBeInTheDocument();
  });

  it('renders stat labels', () => {
    render(<LearnHero stats={mockStats} />);
    expect(screen.getByText('Lessons')).toBeInTheDocument();
    expect(screen.getByText('Architectures')).toBeInTheDocument();
    expect(screen.getByText('Problems')).toBeInTheDocument();
    expect(screen.getByText('Research Papers')).toBeInTheDocument();
  });

  it('renders Continue Learning CTA link', () => {
    render(<LearnHero stats={mockStats} />);
    const link = screen.getByRole('link', { name: /continue learning/i });
    expect(link).toBeInTheDocument();
  });

  it('renders Explore Domains CTA link', () => {
    render(<LearnHero stats={mockStats} />);
    const link = screen.getByRole('link', { name: /explore.*learning domains/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute('href', '#domains');
  });

  it('renders the AI Engineering University badge', () => {
    render(<LearnHero stats={mockStats} />);
    expect(screen.getByText('AI Engineering University')).toBeInTheDocument();
  });

  it('renders the subtitle paragraph', () => {
    render(<LearnHero stats={mockStats} />);
    expect(screen.getByText(/master mathematics/i)).toBeInTheDocument();
  });

  it('graph SVG is aria-hidden', () => {
    render(<LearnHero stats={mockStats} />);
    const hiddenSvg = document.querySelector('svg[aria-hidden="true"]');
    expect(hiddenSvg).not.toBeNull();
  });
});

