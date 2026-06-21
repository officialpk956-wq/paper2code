import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DomainGrid } from '@/components/learn/DomainGrid';
import type { Domain } from '@/data/learn/domains';

vi.mock('framer-motion', () => ({
  motion: {
    div:    ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
    circle: ({ children, ...props }: React.SVGAttributes<SVGCircleElement>) => <circle {...props}>{children}</circle>,
  },
}));

const MOCK_DOMAINS: Domain[] = [
  {
    id: 'mathematics', slug: 'mathematics', name: 'Mathematics', icon: 'Sigma',
    description: 'Linear algebra, calculus, probability.', topicCount: 18, lessonCount: 48,
    progress: 72, color: '#8B5CF6', glowColor: 'rgba(139,92,246,0.25)',
    difficulty: 'intermediate', category: 'foundation', estimatedHours: 40,
    href: '/learn/mathematics',
  },
  {
    id: 'deep-learning', slug: 'deep-learning', name: 'Deep Learning', icon: 'Layers3',
    description: 'Neural networks and backpropagation.', topicCount: 26, lessonCount: 72,
    progress: 0, color: '#06B6D4', glowColor: 'rgba(6,182,212,0.25)',
    difficulty: 'advanced', category: 'core', estimatedHours: 65,
    href: '/learn/deep-learning',
  },
];

describe('DomainGrid', () => {
  it('renders the section heading', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    expect(screen.getByRole('heading', { name: /learning domains/i })).toBeInTheDocument();
  });

  it('renders all domain names', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    expect(screen.getByText('Mathematics')).toBeInTheDocument();
    expect(screen.getByText('Deep Learning')).toBeInTheDocument();
  });

  it('renders domain descriptions', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    expect(screen.getByText(/linear algebra/i)).toBeInTheDocument();
  });

  it('renders topic and lesson counts', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    expect(screen.getByText(/18 topics · 48 lessons/i)).toBeInTheDocument();
  });

  it('renders links to each domain', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    const links = screen.getAllByRole('link');
    const hrefs = links.map(l => l.getAttribute('href'));
    expect(hrefs).toContain('/learn/mathematics');
    expect(hrefs).toContain('/learn/deep-learning');
  });

  it('has role=list on the grid', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    expect(screen.getByRole('list', { name: /learning domains/i })).toBeInTheDocument();
  });

  it('renders the section with id="domains" for anchor links', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    expect(document.getElementById('domains')).not.toBeNull();
  });

  it('renders progress ring showing 0 for unstarted domains', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    const dashes = screen.getAllByText('—');
    expect(dashes.length).toBeGreaterThan(0);
  });

  it('renders progress percentages for started domains', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    expect(screen.getByText('72%')).toBeInTheDocument();
  });

  it('renders all domains as list items', () => {
    render(<DomainGrid domains={MOCK_DOMAINS} />);
    const items = screen.getAllByRole('listitem');
    expect(items.length).toBe(MOCK_DOMAINS.length);
  });
});
