import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RecentlyAdded } from '@/components/learn/RecentlyAdded';
import type { RecentlyAddedItem } from '@/data/learn/recommendations';

vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
  },
}));

const MOCK_ITEMS: RecentlyAddedItem[] = [
  {
    id: 'ra-deepseek', title: 'DeepSeek-R1', type: 'paper', domain: 'LLMs',
    addedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    description: 'Process reward models for reasoning.', color: '#A78BFA',
    href: '/papers/deepseek-r1', readMinutes: 40,
  },
  {
    id: 'ra-moe-lesson', title: 'Sparse MoE: Switch to Mixtral', type: 'lesson', domain: 'LLMs',
    addedAt: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
    description: 'Token routing in sparse MoE.', color: '#7C3AED',
    href: '/learn/llms/sparse-moe', readMinutes: 25,
  },
];

describe('RecentlyAdded', () => {
  it('renders the section heading', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    expect(screen.getByRole('heading', { name: /recently added/i })).toBeInTheDocument();
  });

  it('renders all item titles', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    expect(screen.getByText('DeepSeek-R1')).toBeInTheDocument();
    expect(screen.getByText('Sparse MoE: Switch to Mixtral')).toBeInTheDocument();
  });

  it('renders type badges', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    expect(screen.getByText('Paper')).toBeInTheDocument();
    expect(screen.getByText('Lesson')).toBeInTheDocument();
  });

  it('renders read time', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    expect(screen.getByText('40 min')).toBeInTheDocument();
    expect(screen.getByText('25 min')).toBeInTheDocument();
  });

  it('renders relative dates', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    expect(screen.getByText('2 days ago')).toBeInTheDocument();
  });

  it('renders links with correct hrefs', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    const links = screen.getAllByRole('link');
    const hrefs = links.map(l => l.getAttribute('href'));
    expect(hrefs).toContain('/papers/deepseek-r1');
    expect(hrefs).toContain('/learn/llms/sparse-moe');
  });

  it('renders the View all link', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    const link = screen.getByRole('link', { name: /view all/i });
    expect(link).toHaveAttribute('href', '/learn/new');
  });

  it('renders empty state when items is empty', () => {
    render(<RecentlyAdded items={[]} />);
    expect(screen.getByText(/no new content/i)).toBeInTheDocument();
  });

  it('has role=list on the timeline', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    expect(screen.getByRole('list', { name: /recently added content/i })).toBeInTheDocument();
  });

  it('renders item descriptions', () => {
    render(<RecentlyAdded items={MOCK_ITEMS} />);
    expect(screen.getByText(/process reward models/i)).toBeInTheDocument();
  });
});
