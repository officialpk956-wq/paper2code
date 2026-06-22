import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TrendingTopics } from '@/components/learn/TrendingTopics';
import type { TrendingTopic } from '@/data/learn/topics';

vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
  },
}));

const MOCK_TOPICS: TrendingTopic[] = [
  {
    id: 'reasoning-models', title: 'Reasoning Models', description: 'CoT and MCTS for LLMs.',
    popularity: 99, learnerCount: 14300, weeklyGrowth: 67, domain: 'Large Language Models',
    color: '#A78BFA', glowColor: 'rgba(167,139,250,0.25)', trend: 'hot', isNew: true,
    tags: ['CoT', 'MCTS'],
  },
  {
    id: 'diffusion-models', title: 'Diffusion Models', description: 'Score matching and DDPM.',
    popularity: 91, learnerCount: 9140, weeklyGrowth: 19, domain: 'Computer Vision',
    color: '#10B981', glowColor: 'rgba(16,185,129,0.25)', trend: 'stable', isNew: false,
    tags: ['DDPM'],
  },
];

describe('TrendingTopics', () => {
  it('renders the section heading', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    expect(screen.getByRole('heading', { name: /trending topics/i })).toBeInTheDocument();
  });

  it('renders all topic titles', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    expect(screen.getByText('Reasoning Models')).toBeInTheDocument();
    expect(screen.getByText('Diffusion Models')).toBeInTheDocument();
  });

  it('renders popularity percentages', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    expect(screen.getByText('99%')).toBeInTheDocument();
    expect(screen.getByText('91%')).toBeInTheDocument();
  });

  it('renders learner counts', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    expect(screen.getByText(/14,300 learners/)).toBeInTheDocument();
  });

  it('shows New badge for new topics', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    expect(screen.getByText('New')).toBeInTheDocument();
  });

  it('does not show New badge for non-new topics', () => {
    render(<TrendingTopics topics={[MOCK_TOPICS[1]]} />);
    expect(screen.queryByText('New')).not.toBeInTheDocument();
  });

  it('renders weekly growth', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    expect(screen.getByText('+67% this week')).toBeInTheDocument();
  });

  it('renders links for each topic', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    const links = screen.getAllByRole('link');
    const hrefs = links.map(l => l.getAttribute('href'));
    expect(hrefs).toContain('/learn/topics/reasoning-models');
    expect(hrefs).toContain('/learn/topics/diffusion-models');
  });

  it('has aria-label on the scroll list', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    expect(screen.getByRole('list', { name: /trending topics/i })).toBeInTheDocument();
  });

  it('renders topic descriptions', () => {
    render(<TrendingTopics topics={MOCK_TOPICS} />);
    expect(screen.getByText(/cot and mcts for llms/i)).toBeInTheDocument();
  });
});
