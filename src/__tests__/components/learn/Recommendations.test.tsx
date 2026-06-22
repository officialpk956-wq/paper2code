import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Recommendations } from '@/components/learn/Recommendations';
import type { Recommendation } from '@/data/learn/recommendations';

vi.mock('framer-motion', () => ({
  motion: {
    div:    ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
    circle: ({ children, ...props }: React.SVGAttributes<SVGCircleElement>) => <circle {...props}>{children}</circle>,
  },
}));

const MOCK_RECS: Recommendation[] = [
  {
    id: 'rec-flash-attention', title: 'FlashAttention', type: 'paper', domain: 'Deep Learning',
    confidence: 96, reason: 'Based on your Transformer progress', estimatedMinutes: 35,
    color: '#06B6D4', href: '/papers/flash-attention', difficulty: 'advanced',
  },
  {
    id: 'rec-rope', title: 'Rotary Positional Embeddings', type: 'lesson', domain: 'LLMs',
    confidence: 91, reason: 'Follows from attention studies', estimatedMinutes: 22,
    color: '#A78BFA', href: '/learn/llms/rope', difficulty: 'advanced',
  },
];

describe('Recommendations', () => {
  it('renders the section heading', () => {
    render(<Recommendations items={MOCK_RECS} />);
    expect(screen.getByRole('heading', { name: /recommended for you/i })).toBeInTheDocument();
  });

  it('renders all item titles', () => {
    render(<Recommendations items={MOCK_RECS} />);
    expect(screen.getByText('FlashAttention')).toBeInTheDocument();
    expect(screen.getByText('Rotary Positional Embeddings')).toBeInTheDocument();
  });

  it('renders type labels', () => {
    render(<Recommendations items={MOCK_RECS} />);
    expect(screen.getByText('Paper')).toBeInTheDocument();
    expect(screen.getByText('Lesson')).toBeInTheDocument();
  });

  it('renders confidence percentages in rings', () => {
    render(<Recommendations items={MOCK_RECS} />);
    expect(screen.getByText('96%')).toBeInTheDocument();
    expect(screen.getByText('91%')).toBeInTheDocument();
  });

  it('renders reason text', () => {
    render(<Recommendations items={MOCK_RECS} />);
    expect(screen.getByText(/based on your transformer progress/i)).toBeInTheDocument();
  });

  it('renders estimated time', () => {
    render(<Recommendations items={MOCK_RECS} />);
    expect(screen.getByText('35 min')).toBeInTheDocument();
  });

  it('renders links with correct hrefs', () => {
    render(<Recommendations items={MOCK_RECS} />);
    const links = screen.getAllByRole('link');
    const hrefs = links.map(l => l.getAttribute('href'));
    expect(hrefs).toContain('/papers/flash-attention');
    expect(hrefs).toContain('/learn/llms/rope');
  });

  it('renders empty state when items is empty', () => {
    render(<Recommendations items={[]} />);
    expect(screen.getByText(/complete a few lessons/i)).toBeInTheDocument();
  });

  it('has role=list when items are present', () => {
    render(<Recommendations items={MOCK_RECS} />);
    expect(screen.getByRole('list', { name: /recommended content/i })).toBeInTheDocument();
  });

  it('renders domain names', () => {
    render(<Recommendations items={MOCK_RECS} />);
    expect(screen.getByText('Deep Learning')).toBeInTheDocument();
  });
});
