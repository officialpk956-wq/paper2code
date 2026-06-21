import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ResearchTimeline } from '@/components/topic/ResearchTimeline';
import type { TimelineEvent } from '@/data/topics/types';

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

const events: TimelineEvent[] = [
  {
    id: 'bahdanau',
    year: 2014,
    title: 'Bahdanau Attention',
    summary: 'First neural attention mechanism for machine translation.',
    color: '#0EA5E9',
    isHighlighted: false,
  },
  {
    id: 'transformer',
    year: 2017,
    title: 'Transformer',
    summary: 'Attention Is All You Need: scaled dot-product self-attention.',
    color: '#7C3AED',
    isHighlighted: true,
  },
];

describe('ResearchTimeline', () => {
  it('renders section heading', () => {
    render(<ResearchTimeline events={events} />);
    expect(screen.getByRole('heading', { name: /research timeline/i })).toBeInTheDocument();
  });

  it('renders section with id="timeline"', () => {
    render(<ResearchTimeline events={events} />);
    expect(document.querySelector('#timeline')).not.toBeNull();
  });

  it('renders all event titles in the sidebar', () => {
    render(<ResearchTimeline events={events} />);
    // Both titles appear at least once (one in sidebar button, one possibly in detail panel)
    expect(screen.getAllByText('Bahdanau Attention').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Transformer').length).toBeGreaterThanOrEqual(1);
  });

  it('renders years in the sidebar', () => {
    render(<ResearchTimeline events={events} />);
    expect(screen.getAllByText('2014').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('2017').length).toBeGreaterThanOrEqual(1);
  });

  it('renders "Landmark" badge for highlighted events', () => {
    render(<ResearchTimeline events={events} />);
    expect(screen.getByText('Landmark')).toBeInTheDocument();
  });

  it('defaults to showing highlighted event summary', () => {
    render(<ResearchTimeline events={events} />);
    expect(screen.getByText('Attention Is All You Need: scaled dot-product self-attention.')).toBeInTheDocument();
  });

  it('clicking a different event shows its summary', () => {
    render(<ResearchTimeline events={events} />);
    // The button role is overridden by role="listitem", query by listitem accessible name
    const bahdanauBtn = screen.getByRole('listitem', { name: /2014.*Bahdanau/i });
    fireEvent.click(bahdanauBtn);
    expect(screen.getByText('First neural attention mechanism for machine translation.')).toBeInTheDocument();
  });

  it('renders list with aria-label', () => {
    render(<ResearchTimeline events={events} />);
    expect(screen.getByRole('list', { name: /timeline milestones/i })).toBeInTheDocument();
  });

  it('renders subtitle text', () => {
    render(<ResearchTimeline events={events} />);
    expect(screen.getByText(/click any milestone/i)).toBeInTheDocument();
  });

  it('default selected event shows heading in detail panel', () => {
    render(<ResearchTimeline events={events} />);
    expect(screen.getByRole('heading', { name: 'Transformer' })).toBeInTheDocument();
  });
});
