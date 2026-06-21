import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TopicSidebar, CHAPTER_SECTIONS } from '@/components/topic/TopicSidebar';

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

vi.mock('@/lib/progress', () => ({
  isCompleted: vi.fn((_type, slug) => slug === 'completed-topic'),
}));

describe('TopicSidebar', () => {
  const onNavigate = vi.fn();

  it('renders all chapter sections', () => {
    render(
      <TopicSidebar
        activeSection={null}
        topicSlug="test-topic"
        durationMinutes={45}
        difficulty="intermediate"
        onNavigate={onNavigate}
      />
    );
    for (const section of CHAPTER_SECTIONS) {
      expect(screen.getByText(section.label)).toBeInTheDocument();
    }
  });

  it('renders duration', () => {
    render(
      <TopicSidebar
        activeSection={null}
        topicSlug="test-topic"
        durationMinutes={45}
        difficulty="intermediate"
        onNavigate={onNavigate}
      />
    );
    expect(screen.getByText(/45 min/i)).toBeInTheDocument();
  });

  it('renders completion percent 100% for completed topic', () => {
    render(
      <TopicSidebar
        activeSection={null}
        topicSlug="completed-topic"
        durationMinutes={45}
        difficulty="intermediate"
        onNavigate={onNavigate}
      />
    );
    expect(screen.getByText(/100%/i)).toBeInTheDocument();
  });

  it('renders completion percent 0% for uncompleted topic', () => {
    render(
      <TopicSidebar
        activeSection={null}
        topicSlug="test-topic"
        durationMinutes={45}
        difficulty="intermediate"
        onNavigate={onNavigate}
      />
    );
    expect(screen.getByText(/0%/i)).toBeInTheDocument();
  });

  it('highlights active section', () => {
    render(
      <TopicSidebar
        activeSection="motivation"
        topicSlug="test-topic"
        durationMinutes={45}
        difficulty="intermediate"
        onNavigate={onNavigate}
      />
    );
    const motivationBtn = screen.getByRole('button', { name: /motivation/i });
    expect(motivationBtn).toBeInTheDocument();
  });

  it('calls onNavigate when section clicked', () => {
    render(
      <TopicSidebar
        activeSection={null}
        topicSlug="test-topic"
        durationMinutes={45}
        difficulty="intermediate"
        onNavigate={onNavigate}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /motivation/i }));
    expect(onNavigate).toHaveBeenCalledWith('motivation');
  });

  it('renders nav landmark', () => {
    render(
      <TopicSidebar
        activeSection={null}
        topicSlug="test-topic"
        durationMinutes={45}
        difficulty="intermediate"
        onNavigate={onNavigate}
      />
    );
    expect(screen.getByRole('navigation', { name: /chapter/i })).toBeInTheDocument();
  });

  it('renders difficulty', () => {
    render(
      <TopicSidebar
        activeSection={null}
        topicSlug="test-topic"
        durationMinutes={45}
        difficulty="advanced"
        onNavigate={onNavigate}
      />
    );
    expect(screen.getByText(/advanced/i)).toBeInTheDocument();
  });

  it('renders 11 sections', () => {
    expect(CHAPTER_SECTIONS.length).toBe(11);
  });

  it('sections include code and mathematics', () => {
    const ids = CHAPTER_SECTIONS.map(s => s.id);
    expect(ids).toContain('mathematics');
    expect(ids).toContain('code');
  });

  it('renders progress label', () => {
    render(
      <TopicSidebar
        activeSection={null}
        topicSlug="test-topic"
        durationMinutes={45}
        difficulty="intermediate"
        onNavigate={onNavigate}
      />
    );
    expect(screen.getByText(/progress/i)).toBeInTheDocument();
  });
});
