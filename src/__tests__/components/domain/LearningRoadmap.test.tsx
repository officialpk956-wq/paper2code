import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { LearningRoadmap } from '@/components/domain/LearningRoadmap';
import type { RoadmapStage } from '@/data/domains/types';

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

const stages: RoadmapStage[] = [
  { id: 'foundations', title: 'Foundations', description: 'Basics of DL', status: 'completed', topics: ['Perceptron', 'Sigmoid'], order: 1 },
  { id: 'backprop', title: 'Backpropagation', description: 'Chain rule', status: 'in_progress', topics: ['Chain Rule', 'SGD'], order: 2 },
  { id: 'advanced', title: 'Advanced Topics', description: 'Transformers', status: 'locked', topics: ['Attention'], order: 3 },
];

describe('LearningRoadmap', () => {
  it('renders section heading', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByRole('heading', { name: /learning roadmap/i })).toBeInTheDocument();
  });

  it('renders all stage titles', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText('Foundations')).toBeInTheDocument();
    expect(screen.getByText('Backpropagation')).toBeInTheDocument();
    expect(screen.getByText('Advanced Topics')).toBeInTheDocument();
  });

  it('renders stage descriptions for expanded stages', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    // in_progress is auto-expanded
    expect(screen.getByText('Chain rule')).toBeInTheDocument();
  });

  it('renders topic chips for expanded stage', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText('Chain Rule')).toBeInTheDocument();
    expect(screen.getByText('SGD')).toBeInTheDocument();
  });

  it('expands a completed stage on click', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    const foundationsBtn = screen.getByRole('button', { name: /foundations/i });
    fireEvent.click(foundationsBtn);
    expect(screen.getByText('Basics of DL')).toBeInTheDocument();
  });

  it('auto-expands the in_progress stage (aria-expanded=true initially)', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    const backpropBtn = screen.getByRole('button', { name: /backpropagation/i });
    expect(backpropBtn).toHaveAttribute('aria-expanded', 'true');
  });

  it('renders stage order badges', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    expect(screen.getByText('Stage 1')).toBeInTheDocument();
    expect(screen.getByText('Stage 2')).toBeInTheDocument();
    expect(screen.getByText('Stage 3')).toBeInTheDocument();
  });

  it('renders empty state without crashing', () => {
    expect(() => render(<LearningRoadmap stages={[]} domainSlug="deep-learning" color="#0EA5E9" />)).not.toThrow();
  });

  it('renders topic chips as links to topic pages', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    const links = screen.getAllByRole('link');
    const topicLinks = links.filter(l => l.getAttribute('href')?.includes('/deep-learning/'));
    expect(topicLinks.length).toBeGreaterThan(0);
  });

  it('topic link has correct domain slug', () => {
    render(<LearningRoadmap stages={stages} domainSlug="deep-learning" color="#0EA5E9" />);
    const link = screen.getByRole('link', { name: /chain rule/i });
    expect(link.getAttribute('href')).toContain('/deep-learning/');
  });
});
