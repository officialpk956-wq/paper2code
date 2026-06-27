import { describe, it, expect, vi, beforeEach } from 'vitest';
import { generateYourPath, getPathProgress } from '@/lib/assessment/path';
import type { Profile } from '@/lib/assessment';
import * as topicsData from '@/data/topics';
import * as rels from '@/lib/content/relationships';
import * as progress from '@/lib/progress';

// Mock the dependencies
vi.mock('@/data/topics', () => ({
  resolveTopicPath: vi.fn(),
  TOPIC_REGISTRY: {}
}));

vi.mock('@/lib/content/relationships', () => ({
  getIndexEntry: vi.fn(),
}));

vi.mock('@/lib/progress', () => ({
  isCompleted: vi.fn(),
}));

vi.mock('@/lib/recommendations', () => ({
  getRecommendations: vi.fn(() => [
    { title: 'Fallback Rec', href: '/fallback', type: 'problem' }
  ]),
}));

vi.mock('@/data/roadmaps', () => ({
  ROADMAPS: {
    'ai-engineer': {
      id: 'ai-engineer',
      phases: [
        {
          id: 'phase-1',
          nodes: [
            {
              linkedResources: [
                { url: '/learn/python', title: 'Invalid Topic' }, // should be dropped
                { url: '/learn/deep-learning/attention', title: 'Valid Topic' },
                { url: '/papers/attention-is-all-you-need', title: 'Valid Paper' },
                { url: '/math/vectors', title: 'Invalid Math' }, // dropped
              ]
            }
          ]
        }
      ]
    }
  }
}));

describe('generateYourPath', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    // Setup mocks
    vi.mocked(topicsData.resolveTopicPath).mockImplementation((slug) => {
      if (slug === 'attention') return '/learn/deep-learning/attention';
      return null;
    });
    
    vi.mocked(rels.getIndexEntry).mockImplementation((type, slug) => {
      if (type === 'paper' && slug === 'attention-is-all-you-need') return {} as unknown;
      return null;
    });

    vi.mocked(progress.isCompleted).mockReturnValue(false);
  });

  it('drops non-existent roadmap urls and returns only resolving items', () => {
    const profile: Profile = {
      level: 'foundation-builder',
      goal: 'AI/ML Engineer',
      roadmapId: 'ai-engineer',
      hoursPerWeek: '10+',
      skillProfile: { foundations: 0, architectures: 0, systems: 0, implementation: 0 },
      completedAt: Date.now()
    };

    const path = generateYourPath(profile, 5);
    
    // We expect exactly 2 valid items from the roadmap, plus 3 fallback
    // Since limit is 5, it should fallback to add 1 from our mocked getRecommendations
    expect(path.length).toBe(3);
    expect(path[0].href).toBe('/learn/deep-learning/attention');
    expect(path[1].href).toBe('/papers/attention-is-all-you-need');
    expect(path[2].href).toBe('/fallback');
  });

  it('excludes completed items', () => {
    const profile: Profile = {
      level: 'foundation-builder',
      goal: 'AI/ML Engineer',
      roadmapId: 'ai-engineer',
      hoursPerWeek: '10+',
      skillProfile: { foundations: 0, architectures: 0, systems: 0, implementation: 0 },
      completedAt: Date.now()
    };

    // Mark 'attention' topic as completed
    vi.mocked(progress.isCompleted).mockImplementation((type, slug) => {
      if (type === 'topic' && slug === 'attention') return true;
      return false;
    });

    const path = generateYourPath(profile, 5);
    
    // attention should be skipped
    expect(path[0].href).toBe('/papers/attention-is-all-you-need');
  });
});
