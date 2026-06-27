import { ROADMAPS } from '@/data/roadmaps';

import { resolveTopicPath } from '@/data/topics';
import { getIndexEntry } from '@/lib/content/relationships';
import { isCompleted } from '@/lib/progress';
import { getRecommendations } from '@/lib/recommendations';
import type { Recommendation } from '@/lib/recommendations';
import type { Profile } from './index';
import type { ProgressContentType } from '@/lib/progress';

export function generateYourPath(profile: Profile, limit = 5): Recommendation[] {
  const roadmap = ROADMAPS[profile.roadmapId];
  if (!roadmap) return [];

  const startIndex = roadmap.phases.findIndex(p => p.id === profile.startPhaseId);
  const phasesToCheck = roadmap.phases.slice(Math.max(0, startIndex));

  const validItems: Recommendation[] = [];

  for (const phase of phasesToCheck) {
    for (const node of phase.nodes) {
      for (const res of node.linkedResources) {
        // Validate and classify the URL
        let type: ProgressContentType | null = null;
        let slug = '';
        let isValid = false;

        if (res.url.startsWith('/learn/')) {
          // e.g. /learn/deep-learning/attention
          const parts = res.url.split('/');
          if (parts.length >= 4) {
            slug = parts[3];
            type = 'topic';
            isValid = resolveTopicPath(slug) !== null;
          }
        } else if (res.url.startsWith('/papers/')) {
          slug = res.url.replace('/papers/', '');
          type = 'paper';
          isValid = getIndexEntry('paper', slug) !== null;
        } else if (res.url.startsWith('/architectures/')) {
          slug = res.url.replace('/architectures/', '');
          type = 'architecture';
          isValid = getIndexEntry('architecture', slug) !== null;
        } else if (res.url.startsWith('/problems/')) {
          slug = res.url.replace('/problems/', '');
          type = 'problem';
          isValid = getIndexEntry('problem', slug) !== null;
        }

        if (isValid && type && slug) {
          if (!isCompleted(type, slug)) {
            // Found a valid, incomplete item
            validItems.push({
              type,
              slug,
              title: res.title,
              description: node.title,
              href: res.url,
              category: 'continue_learning',
              reason: `From your path: ${phase.name}`,
              score: 100 - validItems.length // strict ordering
            });

            if (validItems.length >= limit) {
              return validItems;
            }
          }
        }
      }
    }
  }

  // Fallback: pad with recommendations if we run out of valid roadmap items
  if (validItems.length < limit) {
    const recs = getRecommendations(limit - validItems.length);
    for (const r of recs) {
      // Avoid duplicates
      if (!validItems.some(v => v.href === r.href)) {
        validItems.push(r);
      }
    }
  }

  return validItems;
}

export function getPathProgress(profile: Profile): number {
  const roadmap = ROADMAPS[profile.roadmapId];
  if (!roadmap) return 0;

  // We only count progress for phases up to and including the current starting phase,
  // OR all phases if they've completed beyond it.
  // Actually, a simpler and honest metric: out of all *valid* resources in the entire roadmap,
  // what percentage are completed?
  
  let validCount = 0;
  let completedCount = 0;

  for (const phase of roadmap.phases) {
    for (const node of phase.nodes) {
      for (const res of node.linkedResources) {
        let type: ProgressContentType | null = null;
        let slug = '';
        let isValid = false;

        if (res.url.startsWith('/learn/')) {
          const parts = res.url.split('/');
          if (parts.length >= 4) {
            slug = parts[3];
            type = 'topic';
            isValid = resolveTopicPath(slug) !== null;
          }
        } else if (res.url.startsWith('/papers/')) {
          slug = res.url.replace('/papers/', '');
          type = 'paper';
          isValid = getIndexEntry('paper', slug) !== null;
        } else if (res.url.startsWith('/architectures/')) {
          slug = res.url.replace('/architectures/', '');
          type = 'architecture';
          isValid = getIndexEntry('architecture', slug) !== null;
        } else if (res.url.startsWith('/problems/')) {
          slug = res.url.replace('/problems/', '');
          type = 'problem';
          isValid = getIndexEntry('problem', slug) !== null;
        }

        if (isValid && type && slug) {
          validCount++;
          if (isCompleted(type, slug)) {
            completedCount++;
          }
        }
      }
    }
  }

  if (validCount === 0) return 0;
  return Math.round((completedCount / validCount) * 100);
}
