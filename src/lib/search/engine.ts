/**
 * Shared search engine used by /search page and CommandPalette.
 * Indexes all static data (problems, architectures, roadmaps, tracks).
 * Pure client-side fuzzy match — no backend required.
 */

import { PROBLEMS } from '@/data/problems';
import { EVOLUTION_NODES } from '@/data/evolution';
import { ROADMAP_LIST } from '@/data/roadmaps';
import { LEARNING_TRACKS } from '@/data/learning-tracks';
import { TOPIC_REGISTRY } from '@/data/topics';
import { getCompletedByType, getInProgress } from '@/lib/progress';
import { getAllMasteryLevels } from '@/lib/persistence';

export type SearchResultType =
  | 'problem'
  | 'architecture'
  | 'roadmap'
  | 'track'
  | 'topic'
  | 'navigation';

export interface SearchResult {
  id: string;
  title: string;
  description: string;
  href: string;
  type: SearchResultType;
  tags?: string[];
  matchScore: number;
  // Enrichment fields (populated by searchWithContext)
  whyThisResult?: string;
  difficulty?: string;
  estimatedMinutes?: number;
  completionState?: 'completed' | 'in_progress' | 'not_started';
  prerequisites?: string[];
}

// ---------------------------------------------------------------------------
// Build flat index from static data
// ---------------------------------------------------------------------------

interface IndexEntry {
  id: string;
  title: string;
  description: string;
  href: string;
  type: SearchResultType;
  tags: string[];
  searchText: string; // lower-cased combined field for matching
}

function buildIndex(): IndexEntry[] {
  const entries: IndexEntry[] = [];

  // Problems
  for (const p of PROBLEMS) {
    const text = [p.title, p.description, ...p.tags, p.category, p.difficulty]
      .join(' ')
      .toLowerCase();
    entries.push({
      id: `problem-${p.slug}`,
      title: p.title,
      description: p.description.slice(0, 120),
      href: `/problems/${p.slug}`,
      type: 'problem',
      tags: p.tags,
      searchText: text,
    });
  }

  // Evolution nodes (architectures / papers)
  for (const n of EVOLUTION_NODES) {
    const text = [
      n.title,
      n.type,
      ...n.innovationsIntroduced,
      ...n.problemsSolved,
    ]
      .join(' ')
      .toLowerCase();
    entries.push({
      id: `arch-${n.slug}`,
      title: n.title,
      description: n.innovationsIntroduced[0] ?? n.type,
      href: `/papers/${n.slug}`,
      type: 'architecture',
      tags: [n.type, ...n.innovationsIntroduced.slice(0, 2)],
      searchText: text,
    });
  }

  // Roadmaps
  for (const roadmap of ROADMAP_LIST) {
    const text = [roadmap.title, roadmap.description].join(' ').toLowerCase();
    entries.push({
      id: `roadmap-${roadmap.id}`,
      title: roadmap.title,
      description: roadmap.description,
      href: `/roadmaps`,
      type: 'roadmap',
      tags: ['roadmap', 'learning path'],
      searchText: text,
    });
  }

  // Learning tracks
  for (const track of LEARNING_TRACKS) {
    const text = [
      track.title,
      track.subtitle,
      track.description,
      track.architecture.title,
      ...track.keyObjectives,
    ]
      .join(' ')
      .toLowerCase();
    entries.push({
      id: `track-${track.slug}`,
      title: track.title,
      description: track.subtitle,
      href: `/learning-tracks/${track.slug}`,
      type: 'track',
      tags: [track.architecture.title, track.difficulty],
      searchText: text,
    });
  }

  // Topics
  for (const [domain, topics] of Object.entries(TOPIC_REGISTRY)) {
    for (const [slug, data] of Object.entries(topics)) {
      const text = [
        data.meta.title,
        data.meta.subtitle,
        ...data.meta.tags,
      ].join(' ').toLowerCase();
      entries.push({
        id: `topic-${slug}`,
        title: data.meta.title,
        description: data.meta.subtitle,
        href: `/learn/${domain}/${slug}`,
        type: 'topic',
        tags: data.meta.tags,
        searchText: text,
      });
    }
  }

  // Static navigation entries
  const navEntries: Omit<IndexEntry, 'searchText'>[] = [
    { id: 'nav-dashboard', title: 'Dashboard', description: 'Your learning overview and progress', href: '/dashboard', type: 'navigation', tags: [] },
    { id: 'nav-papers', title: 'Papers', description: 'Browse AI research papers', href: '/papers', type: 'navigation', tags: [] },
    { id: 'nav-problems', title: 'Problems', description: 'Coding problems for ML/DL', href: '/problems', type: 'navigation', tags: [] },
    { id: 'nav-architectures', title: 'Architectures', description: 'Interactive architecture explorer', href: '/architectures', type: 'navigation', tags: [] },
    { id: 'nav-system-design', title: 'System Design', description: 'AI system design patterns', href: '/system-design', type: 'navigation', tags: [] },
    { id: 'nav-evolution', title: 'Research Evolution', description: 'Trace lineage of AI research ideas', href: '/evolution', type: 'navigation', tags: [] },
    { id: 'nav-compare', title: 'Compare Architectures', description: 'Side-by-side architecture comparison', href: '/compare', type: 'navigation', tags: [] },
    { id: 'nav-tensor-trace', title: 'Tensor Trace', description: 'Visualize tensor shapes through networks', href: '/tensor-trace', type: 'navigation', tags: [] },
    { id: 'nav-attention-viz', title: 'Attention Visualizer', description: 'Interactive attention head visualization', href: '/attention-viz', type: 'navigation', tags: [] },
    { id: 'nav-roadmaps', title: 'Roadmaps', description: 'Structured learning paths', href: '/roadmaps', type: 'navigation', tags: [] },
    { id: 'nav-knowledge', title: 'Knowledge Intelligence', description: 'AI learning coach and knowledge graph', href: '/knowledge-intelligence', type: 'navigation', tags: [] },
    { id: 'nav-collab', title: 'Collaboration', description: 'Real-time collaborative workspace', href: '/real-time-collaboration', type: 'navigation', tags: [] },
  ];
  for (const nav of navEntries) {
    entries.push({
      ...nav,
      searchText: [nav.title, nav.description].join(' ').toLowerCase(),
    });
  }

  return entries;
}

// Lazy-initialize once
let _index: IndexEntry[] | null = null;
function getIndex(): IndexEntry[] {
  if (_index === null) {
    _index = buildIndex();
  }
  return _index;
}

// ---------------------------------------------------------------------------
// Fuzzy scoring — simple but effective for static data
// ---------------------------------------------------------------------------

function score(entry: IndexEntry, query: string): number {
  const q = query.toLowerCase().trim();
  if (q.length === 0) return 0;

  const words = q.split(/\s+/);
  let total = 0;

  for (const word of words) {
    if (entry.searchText.includes(word)) {
      // Exact word in full text
      total += 1;
      // Bonus: appears in title
      if (entry.title.toLowerCase().includes(word)) total += 3;
      // Bonus: starts title
      if (entry.title.toLowerCase().startsWith(word)) total += 2;
    }
  }

  // Bonus for full phrase match
  if (entry.searchText.includes(q)) total += 5;

  return total;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export interface SearchOptions {
  limit?: number;
  types?: SearchResultType[];
}

export function search(query: string, options: SearchOptions = {}): SearchResult[] {
  const { limit = 20, types } = options;
  const q = query.trim();
  if (q.length === 0) return [];

  const index = getIndex();
  const results: SearchResult[] = [];

  for (const entry of index) {
    if (types && !types.includes(entry.type)) continue;
    const matchScore = score(entry, q);
    if (matchScore > 0) {
      results.push({
        id: entry.id,
        title: entry.title,
        description: entry.description,
        href: entry.href,
        type: entry.type,
        tags: entry.tags,
        matchScore,
      });
    }
  }

  return results
    .sort((a, b) => b.matchScore - a.matchScore)
    .slice(0, limit);
}

/** Highlight matching query terms in a string with <mark> tags.
 *  Safe: escapes HTML in the input before inserting marks. */
export function highlight(text: string, query: string): string {
  if (!query.trim()) return escapeHtml(text);
  const escaped = escapeHtml(text);
  const words = query
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 1)
    .map((w) => escapeHtml(w).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  if (words.length === 0) return escaped;
  const pattern = new RegExp(`(${words.join('|')})`, 'gi');
  return escaped.replace(pattern, '<mark class="bg-[--accent-primary]/20 text-[--accent-light] rounded-sm">$1</mark>');
}

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export const TYPE_LABELS: Record<SearchResultType, string> = {
  problem: 'Problem',
  architecture: 'Architecture',
  roadmap: 'Roadmap',
  track: 'Learning Track',
  topic: 'Topic',
  navigation: 'Page',
};

export const TYPE_COLORS: Record<SearchResultType, string> = {
  problem: '#F59E0B',
  architecture: '#06B6D4',
  roadmap: '#10B981',
  track: '#7C3AED',
  topic: '#EC4899',
  navigation: '#64748B',
};

// ---------------------------------------------------------------------------
// Mastery-aware search (Phase H)
// Augments base scoring with user progress signals.
// Imports are inline to keep SSR-safe (called only client-side).
// ---------------------------------------------------------------------------

export interface SearchContext {
  completedSlugs: Set<string>;
  inProgressSlugs: Set<string>;
  masteryMap: Record<string, number>; // slug → 0-4
}

export function searchWithContext(
  query: string,
  context: SearchContext,
  options: SearchOptions = {},
): SearchResult[] {
  const base = search(query, { ...options, limit: (options.limit ?? 20) * 3 });

  const enriched: SearchResult[] = base.map((result) => {
    // Extract slug from id: "problem-<slug>", "arch-<slug>", "roadmap-<slug>", "track-<slug>", "topic-<slug>"
    const slugPart = result.id.replace(/^(problem|arch|roadmap|track|topic|nav)-/, '');
    let bonus = 0;
    let why = '';
    let completionState: SearchResult['completionState'] = 'not_started';

    if (context.completedSlugs.has(slugPart)) {
      completionState = 'completed';
      // Completed items: surface for revision only if mastery is low
      const mastery = context.masteryMap[slugPart] ?? 0;
      if (mastery < 2) {
        bonus += 1;
        why = 'Completed — low mastery, revisit to reinforce';
      } else {
        bonus -= 2; // deprioritise already-mastered
        why = 'Already completed';
      }
    } else if (context.inProgressSlugs.has(slugPart)) {
      completionState = 'in_progress';
      bonus += 4;
      why = 'In progress — continue where you left off';
    } else {
      // Check mastery of related content via tags as a proxy
      const tagMastery = (result.tags ?? []).reduce(
        (s, tag) => s + (context.masteryMap[tag] ?? 0),
        0,
      );
      if (tagMastery > 0) {
        bonus += Math.min(tagMastery, 3);
        why = `Related to ${tagMastery} mastery point${tagMastery > 1 ? 's' : ''} in your profile`;
      } else {
        why = 'Matches your search';
      }
    }

    return {
      ...result,
      matchScore: result.matchScore + bonus,
      whyThisResult: why,
      completionState,
    };
  });

  return enriched
    .sort((a, b) => b.matchScore - a.matchScore)
    .slice(0, options.limit ?? 20);
}

/** Build a SearchContext from the user's current progress. Client-side only. */
export function buildSearchContext(): SearchContext {
  try {
    const completedSlugs = new Set<string>([
      ...getCompletedByType('problem'),
      ...getCompletedByType('paper'),
      ...getCompletedByType('architecture'),
    ]);
    const inProgressSlugs = new Set<string>(getInProgress().map((p) => p.slug));
    const masteryMap = getAllMasteryLevels() as Record<string, number>;
    return { completedSlugs, inProgressSlugs, masteryMap };
  } catch {
    return { completedSlugs: new Set(), inProgressSlugs: new Set(), masteryMap: {} };
  }
}
