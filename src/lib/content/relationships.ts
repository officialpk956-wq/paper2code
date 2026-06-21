import {
  ContentType,
  IndexEntry,
  Relationships,
} from "./schemas";
import { TRACES } from "@/data/tensor-traces";
import contentIndex from "@/generated/content-index.json";

/**
 * Relationship graph over the build-time content index.
 *
 * Edges are declared one-way in metadata (e.g. a paper lists the
 * architecture it introduces). This module resolves both directions:
 * forward edges (what this item points at) and reverse edges (everything
 * that points at this item), so "Related Content" is always symmetric
 * without authors having to maintain both sides.
 */

const ENTRIES = contentIndex.entries as IndexEntry[];

const RELATIONSHIP_KEY_TO_TYPE: Record<keyof Relationships, ContentType> = {
  architectures: "architecture",
  papers: "paper",
  math: "math",
  problems: "problem",
  systemDesign: "system-design",
  interview: "interview",
  roadmaps: "roadmap",
  implementations: "implementation",
};

const TYPE_TO_RELATIONSHIP_KEY = Object.fromEntries(
  Object.entries(RELATIONSHIP_KEY_TO_TYPE).map(([k, v]) => [v, k])
) as Record<ContentType, keyof Relationships>;

export interface RelatedGroup {
  type: ContentType;
  items: IndexEntry[];
}

export function getIndexEntry(
  type: ContentType,
  slug: string
): IndexEntry | undefined {
  return ENTRIES.find((e) => e.type === type && e.slug === slug);
}

export function getAllEntries(): IndexEntry[] {
  return ENTRIES;
}

export function getEntriesByType(type: ContentType): IndexEntry[] {
  return ENTRIES.filter((e) => e.type === type);
}

/** Forward edges: items this entry's metadata explicitly references. */
function getForwardRelated(entry: IndexEntry): IndexEntry[] {
  const related: IndexEntry[] = [];
  const rels = entry.relationships ?? {};
  for (const [key, slugs] of Object.entries(rels)) {
    const targetType = RELATIONSHIP_KEY_TO_TYPE[key as keyof Relationships];
    if (!targetType || !slugs) continue;
    for (const slug of slugs) {
      const target = getIndexEntry(targetType, slug);
      if (target) related.push(target);
    }
  }
  return related;
}

/** Reverse edges: every entry whose metadata references this one. */
function getReverseRelated(entry: IndexEntry): IndexEntry[] {
  const key = TYPE_TO_RELATIONSHIP_KEY[entry.type];
  return ENTRIES.filter((other) => {
    if (other.type === entry.type && other.slug === entry.slug) return false;
    const slugs = (other.relationships ?? {})[key];
    return Array.isArray(slugs) && slugs.includes(entry.slug);
  });
}

/**
 * All related content for an item, deduplicated and grouped by type.
 * Used by the <RelatedContent /> section on every content page.
 */
export function getRelatedContent(
  type: ContentType,
  slug: string
): RelatedGroup[] {
  const entry = getIndexEntry(type, slug);
  if (!entry) return [];

  const seen = new Set<string>();
  const all: IndexEntry[] = [];
  for (const item of [...getForwardRelated(entry), ...getReverseRelated(entry)]) {
    const id = `${item.type}:${item.slug}`;
    if (seen.has(id)) continue;
    seen.add(id);
    all.push(item);
  }

  const groups = new Map<ContentType, IndexEntry[]>();
  for (const item of all) {
    const list = groups.get(item.type) ?? [];
    list.push(item);
    groups.set(item.type, list);
  }

  return Array.from(groups.entries()).map(([groupType, items]) => ({
    type: groupType,
    items,
  }));
}

/** Lightweight full-text search over the index (title, description, tags). */
export function searchContent(query: string): IndexEntry[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const terms = q.split(/\s+/);

  // Convert traces to fake IndexEntry for search
  const traceEntries = Object.values(TRACES).map((t) => ({
    slug: t.slug,
    type: "tensor-trace" as const,
    title: t.title + " Trace",
    description: t.description,
    tags: ["tensor", "trace", "architecture", t.paperSlug],
    difficulty: "advanced" as const,
    relationships: {},
  }));

  const allSearchable = [...ENTRIES, ...traceEntries];

  return allSearchable.map((entry) => {
    const haystackTitle = entry.title.toLowerCase();
    const haystackDesc = entry.description.toLowerCase();
    const haystackTags = entry.tags.join(" ").toLowerCase();

    let score = 0;
    for (const term of terms) {
      if (haystackTitle.includes(term)) score += 3;
      if (haystackTags.includes(term)) score += 2;
      if (haystackDesc.includes(term)) score += 1;
    }
    return { entry, score };
  })
    .filter((r) => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .map((r) => r.entry);
}

/** Route prefix for each content type's detail pages. */
export const TYPE_ROUTES: Record<ContentType, string> = {
  architecture: "/architectures",
  paper: "/papers",
  math: "#", // unbuilt section
  "system-design": "/system-design",
  problem: "/problems",
  interview: "#", // unbuilt section
  roadmap: "/roadmaps",
  implementation: "/paper-to-code",
  "tensor-trace": "/tensor-trace",
};

export function contentUrl(entry: Pick<IndexEntry, "type" | "slug">): string {
  return `${TYPE_ROUTES[entry.type]}/${entry.slug}`;
}
