import Link from 'next/link';
import { TOPIC_REGISTRY, resolveTopicPath } from '@/data/topics';
import type { TopicData } from '@/data/topics/types';

interface PaperRelatedTopicsProps {
  paperSlug: string;
}

/**
 * Server component — reads TOPIC_REGISTRY at build time.
 * For each topic whose relatedPapers contains an entry matching this paper slug,
 * surface a topic card linking to /learn/{domain}/{slug}.
 * Topics are deduplicated by meta.slug (handles registry aliases).
 * Returns null (renders nothing) if no topic references this paper.
 */
export function PaperRelatedTopics({ paperSlug }: PaperRelatedTopicsProps) {
  // Invert topic registry: collect all unique topics that reference this paper
  const matchingTopics = new Map<string, TopicData>();

  for (const domainTopics of Object.values(TOPIC_REGISTRY)) {
    for (const topic of Object.values(domainTopics)) {
      const referencesPaper = topic.relatedPapers.some(
        (paper) => paper.id === paperSlug || paper.href === `/papers/${paperSlug}`
      );
      if (referencesPaper) {
        // Dedupe by canonical meta.slug
        matchingTopics.set(topic.meta.slug, topic);
      }
    }
  }

  const topics = Array.from(matchingTopics.values());
  if (topics.length === 0) return null;

  return (
    <section
      className="mt-16 pt-10"
      style={{ borderTop: '1px solid var(--color-border)' }}
      aria-labelledby="related-topics-heading"
    >
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <h2
          id="related-topics-heading"
          className="text-xl font-bold mb-2"
          style={{ color: 'var(--color-text-primary)' }}
        >
          Master the ideas behind this paper
        </h2>
        <p className="text-sm mb-6" style={{ color: 'var(--color-text-secondary)' }}>
          Deep-dive into the core concepts and architectures introduced in this research.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {topics.map((topic) => {
            const path = resolveTopicPath(topic.meta.slug) ?? '#';
            return (
              <Link
                key={topic.meta.slug}
                href={path}
                className="group relative overflow-hidden rounded-xl border p-5 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500"
                style={{
                  background: 'var(--bg-surface)',
                  borderColor: 'var(--color-border)',
                }}
              >
                {/* Hover glow overlay */}
                <div
                  className="absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-300 pointer-events-none"
                  style={{ background: topic.meta.domainColor }}
                />

                <div className="relative z-10 flex flex-col h-full">
                  {/* Badges */}
                  <div className="flex items-center gap-2 mb-3">
                    <span
                      className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider"
                      style={{
                        background: `${topic.meta.domainColor}22`,
                        color: topic.meta.domainColor,
                      }}
                    >
                      {topic.meta.difficulty}
                    </span>
                    <span
                      className="text-[10px] uppercase tracking-wider font-semibold truncate"
                      style={{ color: 'var(--color-text-tertiary)' }}
                    >
                      {topic.meta.domain}
                    </span>
                  </div>

                  {/* Topic title */}
                  <h3
                    className="text-sm font-bold leading-snug"
                    style={{ color: 'var(--color-text-primary)' }}
                  >
                    {topic.meta.title}
                  </h3>

                  {/* Footer */}
                  <div className="mt-auto pt-4 flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
                      {topic.meta.durationMinutes} min
                    </span>
                    <span
                      className="text-xs font-semibold transition-all duration-200 opacity-0 translate-x-[-8px] group-hover:opacity-100 group-hover:translate-x-0"
                      style={{ color: topic.meta.domainColor }}
                    >
                      Start Lesson →
                    </span>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </section>
  );
}
