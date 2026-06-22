import Link from "next/link";
import { ContentType } from "@/lib/content/schemas";
import {
  getRelatedContent,
  contentUrl,
} from "@/lib/content/relationships";

const TYPE_LABELS: Record<ContentType, { label: string; icon: string }> = {
  paper: { label: "Related Papers", icon: "📄" },
  architecture: { label: "Related Architectures", icon: "⚡" },
  math: { label: "Related Math", icon: "∑" },
  problem: { label: "Related Problems", icon: "💻" },
  "system-design": { label: "Related System Designs", icon: "🏗️" },
  interview: { label: "Related Interview Questions", icon: "🎤" },
  roadmap: { label: "Featured In Roadmaps", icon: "🗺️" },
  implementation: { label: "Implementation Journey", icon: "🔧" },
  "tensor-trace": { label: "Tensor Trace", icon: "🔍" },
};

const DIFFICULTY_STYLES: Record<string, string> = {
  beginner: "bg-[rgb(16,185,129,0.15)] text-[#6EE7B7]",
  intermediate: "bg-[rgb(245,158,11,0.15)] text-[#FCD34D]",
  advanced: "bg-[rgb(239,68,68,0.15)] text-[#FCA5A5]",
};

/**
 * Automatically rendered on every content page. Resolves both forward edges
 * (declared in this item's metadata) and reverse edges (items that declare
 * this one) from the relationship graph.
 */
export function RelatedContent({
  type,
  slug,
}: {
  type: ContentType;
  slug: string;
}) {
  const groups = getRelatedContent(type, slug);
  if (groups.length === 0) return null;

  return (
    <section className="mt-16 pt-8 border-t border-[--color-border]">
      <h2 className="text-2xl font-bold mb-8">Connected Learning</h2>

      <div className="space-y-8">
        {groups.map((group) => {
          const config = TYPE_LABELS[group.type];
          return (
            <div key={group.type}>
              <h3 className="text-sm font-bold uppercase tracking-wide text-[--color-text-tertiary] mb-3 flex items-center gap-2">
                <span>{config.icon}</span>
                {config.label}
              </h3>
              <div className="grid sm:grid-cols-2 gap-3">
                {group.items.map((item) => (
                  <Link
                    key={item.slug}
                    href={contentUrl(item)}
                    className="group p-4 rounded-lg bg-[--bg-surface] border border-[--color-border] hover:border-[--accent-primary] transition-all"
                  >
                    <div className="flex items-start justify-between gap-3 mb-1">
                      <h4 className="font-semibold text-sm group-hover:text-[--accent-primary] transition-colors">
                        {item.title}
                      </h4>
                      <span
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded-full flex-shrink-0 ${
                          DIFFICULTY_STYLES[item.difficulty]
                        }`}
                      >
                        {item.difficulty}
                      </span>
                    </div>
                    <p className="text-xs text-[--color-text-tertiary] line-clamp-2">
                      {item.description}
                    </p>
                  </Link>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
