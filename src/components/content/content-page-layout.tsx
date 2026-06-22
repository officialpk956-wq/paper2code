import { ContentMeta, ContentType } from "@/lib/content/schemas";
import { Breadcrumb, BreadcrumbItem } from "@/components/breadcrumb";
import { MarkdownRenderer } from "./markdown-renderer";
import { RelatedContent } from "./related-content";
import { slugifyHeading } from "@/lib/slugify";
import Link from "next/link";
import { GitMerge, SkipForward, Code2 } from "lucide-react";
import { getAllImplementations } from "@/lib/content/loader";
/** Extract `## ` headings from the MDX body for the "On this page" TOC. */
function extractSections(body: string): Array<{ id: string; label: string }> {
  const sections: Array<{ id: string; label: string }> = [];
  for (const line of body.split("\n")) {
    const match = line.match(/^##\s+(.+)$/);
    if (match) {
      sections.push({ id: slugifyHeading(match[1]), label: match[1] });
    }
  }
  return sections;
}

const TYPE_META: Record<
  ContentType,
  { label: string; sectionHref: string; sectionLabel: string }
> = {
  architecture: {
    label: "Architecture",
    sectionHref: "/architectures",
    sectionLabel: "Architectures",
  },
  paper: { label: "Paper", sectionHref: "/papers", sectionLabel: "Papers" },
  math: { label: "Math Topic", sectionHref: "#", sectionLabel: "Math" },
  "system-design": {
    label: "System Design",
    sectionHref: "/system-design",
    sectionLabel: "System Design",
  },
  problem: {
    label: "Problem",
    sectionHref: "/problems",
    sectionLabel: "Problems",
  },
  interview: {
    label: "Interview",
    sectionHref: "#",
    sectionLabel: "Interview",
  },
  roadmap: {
    label: "Roadmap",
    sectionHref: "/roadmaps",
    sectionLabel: "Roadmaps",
  },
  implementation: {
    label: "Implementation",
    sectionHref: "/paper-to-code",
    sectionLabel: "Paper → Code",
  },
  "tensor-trace": {
    label: "Tensor Trace",
    sectionHref: "/tensor-trace",
    sectionLabel: "Tensor Trace",
  },
};

const DIFFICULTY_STYLES: Record<string, string> = {
  beginner: "bg-[rgb(16,185,129,0.15)] text-[#6EE7B7] border-[rgb(16,185,129,0.3)]",
  intermediate:
    "bg-[rgb(245,158,11,0.15)] text-[#FCD34D] border-[rgb(245,158,11,0.3)]",
  advanced: "bg-[rgb(239,68,68,0.15)] text-[#FCA5A5] border-[rgb(239,68,68,0.3)]",
};

interface ContentPageLayoutProps {
  meta: ContentMeta;
  body: string;
  /** Type-specific header facts, rendered as a chip row under the title. */
  facts?: Array<{ label: string; value: string }>;
}

/**
 * Generic template for every content detail page. Type-specific pages only
 * supply the facts row — everything else (breadcrumb, hero, tags, MDX body,
 * related content) is shared.
 */
export function ContentPageLayout({ meta, body, facts }: ContentPageLayoutProps) {
  const typeMeta = TYPE_META[meta.type];
  const sections = extractSections(body);

  const breadcrumbs: BreadcrumbItem[] = [
    { label: typeMeta.sectionLabel, href: typeMeta.sectionHref },
    { label: meta.title, current: true },
  ];

  // Check if there is an implementation journey for this content
  const impls = getAllImplementations();
  const relatedImpl = impls.find(i => 
    i.meta.paperSlug === meta.slug || 
    i.meta.relationships.papers?.includes(meta.slug) || 
    i.meta.relationships.architectures?.includes(meta.slug)
  );

  return (
    <div>
      {/* Hero */}
      <div className="border-b border-[--color-border] bg-[--bg-surface]">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Breadcrumb items={breadcrumbs} />

          <div className="flex items-center gap-2 mb-4">
            <span className="text-xs font-bold uppercase tracking-wider text-[--accent-cyan] bg-[rgb(6,182,212,0.1)] border border-[rgb(6,182,212,0.3)] px-2.5 py-1 rounded-full">
              {typeMeta.label}
            </span>
            <span
              className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${
                DIFFICULTY_STYLES[meta.difficulty]
              }`}
            >
              {meta.difficulty}
            </span>
          </div>

          <h1 className="text-4xl font-bold mb-3">{meta.title}</h1>
          <p className="text-lg text-[--color-text-secondary] mb-5">
            {meta.description}
          </p>

          {facts && facts.length > 0 && (
            <div className="flex flex-wrap gap-x-6 gap-y-2 mb-5">
              {facts.map((fact) => (
                <div key={fact.label} className="text-sm">
                  <span className="text-[--color-text-tertiary]">
                    {fact.label}:{" "}
                  </span>
                  <span className="font-semibold text-[--color-text-secondary]">
                    {fact.value}
                  </span>
                </div>
              ))}
            </div>
          )}

          {meta.tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {meta.tags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs text-[--color-text-tertiary] bg-[--bg-body] border border-[--color-border] px-2.5 py-1 rounded-full"
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}

          {/* Evolution Engine Integration */}
          {(meta.type === "paper" || meta.type === "architecture") && (
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link 
                href={`/compare?a=${meta.slug}`} 
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[--bg-surface] border border-[--accent-primary] text-[--accent-light] text-sm font-semibold hover:bg-[--accent-primary] hover:text-white transition-colors"
              >
                <GitMerge className="w-4 h-4" /> Compare With...
              </Link>
              <Link 
                href={`/evolution`} 
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[--bg-surface] border border-[--color-border] text-[--color-text-secondary] text-sm font-semibold hover:border-[--accent-cyan] hover:text-[--accent-cyan] transition-colors"
              >
                <SkipForward className="w-4 h-4" /> Evolution Path
              </Link>
              
              {/* Paper to Code Integration */}
              {relatedImpl && (
                <Link 
                  href={`/paper-to-code/${relatedImpl.meta.slug}`} 
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm font-semibold hover:bg-emerald-500 hover:text-white hover:border-emerald-500 transition-colors"
                >
                  <Code2 className="w-4 h-4" /> Implementation Journey
                </Link>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Body + TOC + related content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="flex gap-12">
          <div className="flex-1 min-w-0 max-w-4xl">
            <MarkdownRenderer body={body} />
            <RelatedContent type={meta.type} slug={meta.slug} />
          </div>

          {sections.length > 2 && (
            <aside className="hidden xl:block w-56 flex-shrink-0">
              <nav className="sticky top-24">
                <p className="text-xs font-bold uppercase tracking-wide text-[--color-text-tertiary] mb-3">
                  On this page
                </p>
                <ul className="space-y-1.5 border-l border-[--color-border]">
                  {sections.map((section) => (
                    <li key={section.id}>
                      <a
                        href={`#${section.id}`}
                        className="block pl-3 -ml-px border-l border-transparent text-[13px] text-[--color-text-tertiary] hover:text-[--color-text-primary] hover:border-[--accent-primary] transition-colors leading-snug py-0.5"
                      >
                        {section.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </nav>
            </aside>
          )}
        </div>
      </div>
    </div>
  );
}
