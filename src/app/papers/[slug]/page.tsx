import { notFound } from "next/navigation";
import { getPaper, getAllSlugs } from "@/lib/content/loader";
import { ContentPageLayout } from "@/components/content/content-page-layout";
import { PaperRelatedTopics } from "@/components/paper/PaperRelatedTopics";
import Link from "next/link";
import { Code2 } from "lucide-react";
import { getBackendUrl } from "@/lib/backend";

export function generateStaticParams() {
  return getAllSlugs("paper").map((slug) => ({ slug }));
}

export default async function PaperPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const item = getPaper(slug);
  if (!item) notFound();

  const { meta, body } = item;

  const res = await fetch(getBackendUrl(`/api/papers/${slug}/challenges`), {
    next: { revalidate: 60 }
  });
  let hasChallenges = false;
  if (res.ok) {
    const challenges = await res.json();
    hasChallenges = Array.isArray(challenges) && challenges.length > 0;
  }

  return (
    <div className="min-h-full overflow-y-auto h-full">
      <ContentPageLayout
        meta={meta}
        body={body}
        facts={[
          { label: "Authors", value: meta.authors.join(", ") },
          { label: "Year", value: String(meta.year) },
          ...(meta.venue ? [{ label: "Venue", value: meta.venue }] : []),
          ...(meta.arxivId ? [{ label: "arXiv", value: meta.arxivId }] : []),
          ...(meta.citations !== undefined
            ? [{ label: "Citations", value: meta.citations.toLocaleString() }]
            : []),
        ]}
      />
      <PaperRelatedTopics paperSlug={slug} />
      
      {/* Implement CTA */}
      {hasChallenges && (
        <div className="max-w-4xl mx-auto px-6 pb-24">
          <div className="p-6 rounded-xl flex items-center justify-between"
            style={{ 
              background: 'linear-gradient(135deg, rgba(124,58,237,0.1), rgba(6,182,212,0.1))',
              border: '1px solid rgba(139,92,246,0.2)'
            }}>
            <div>
              <h3 className="text-lg font-bold text-white mb-1">Ready to implement?</h3>
              <p className="text-sm text-slate-400">Put theory into practice by coding the core concepts of this paper.</p>
            </div>
            <Link
              href={`/papers/${slug}/implement`}
              className="flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-bold text-white transition-all hover:scale-105"
              style={{ background: 'linear-gradient(135deg, #7C3AED, #06B6D4)', boxShadow: '0 4px 12px rgba(124,58,237,0.3)' }}
            >
              <Code2 size={16} />
              <span>Implement Paper</span>
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
