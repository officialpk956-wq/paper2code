import { notFound } from "next/navigation";
import { getArchitecture, getAllSlugs } from "@/lib/content/loader";
import { ContentPageLayout } from "@/components/content/content-page-layout";
import { ArchitectureInteractiveModules } from "@/components/architecture-interactive-modules";
import { ArchitectureLearningTrack } from "@/components/architecture-learning-track";

export function generateStaticParams() {
  return getAllSlugs("architecture").map((slug) => ({ slug }));
}

export default async function ArchitecturePage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const item = getArchitecture(slug);
  if (!item) notFound();

  const { meta, body } = item;

  return (
    <>
      <ContentPageLayout
        meta={meta}
        body={body}
        facts={[
          { label: "Introduced", value: String(meta.year) },
          ...(meta.authors.length > 0
            ? [{ label: "Authors", value: meta.authors.join(", ") }]
            : []),
          ...(meta.components.length > 0
            ? [{ label: "Key components", value: meta.components.join(" · ") }]
            : []),
        ]}
      />
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <ArchitectureInteractiveModules slug={slug} />
        <ArchitectureLearningTrack architectureSlug={slug} />
      </div>
    </>
  );
}
