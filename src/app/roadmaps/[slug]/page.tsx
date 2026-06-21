import { notFound } from "next/navigation";
import { getRoadmap, getAllSlugs } from "@/lib/content/loader";
import { ContentPageLayout } from "@/components/content/content-page-layout";

export function generateStaticParams() {
  return getAllSlugs("roadmap").map((slug) => ({ slug }));
}

export default async function RoadmapContentPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const item = getRoadmap(slug);
  if (!item) notFound();

  const { meta, body } = item;

  return (
    <ContentPageLayout
      meta={meta}
      body={body}
      facts={[
        { label: "Phases", value: String(meta.phases.length) },
        {
          label: "Linked assets",
          value: String(
            Object.values(meta.relationships ?? {}).reduce(
              (acc, slugs) => acc + (slugs?.length ?? 0),
              0
            )
          ),
        },
      ]}
    />
  );
}
