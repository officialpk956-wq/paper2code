import { notFound } from "next/navigation";
import { getPaper, getAllSlugs } from "@/lib/content/loader";
import { ContentPageLayout } from "@/components/content/content-page-layout";

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

  return (
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
  );
}
