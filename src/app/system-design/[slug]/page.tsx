import { notFound } from "next/navigation";
import { getSystemDesign, getAllSlugs } from "@/lib/content/loader";
import { ContentPageLayout } from "@/components/content/content-page-layout";

export function generateStaticParams() {
  return getAllSlugs("system-design").map((slug) => ({ slug }));
}

export default async function SystemDesignPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const item = getSystemDesign(slug);
  if (!item) notFound();

  const { meta, body } = item;

  return (
    <ContentPageLayout
      meta={meta}
      body={body}
      facts={[
        ...(meta.scale ? [{ label: "Scale", value: meta.scale }] : []),
        ...(meta.components.length > 0
          ? [{ label: "Components", value: meta.components.join(" · ") }]
          : []),
      ]}
    />
  );
}
