import { notFound } from "next/navigation";
import { getMathTopic, getAllSlugs } from "@/lib/content/loader";
import { ContentPageLayout } from "@/components/content/content-page-layout";

export function generateStaticParams() {
  return getAllSlugs("math").map((slug) => ({ slug }));
}

export default async function MathTopicPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const item = getMathTopic(slug);
  if (!item) notFound();

  const { meta, body } = item;

  return (
    <ContentPageLayout
      meta={meta}
      body={body}
      facts={[
        { label: "Category", value: meta.category.replace(/-/g, " ") },
        ...(meta.keyFormula
          ? [{ label: "Key formula", value: meta.keyFormula }]
          : []),
      ]}
    />
  );
}
