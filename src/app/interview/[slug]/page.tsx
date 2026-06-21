import { notFound } from "next/navigation";
import { getInterviewQuestion, getAllSlugs } from "@/lib/content/loader";
import { ContentPageLayout } from "@/components/content/content-page-layout";

export function generateStaticParams() {
  return getAllSlugs("interview").map((slug) => ({ slug }));
}

export default async function InterviewQuestionPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const item = getInterviewQuestion(slug);
  if (!item) notFound();

  const { meta, body } = item;

  return (
    <ContentPageLayout
      meta={meta}
      body={body}
      facts={[
        { label: "Question type", value: meta.questionType.replace(/-/g, " ") },
        ...(meta.companies.length > 0
          ? [{ label: "Asked at", value: meta.companies.join(", ") }]
          : []),
        ...(meta.followUps.length > 0
          ? [{ label: "Follow-ups", value: String(meta.followUps.length) }]
          : []),
      ]}
    />
  );
}
