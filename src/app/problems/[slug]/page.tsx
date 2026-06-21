import { notFound } from "next/navigation";
import { PROBLEMS } from "@/data/problems";
import { ProblemDetailClient } from "@/components/problem-detail-client";

export default async function ProblemDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const problem = PROBLEMS.find((p) => p.slug === slug);

  if (!problem) {
    notFound();
  }

  return <ProblemDetailClient problem={problem} />;
}