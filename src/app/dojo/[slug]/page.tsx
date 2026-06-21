import { notFound } from 'next/navigation';
import { PROBLEMS } from '@/data/problems';
import { DojoProblemPage } from '@/components/dojo/DojoProblemPage';

interface Props {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return PROBLEMS.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({ params }: Props) {
  const { slug } = await params;
  const problem = PROBLEMS.find((p) => p.slug === slug);
  if (!problem) return {};
  return {
    title: `${problem.title} — DS Dojo | Paper2Code`,
    description: problem.description,
  };
}

export default async function DojoProblemDetailPage({ params }: Props) {
  const { slug } = await params;
  const problem = PROBLEMS.find((p) => p.slug === slug);
  if (!problem) notFound();
  return <DojoProblemPage problem={problem} />;
}
