import type { Metadata } from 'next';
import { getMdxContent } from '@/lib/mdx';
import MdxRenderer from '@/components/MdxRenderer';
import DojoEditor from './DojoEditor';
import { getProblemBySlug, getProblemIndex } from '@/data/problems';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const problem = getProblemBySlug(slug);
  if (!problem) return { title: 'Problem Not Found | paper2code Dojo' };
  return {
    title: `${problem.title} (${problem.difficulty}) — Dojo | paper2code`,
    description: problem.description.slice(0, 155),
  };
}

export default async function Page({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const mdxSource = getMdxContent('problems', slug);
  const problem = getProblemBySlug(slug);
  const index = getProblemIndex(slug);

  return (
    <>
      {problem && (
        <PageSummary
          eyebrow={`Dojo #${String(index + 1).padStart(3, '0')} · ${problem.difficulty}`}
          title={problem.title}
          description={problem.description.split('\n')[0]}
        />
      )}
      <AuthGuard>
        <DojoEditor>
          {mdxSource ? <MdxRenderer source={mdxSource} /> : null}
        </DojoEditor>
      </AuthGuard>
    </>
  );
}
