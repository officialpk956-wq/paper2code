import { NextRequest, NextResponse } from 'next/server';
import { getTopicData } from '@/data/topics';

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ domain: string; topic: string }> }
) {
  const { domain, topic } = await params;
  const data = getTopicData(domain, topic);

  if (!data) {
    return NextResponse.json(
      { error: `Topic "${topic}" not found in domain "${domain}"` },
      { status: 404 }
    );
  }

  return NextResponse.json({
    topic: {
      meta: data.meta,
      formulas: data.formulas,
      code: data.codeSnippets,
      papers: data.relatedPapers,
      roadmap: data.prerequisites,
      practice: data.practice,
    },
  });
}
