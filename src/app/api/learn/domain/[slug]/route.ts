import { NextRequest, NextResponse } from 'next/server';
import { getDomainData } from '@/data/domains';

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  const { slug } = await params;
  const data = getDomainData(slug);

  if (!data) {
    return NextResponse.json(
      { error: `Domain "${slug}" not found` },
      { status: 404 }
    );
  }

  return NextResponse.json({ domain: data });
}
