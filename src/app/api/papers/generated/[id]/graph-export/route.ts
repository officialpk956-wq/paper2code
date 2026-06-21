import { NextRequest, NextResponse } from 'next/server';
import { getBackendUrl } from '@/lib/backend';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
): Promise<NextResponse> {
  const { id } = await params;

  if (!id || !/^\d+$/.test(id)) {
    return NextResponse.json({ error: 'Invalid paper id' }, { status: 400 });
  }

  const format = req.nextUrl.searchParams.get('format') ?? 'json';

  if (!['json', 'mermaid', 'dot'].includes(format)) {
    return NextResponse.json(
      { error: 'format must be json, mermaid, or dot' },
      { status: 400 },
    );
  }

  const backendRes = await fetch(
    getBackendUrl(`/api/papers/${id}/graph-export?format=${format}`),
    { cache: 'no-store' },
  );

  if (backendRes.status === 404) {
    return NextResponse.json({ error: 'Executable graph not found' }, { status: 404 });
  }

  if (!backendRes.ok) {
    return NextResponse.json(
      { error: `Backend error ${backendRes.status}` },
      { status: 502 },
    );
  }

  const text = await backendRes.text();
  const contentType =
    format === 'json' ? 'application/json' : 'text/plain';
  return new NextResponse(text, {
    headers: { 'Content-Type': contentType },
  });
}
