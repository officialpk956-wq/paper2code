import { NextRequest, NextResponse } from 'next/server';
import { getBackendUrl } from '@/lib/backend';

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
): Promise<NextResponse> {
  const { id } = await params;

  if (!id || !/^\d+$/.test(id)) {
    return NextResponse.json({ error: 'Invalid paper id' }, { status: 400 });
  }

  const backendRes = await fetch(getBackendUrl(`/api/papers/${id}`), {
    cache: 'no-store',
  });

  if (backendRes.status === 404) {
    return NextResponse.json({ error: 'Paper not found' }, { status: 404 });
  }

  if (!backendRes.ok) {
    return NextResponse.json(
      { error: `Backend error ${backendRes.status}` },
      { status: 502 },
    );
  }

  const data: unknown = await backendRes.json();
  return NextResponse.json(data);
}
