import { NextResponse } from 'next/server';
import { getBackendUrl } from '@/lib/backend';

export async function GET() {
  try {
    const response = await fetch(getBackendUrl('/api/papers'), {
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
    });

    if (!response.ok) {
      const body = await response.text();
      return NextResponse.json({ error: body }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ error: 'Backend unavailable' }, { status: 503 });
  }
}
