import { NextRequest, NextResponse } from 'next/server';
import { getBackendUrl } from '@/lib/backend';

const MAX_SIZE = 20 * 1024 * 1024;

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file');

    if (!(file instanceof File)) {
      return NextResponse.json({ error: 'file is required' }, { status: 400 });
    }
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      return NextResponse.json({ error: 'Only PDF files are supported.' }, { status: 400 });
    }
    if (file.size > MAX_SIZE) {
      return NextResponse.json({ error: 'File exceeds 20MB limit.' }, { status: 400 });
    }

    const forwardBody = new FormData();
    forwardBody.append('file', file, file.name);

    const paperName = formData.get('paperName');
    if (typeof paperName === 'string' && paperName.trim()) {
      forwardBody.append('paperName', paperName.trim());
    }

    const response = await fetch(getBackendUrl('/api/papers/upload'), {
      method: 'POST',
      body: forwardBody,
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const message = payload?.detail ?? payload?.error ?? 'Upload failed';
      return NextResponse.json({ error: message }, { status: response.status });
    }

    return NextResponse.json(payload, { status: response.status });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: `Upload error: ${message}` }, { status: 500 });
  }
}
