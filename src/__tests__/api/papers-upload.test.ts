// @vitest-environment node
import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.stubGlobal('fetch', vi.fn());

async function postUpload(formData: FormData) {
  const { POST } = await import('@/app/api/papers/upload/route');
  const request = new Request('http://localhost/api/papers/upload', {
    method: 'POST',
    body: formData,
  });
  return POST(request as never);
}

describe('POST /api/papers/upload', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns 400 when file is missing', async () => {
    const res = await postUpload(new FormData());
    expect(res.status).toBe(400);
  });

  it('returns 400 for non-PDF uploads', async () => {
    const formData = new FormData();
    formData.append('file', new File(['hello'], 'notes.txt', { type: 'text/plain' }));

    const res = await postUpload(formData);
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.error).toMatch(/PDF/);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('forwards the upload to the backend service', async () => {
    const backendPayload = {
      paper_id: 42,
      title: 'attention-is-all-you-need',
      classification: 'Transformer',
      status: 'Draft',
      report: { nodes: 12, edges: 11, modules: 6, figure_count: 3, equation_count: 4 },
    };

    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(JSON.stringify(backendPayload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );

    const formData = new FormData();
    formData.append('file', new File(['%PDF-1.7'], 'paper.pdf', { type: 'application/pdf' }));
    formData.append('paperName', 'Attention Is All You Need');

    const res = await postUpload(formData);
    expect(res.status).toBe(200);

    const data = await res.json();
    expect(data.paper_id).toBe(42);
    expect(fetch).toHaveBeenCalledTimes(1);

    const fetchArgs = vi.mocked(fetch).mock.calls[0];
    expect(String(fetchArgs[0])).toContain('/api/papers/upload');

    const forwardedBody = fetchArgs[1]?.body as FormData;
    expect(forwardedBody.get('paperName')).toBe('Attention Is All You Need');
  });
});
