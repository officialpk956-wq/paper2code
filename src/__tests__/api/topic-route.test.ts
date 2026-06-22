import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('@/data/topics', () => ({
  getTopicData: vi.fn(),
}));

import { GET } from '@/app/api/learn/topic/[domain]/[topic]/route';
import { getTopicData } from '@/data/topics';
import { NextRequest } from 'next/server';

const mockGetTopicData = vi.mocked(getTopicData);

function makeRequest(url: string): NextRequest {
  return new NextRequest(url);
}

const mockTopicData = {
  meta: {
    slug: 'attention',
    title: 'Attention Mechanism',
    subtitle: 'Core of Transformers.',
    difficulty: 'intermediate' as const,
    durationMinutes: 45,
    prerequisites: ['Linear Algebra'],
    domain: 'Deep Learning',
    domainSlug: 'deep-learning',
    domainColor: '#0EA5E9',
    tags: ['transformer'],
    completionPercent: 0,
  },
  formulas: [],
  codeSnippets: [],
  relatedPapers: [],
  prerequisites: ['Linear Algebra'],
  practice: [],
};

describe('GET /api/learn/topic/[domain]/[topic]', () => {
  beforeEach(() => {
    mockGetTopicData.mockReset();
  });

  it('returns 200 with topic data when found', async () => {
    mockGetTopicData.mockReturnValue(mockTopicData as never);
    const res = await GET(
      makeRequest('http://localhost/api/learn/topic/deep-learning/attention'),
      { params: { domain: 'deep-learning', topic: 'attention' } }
    );
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.topic.meta.slug).toBe('attention');
  });

  it('calls getTopicData with correct domain and topic', async () => {
    mockGetTopicData.mockReturnValue(mockTopicData as never);
    await GET(
      makeRequest('http://localhost/api/learn/topic/deep-learning/attention'),
      { params: { domain: 'deep-learning', topic: 'attention' } }
    );
    expect(mockGetTopicData).toHaveBeenCalledWith('deep-learning', 'attention');
  });

  it('returns 404 when topic not found', async () => {
    mockGetTopicData.mockReturnValue(null);
    const res = await GET(
      makeRequest('http://localhost/api/learn/topic/deep-learning/unknown'),
      { params: { domain: 'deep-learning', topic: 'unknown' } }
    );
    expect(res.status).toBe(404);
  });

  it('404 response includes error message', async () => {
    mockGetTopicData.mockReturnValue(null);
    const res = await GET(
      makeRequest('http://localhost/api/learn/topic/deep-learning/unknown'),
      { params: { domain: 'deep-learning', topic: 'unknown' } }
    );
    const json = await res.json();
    expect(json.error).toMatch(/unknown/i);
  });

  it('response includes formulas field', async () => {
    mockGetTopicData.mockReturnValue(mockTopicData as never);
    const res = await GET(
      makeRequest('http://localhost/api/learn/topic/deep-learning/attention'),
      { params: { domain: 'deep-learning', topic: 'attention' } }
    );
    const json = await res.json();
    expect(json.topic).toHaveProperty('formulas');
  });

  it('response includes code field', async () => {
    mockGetTopicData.mockReturnValue(mockTopicData as never);
    const res = await GET(
      makeRequest('http://localhost/api/learn/topic/deep-learning/attention'),
      { params: { domain: 'deep-learning', topic: 'attention' } }
    );
    const json = await res.json();
    expect(json.topic).toHaveProperty('code');
  });

  it('response includes papers field', async () => {
    mockGetTopicData.mockReturnValue(mockTopicData as never);
    const res = await GET(
      makeRequest('http://localhost/api/learn/topic/deep-learning/attention'),
      { params: { domain: 'deep-learning', topic: 'attention' } }
    );
    const json = await res.json();
    expect(json.topic).toHaveProperty('papers');
  });

  it('response includes practice field', async () => {
    mockGetTopicData.mockReturnValue(mockTopicData as never);
    const res = await GET(
      makeRequest('http://localhost/api/learn/topic/deep-learning/attention'),
      { params: { domain: 'deep-learning', topic: 'attention' } }
    );
    const json = await res.json();
    expect(json.topic).toHaveProperty('practice');
  });

  it('returns 404 for missing domain', async () => {
    mockGetTopicData.mockReturnValue(null);
    const res = await GET(
      makeRequest('http://localhost/api/learn/topic/missing-domain/attention'),
      { params: { domain: 'missing-domain', topic: 'attention' } }
    );
    expect(res.status).toBe(404);
  });
});
