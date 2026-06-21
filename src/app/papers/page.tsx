import { getAllContent, getAllSlugs } from '@/lib/content/loader';
import type { PaperMeta, ImplementationMeta } from '@/lib/content/schemas';
import { getBackendUrl } from '@/lib/backend';
import { ResearchHub } from '@/components/research/ResearchHub';
import type { UploadedPaper } from '@/components/research/ResearchHub';

interface BackendPapersResponse {
  statistics?: {
    total_papers?: number;
    processed_papers?: number;
  };
  papers?: UploadedPaper[];
}

export default async function PapersPage() {
  // ── Content (filesystem, always available) ──────────────────────────────
  const contentPapers = getAllContent<PaperMeta>('paper').map(p => ({
    slug: p.meta.slug,
    title: p.meta.title,
    description: p.meta.description,
    authors: p.meta.authors,
    year: p.meta.year,
    venue: p.meta.venue,
    difficulty: p.meta.difficulty,
    tags: p.meta.tags,
  }));

  const implementations = getAllContent<ImplementationMeta>('implementation').map(i => ({
    slug: i.meta.slug,
    title: i.meta.title,
    description: i.meta.description,
    difficulty: i.meta.difficulty,
    tags: i.meta.tags,
    paperSlug: i.meta.paperSlug,
  }));

  const archCount = getAllSlugs('architecture').length;
  const implCount = getAllSlugs('implementation').length;
  const sysDesignCount = getAllSlugs('system-design').length;
  const paperLibCount = contentPapers.length;

  // ── Uploaded papers (FastAPI, may be unavailable) ──────────────────────
  let uploadedPapers: UploadedPaper[] = [];
  try {
    const res = await fetch(getBackendUrl('/api/papers'), {
      cache: 'no-store',
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      const data: BackendPapersResponse = await res.json();
      uploadedPapers = data.papers ?? [];
    }
  } catch {
    // Backend offline — show empty upload state
  }

  return (
    <ResearchHub
      uploadedPapers={uploadedPapers}
      contentPapers={contentPapers}
      implementations={implementations}
      archCount={archCount}
      implCount={implCount}
      sysDesignCount={sysDesignCount}
      paperLibCount={paperLibCount}
    />
  );
}
