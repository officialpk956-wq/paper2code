import type { Metadata } from 'next';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import { getMdxContent, getMethodology } from '@/lib/mdx';
import MdxRenderer from '@/components/MdxRenderer';
import MethodologyTrack from '@/components/paper/MethodologyTrack';
import WorkspacePaperClient from './WorkspacePaperClient';
import ArchDiagramView from '@/components/arch/ArchDiagramView';
import { paperToArchSlug } from '@/components/arch/archFlows';
import { Reveal } from '@/components/anim';
import { PAPER_ROUTE_ALIASES } from '@/data/content/routeAliases';
import { PAPERS } from '@/data/content/papers';
import { FLAGSHIP_META } from '@/data/flagship-papers';
import { AuthGuard } from '@/components/AuthGuard';
import { PageSummary } from '@/components/PageSummary';

// PAPERS (the 200-paper library) and FLAGSHIP_META (the 6 curated workspace
// papers) use different slug spellings for overlapping papers, so a real
// title/abstract can come from either — check both before falling back.
function getPaperSummaryInfo(id: string): { title: string; description: string } | null {
  const libraryEntry = PAPERS.find(p => p.slug === id);
  if (libraryEntry) {
    return {
      title: libraryEntry.title,
      description: libraryEntry.whyImportant || libraryEntry.conceptsIntroduced || `Read and implement "${libraryEntry.title}".`,
    };
  }
  const flagship = FLAGSHIP_META[id];
  if (flagship) {
    return { title: flagship.title, description: flagship.abstract };
  }
  return null;
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const canonicalId = PAPER_ROUTE_ALIASES[id] ?? id;
  const info = getPaperSummaryInfo(canonicalId);
  if (!info) {
    return {
      title: 'Paper Workspace | paper2code',
      description: 'Upload and analyze any ML paper: extract its architecture, generate runnable code, and get AI tutoring.',
    };
  }
  return { title: `${info.title} — Paper Deep Dive | paper2code`, description: info.description.slice(0, 155) };
}

function getPaperMethodology(id: string) {
  const direct = getMethodology(id);
  if (direct) return direct;

  for (const [alias, canonical] of Object.entries(PAPER_ROUTE_ALIASES)) {
    if (canonical === id) {
      const aliased = getMethodology(alias);
      if (aliased) return aliased;
    }
  }
  return null;
}

export default async function PaperWorkspacePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const canonicalId = PAPER_ROUTE_ALIASES[id];
  if (canonicalId) redirect(`/papers/${canonicalId}`);
  
  const mdxContent = getMdxContent('papers', id);
  const info = getPaperSummaryInfo(id);

  if (mdxContent) {
    const archSlug = paperToArchSlug(id);
    const methodology = getPaperMethodology(id);
    return (
      <>
        {info && <PageSummary eyebrow="Paper" title={info.title} description={info.description} />}
        <AuthGuard>
        <div className="min-h-screen bg-transparent text-white overflow-y-auto">
          <div className="max-w-4xl mx-auto p-12">
            <Link href="/papers?tab=library" className="text-[#A3A3A3] hover:text-white text-[13px] flex items-center gap-1.5 mb-6 transition-colors inline-flex">
              <ArrowLeft size={16} /> Back to Library
            </Link>
            {archSlug && (
              <Reveal className="mb-10">
                <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-4">
                  Architecture
                </div>
                <ArchDiagramView slug={archSlug} />
              </Reveal>
            )}
            {methodology && (
              <Reveal className="mb-10">
                <MethodologyTrack methodology={methodology} />
              </Reveal>
            )}
            <MdxRenderer source={mdxContent} />
          </div>
        </div>
        </AuthGuard>
      </>
    );
  }

  return (
    <>
      <PageSummary
        eyebrow="Paper Workspace"
        title={info?.title ?? 'Analyze Any ML Paper'}
        description={info?.description ?? 'Upload and analyze any ML paper: extract its architecture, generate runnable code, and get AI tutoring.'}
      />
      <AuthGuard>
        <WorkspacePaperClient id={id} />
      </AuthGuard>
    </>
  );
}
