import Link from 'next/link';
import { redirect } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import { getMdxContent } from '@/lib/mdx';
import MdxRenderer from '@/components/MdxRenderer';
import WorkspacePaperClient from './WorkspacePaperClient';
import ArchDiagramView from '@/components/arch/ArchDiagramView';
import { paperToArchSlug } from '@/components/arch/archFlows';
import { Reveal } from '@/components/anim';
import { PAPER_ROUTE_ALIASES } from '@/data/content/routeAliases';

export default async function PaperWorkspacePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const canonicalId = PAPER_ROUTE_ALIASES[id];
  if (canonicalId) redirect(`/papers/${canonicalId}`);
  
  const mdxContent = getMdxContent('papers', id);

  if (mdxContent) {
    const archSlug = paperToArchSlug(id);
    return (
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
          <MdxRenderer source={mdxContent} />
        </div>
      </div>
    );
  }
  
  return <WorkspacePaperClient id={id} />;
}
