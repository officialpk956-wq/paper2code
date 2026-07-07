import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import { getMdxContent } from '@/lib/mdx';
import MdxRenderer from '@/components/MdxRenderer';
import WorkspacePaperClient from './WorkspacePaperClient';

export default async function PaperWorkspacePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  
  const mdxContent = getMdxContent('papers', id);
  
  if (mdxContent) {
    return (
      <div className="min-h-screen bg-transparent text-white overflow-y-auto">
        <div className="max-w-4xl mx-auto p-12">
          <Link href="/papers?tab=library" className="text-[#A3A3A3] hover:text-white text-[13px] flex items-center gap-1.5 mb-6 transition-colors inline-flex">
            <ArrowLeft size={16} /> Back to Library
          </Link>
          <MdxRenderer source={mdxContent} />
        </div>
      </div>
    );
  }
  
  return <WorkspacePaperClient id={id} />;
}
