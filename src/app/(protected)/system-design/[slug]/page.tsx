import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { SD_SYSTEMS } from '@/data/content/systemDesign';
import { getMdxContent } from '@/lib/mdx';
import MdxRenderer from '@/components/MdxRenderer';
import ArchDiagram from '@/components/arch/ArchDiagram';
import { systemToFlowSlug } from '@/components/arch/archFlows';
import { Reveal } from '@/components/anim';

export default async function SystemDesignSlugPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const sys = SD_SYSTEMS.find(s => s.slug === slug);

  if (!sys) {
    return (
      <div className="min-h-screen bg-transparent text-white flex flex-col items-center justify-center">
        <h1 className="text-2xl font-bold mb-4">System not found</h1>
        <Link href="/system-design" className="text-[#A78BFA] hover:underline flex items-center gap-2">
          <ArrowLeft size={16} /> Back to System Design
        </Link>
      </div>
    );
  }

  const mdxContent = getMdxContent('system-design', slug);

  return (
    <div className="min-h-screen bg-transparent text-white flex flex-col">
      {/* HEADER */}
      <div className="border-b border-[#262626] bg-[#0A0A0A]">
        <div className="max-w-4xl mx-auto px-8 py-8">
          <Link href="/system-design" className="text-[#A3A3A3] hover:text-white text-[13px] flex items-center gap-1.5 mb-6 transition-colors inline-flex">
            <ArrowLeft size={16} /> Back to Systems
          </Link>
          <div className="text-[12px] font-bold text-[#A78BFA] uppercase tracking-wider mb-3">
            System {sys.number}
          </div>
          <h1 className="text-[40px] font-bold text-white mb-4 leading-tight">{sys.name}</h1>
        </div>
      </div>

      {/* CONTENT */}
      <div className="flex-1 max-w-4xl mx-auto w-full px-8 py-12">
        <Reveal className="mb-10">
          <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-4">
            System Architecture
          </div>
          <ArchDiagram slug={systemToFlowSlug(sys.slug + ' ' + sys.name)} />
        </Reveal>
        {mdxContent ? (
          <MdxRenderer source={mdxContent} />
        ) : (
          <div className="text-center py-20 text-[#A3A3A3] border border-[#262626] border-dashed rounded-xl bg-[#0A0A0A]">
            <p className="text-[16px] font-semibold text-white mb-2">Deep system content coming soon</p>
            <p className="text-[13px]">We are actively implementing the deep template for {sys.name}.</p>
          </div>
        )}
      </div>

      {/* FOOTER CTAs */}
      <div className="border-t border-[#262626] bg-[#0A0A0A] py-12 mt-auto">
        <div className="max-w-4xl mx-auto px-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <Link href="/architectures" className="bg-[#111111] border border-[#262626] p-6 rounded-xl hover:border-[#A78BFA]/50 transition-colors group flex items-center justify-between">
            <div>
              <div className="text-[16px] font-bold text-white group-hover:text-[#A78BFA] transition-colors mb-1">Browse Architectures</div>
              <div className="text-[13px] text-[#A3A3A3]">Explore ML models behind the systems.</div>
            </div>
            <ArrowRight size={20} className="text-[#525252] group-hover:text-[#A78BFA] transition-colors" />
          </Link>

          <Link href="/dojo" className="bg-[#111111] border border-[#262626] p-6 rounded-xl hover:border-[#A78BFA]/50 transition-colors group flex items-center justify-between">
            <div>
              <div className="text-[16px] font-bold text-white group-hover:text-[#A78BFA] transition-colors mb-1">Practice in Dojo</div>
              <div className="text-[13px] text-[#A3A3A3]">Implement core components from scratch.</div>
            </div>
            <ArrowRight size={20} className="text-[#525252] group-hover:text-[#A78BFA] transition-colors" />
          </Link>
        </div>
      </div>
    </div>
  );
}
