'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowRight, ArrowLeft } from 'lucide-react';
import { ARCHITECTURES } from '@/data/content/architectures';
import { findTopic, dojoSlugFor, papersForArch, LIBRARY_TO_WORKSPACE_ID } from '@/lib/crosslinks';

const ARCH_SECTIONS = [
  "Motivation",
  "Historical Context",
  "Problem Statement",
  "Intuition",
  "Mathematical Foundations",
  "Architecture Diagram",
  "Tensor Flow Walkthrough",
  "Layer-by-Layer Analysis",
  "Complexity Analysis",
  "Training Procedure",
  "PyTorch Implementation",
  "Inference Pipeline",
  "Production Deployment",
  "Strengths",
  "Weaknesses",
  "Research Evolution",
  "Interview Questions",
  "Practice Problems",
  "Related Papers",
  "Further Reading"
];

function SectionPlaceholder() {
  return (
    <div className="bg-[#121D16] border border-[#223429] border-dashed rounded-xl p-8 text-center my-6">
      <p className="text-[13px] text-[#A3A3A3] italic">
        Content in production — structure follows the Paper2Code chapter template
      </p>
    </div>
  );
}

const getDifficultyColor = (diff: string) => {
  switch(diff) {
    case 'Beginner': return '#4ADE80';
    case 'Intermediate': return '#FACC15';
    case 'Advanced': return '#F87171';
    case 'Expert': return '#A78BFA';
    default: return '#525252';
  }
};

const ARCH_MAP = new Map(ARCHITECTURES.map(a => [a.name.toLowerCase().trim(), a]));

export default function ArchitectureSlugPage() {
  const params = useParams();
  const slug = params.slug as string;

  const arch = ARCHITECTURES.find(a => a.slug === slug);

  if (!arch) {
    return (
      <div className="min-h-screen bg-transparent text-white flex flex-col items-center justify-center">
        <h1 className="text-2xl font-bold mb-4">Architecture not found</h1>
        <Link href="/architectures" className="text-[#A3E635] hover:underline flex items-center gap-2">
          <ArrowLeft size={16} /> Back to Library
        </Link>
      </div>
    );
  }

  const renderLineageToken = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || trimmed === '—') return null;
    const match = ARCH_MAP.get(trimmed.toLowerCase());
    if (match) {
      return (
        <Link 
          key={trimmed}
          href={`/architectures/${match.slug}`}
          className="px-2.5 py-1 bg-[#16241B] border border-[#223429] rounded-md text-[12px] text-white hover:border-[#A3E635]/50 hover:bg-[#A3E635]/10 transition-colors inline-block mr-2 mb-2"
        >
          {trimmed}
        </Link>
      );
    }
    return (
      <span key={trimmed} className="px-2.5 py-1 bg-[#16241B] border border-[#223429] rounded-md text-[12px] text-[#A3A3A3] inline-block mr-2 mb-2 cursor-default">
        {trimmed}
      </span>
    );
  };

  const parents = arch.parent ? arch.parent.split(',').map(s => s.trim()).filter(Boolean) : [];
  const derived = arch.derivedInto ? arch.derivedInto.split(',').map(s => s.trim()).filter(Boolean) : [];

  const relatedPapers = papersForArch(arch.name);
  const categorySiblings = ARCHITECTURES.filter(a => a.category === arch.category && a.slug !== arch.slug).slice(0, 8);
  const practiceSlug = dojoSlugFor(arch.name + ' ' + (arch.keyInnovation || ''));
  const learnTopic = findTopic(arch.name);

  return (
    <div className="flex h-[calc(100vh-56px)] bg-transparent text-white overflow-hidden">
      {/* LEFT SIDEBAR (Nav) */}
      <div className="w-[260px] border-r border-[#223429] bg-[#0E1811] flex-shrink-0 flex flex-col h-full">
        <div className="p-4 border-b border-[#223429]">
          <Link href="/architectures" className="text-[12px] text-[#A3A3A3] hover:text-white flex items-center gap-1.5 transition-colors">
            <ArrowLeft size={14} /> Back to Library
          </Link>
        </div>
        <div className="flex-1 overflow-y-auto p-4 space-y-1">
          {ARCH_SECTIONS.map((section, idx) => (
            <a key={section} href={`#section-${idx + 1}`} className="block text-[12px] text-[#A3A3A3] hover:text-white hover:bg-[#121D16] px-3 py-2 rounded-md transition-colors truncate">
              {idx + 1}. {section}
            </a>
          ))}
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="flex-1 overflow-y-auto scroll-smooth">
        <div className="max-w-4xl mx-auto p-12">
          {/* HEADER (SECTION 0) */}
          <div className="mb-12">
            <div className="flex items-center gap-3 mb-4">
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#A3E635]">
                {arch.category}
              </span>
              <span className="text-[#525252]">|</span>
              <span className="text-[11px] text-[#A3A3A3] font-medium">{arch.year}</span>
            </div>
            
            <h1 className="text-[40px] font-bold text-white mb-4 leading-tight">{arch.name}</h1>
            <div className="text-[14px] text-[#A3A3A3] mb-6">By {arch.authors}</div>
            
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 rounded-md text-[12px] font-bold uppercase tracking-wider" 
                style={{ backgroundColor: `${getDifficultyColor(arch.difficulty)}15`, color: getDifficultyColor(arch.difficulty) }}>
                {arch.difficulty}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            <div className="bg-[#0E1811] border border-[#223429] rounded-xl p-6">
              <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-3">Key Innovation</div>
              <div className="text-[14px] text-[#D4D4D4] leading-relaxed">{arch.keyInnovation}</div>
            </div>
            <div className="bg-[#0E1811] border border-[#223429] rounded-xl p-6">
              <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-3">Industry Usage</div>
              <div className="text-[14px] text-[#D4D4D4] leading-relaxed">{arch.industryUsage}</div>
            </div>
          </div>

          {/* LINEAGE */}
          {(parents.length > 0 || derived.length > 0) && (
            <div className="bg-[#121D16] border border-[#223429] rounded-xl p-6 mb-16">
              <h3 className="text-[14px] font-bold text-white mb-4">Architecture Lineage</h3>
              <div className="flex flex-col gap-4">
                {parents.length > 0 && (
                  <div>
                    <div className="text-[11px] font-medium text-[#525252] mb-2 uppercase">Parent Architectures</div>
                    <div className="flex flex-wrap">{parents.map(renderLineageToken)}</div>
                  </div>
                )}
                {derived.length > 0 && (
                  <div>
                    <div className="text-[11px] font-medium text-[#525252] mb-2 uppercase">What This Unlocks</div>
                    <div className="flex flex-wrap">{derived.map(renderLineageToken)}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* DYNAMIC SECTIONS */}
          <div className="space-y-16">
            {ARCH_SECTIONS.map((section, idx) => {
              const isRelatedPapers = section === "Related Papers";
              return (
                <section key={section} id={`section-${idx + 1}`} className="scroll-mt-16">
                  <h2 className="text-[22px] font-bold text-white mb-6 border-b border-[#223429] pb-2">
                    {idx + 1}. {section}
                  </h2>
                  
                  {isRelatedPapers ? (
                    relatedPapers.length > 0 ? (
                      <div className="flex flex-col gap-3">
                        {relatedPapers.map(p => {
                          const workspaceId = LIBRARY_TO_WORKSPACE_ID[p.slug];
                          const href = workspaceId ? `/papers/${workspaceId}` : `/papers?tab=library`;
                          return (
                            <Link key={p.slug} href={href} className="bg-[#121D16] border border-[#223429] p-4 rounded-xl flex items-center justify-between hover:border-[#A78BFA]/50 transition-colors">
                              <div className="flex items-center gap-4">
                                <span className="text-[#525252] font-mono text-[12px] font-medium w-8">#{p.rank}</span>
                                <span className="text-[14px] font-bold text-white">{p.title}</span>
                              </div>
                              {p.year && <span className="text-[12px] text-[#A3A3A3]">{p.year}</span>}
                            </Link>
                          );
                        })}
                      </div>
                    ) : <SectionPlaceholder />
                  ) : (
                    <SectionPlaceholder />
                  )}
                </section>
              );
            })}
          </div>
        </div>
      </div>

      {/* RIGHT SIDEBAR */}
      <div className="w-[280px] border-l border-[#223429] bg-[#0E1811] flex-shrink-0 flex flex-col h-full overflow-y-auto p-6 gap-8">
        
        {/* 1. Learn */}
        {learnTopic && (
          <div className="bg-gradient-to-br from-[#60A5FA]/10 to-transparent border border-[#60A5FA]/20 rounded-xl p-5">
            <div className="text-[13px] font-bold text-white mb-2">Learn Concepts</div>
            <p className="text-[12px] text-[#A3A3A3] mb-4">Study the theory and math behind {arch.name}.</p>
            <Link href={`/learn/${learnTopic.domainSlug}/${learnTopic.topicSlug}`} className="bg-[#60A5FA] text-black text-[12px] font-bold px-4 py-2 rounded-lg hover:brightness-110 transition-all flex items-center justify-center gap-1.5">
              Go to Lesson <ArrowRight size={14} />
            </Link>
          </div>
        )}

        {/* 2. Practice */}
        {practiceSlug && (
          <div className="bg-gradient-to-br from-[#A3E635]/10 to-transparent border border-[#A3E635]/20 rounded-xl p-5">
            <div className="text-[13px] font-bold text-white mb-2">Build from Scratch</div>
            <p className="text-[12px] text-[#A3A3A3] mb-4">Implement this architecture's core components.</p>
            <Link href={`/dojo/${practiceSlug}`} className="bg-[#A3E635] text-black text-[12px] font-bold px-4 py-2 rounded-lg hover:brightness-110 transition-all flex items-center justify-center gap-1.5">
              Practice in Dojo <ArrowRight size={14} />
            </Link>
          </div>
        )}

        {/* 3. Related Papers */}
        {relatedPapers.length > 0 && (
          <div>
            <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-4 border-b border-[#223429] pb-2">
              Related Papers
            </div>
            <div className="space-y-3">
              {relatedPapers.slice(0, 3).map(p => {
                const workspaceId = LIBRARY_TO_WORKSPACE_ID[p.slug];
                const href = workspaceId ? `/papers/${workspaceId}` : `/papers?tab=library`;
                return (
                  <Link key={p.slug} href={href} className="block group">
                    <div className="text-[13px] font-medium text-[#D4D4D4] group-hover:text-white transition-colors line-clamp-2">{p.title}</div>
                    <div className="text-[11px] text-[#525252] mt-0.5">{p.year}</div>
                  </Link>
                );
              })}
            </div>
          </div>
        )}

        {/* 4. Same Category */}
        {categorySiblings.length > 0 && (
          <div>
            <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-4 border-b border-[#223429] pb-2">
              Related: {arch.category}
            </div>
            <div className="space-y-3">
              {categorySiblings.map(sib => (
                <Link key={sib.slug} href={`/architectures/${sib.slug}`} className="block group">
                  <div className="text-[13px] font-medium text-[#D4D4D4] group-hover:text-white transition-colors">{sib.name}</div>
                  <div className="text-[11px] text-[#525252] mt-0.5">{sib.year}</div>
                </Link>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
