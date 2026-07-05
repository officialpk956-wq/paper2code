'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Clock, Key, ArrowRight } from 'lucide-react';
import { CURRICULUM } from '@/data/content/curriculum';
import { PREREQ_EDGES } from '@/data/content/prerequisites';

import { dojoSlugFor, findArch } from '@/lib/crosslinks';

const LEARN_SECTIONS = [
  "Why This Exists",
  "Real World Problem",
  "Historical Context",
  "Intuition",
  "Visual Explanation",
  "Mathematical Derivation",
  "Interactive Example",
  "Code Example",
  "Production Usage",
  "Common Mistakes",
  "Interview Questions",
  "Practice Problems",
  "Research Extensions"
];

function SectionPlaceholder() {
  return (
    <div className="bg-[#111111] border border-[#262626] border-dashed rounded-xl p-8 text-center my-6">
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

const ALL_TOPICS = CURRICULUM.flatMap(d => d.topics.map(t => ({ ...t, domainSlug: d.slug })));
const TOPIC_MAP = new Map(ALL_TOPICS.map(t => [t.title.trim().toLowerCase(), t]));

export default function TopicPage() {
  const params = useParams();
  const domainSlug = params.domain as string;
  const topicSlug = params.topic as string;

  const domain = CURRICULUM.find(d => d.slug === domainSlug);
  const topic = domain?.topics.find(t => t.slug === topicSlug);

  if (!domain || !topic) {
    return (
      <div className="min-h-screen bg-transparent text-white flex flex-col items-center justify-center">
        <h1 className="text-2xl font-bold mb-4">Topic not found</h1>
        <Link href={`/learn/${domainSlug || ''}`} className="text-[#60A5FA] hover:underline flex items-center gap-2">
          <ArrowLeft size={16} /> Back
        </Link>
      </div>
    );
  }

  const renderChip = (text: string, isPrereq: boolean) => {
    const trimmed = text.trim();
    const matched = TOPIC_MAP.get(trimmed.toLowerCase());
    
    if (matched) {
      return (
        <Link 
          key={text} 
          href={`/learn/${matched.domainSlug}/${matched.slug}`} 
          className="bg-[#111111] border border-[#262626] rounded-md px-3 py-1.5 text-[12px] text-white hover:border-[#60A5FA]/50 hover:bg-[#60A5FA]/10 transition-colors flex items-center gap-1.5 cursor-pointer mr-2 mb-2 inline-flex"
        >
          {isPrereq ? <Key size={12} className="text-[#525252]" /> : <ArrowRight size={12} className="text-[#525252]" />}
          {trimmed}
        </Link>
      );
    }
    return (
      <div key={text} className="bg-[#111111] border border-[#262626] rounded-md px-3 py-1.5 text-[12px] text-[#A3A3A3] flex items-center gap-1.5 cursor-default mr-2 mb-2 inline-flex">
        {isPrereq ? <Key size={12} className="text-[#525252]" /> : <ArrowRight size={12} className="text-[#525252]" />}
        {trimmed}
      </div>
    );
  };

  // Find prerequisites chain
  const prereqEdgesTo = PREREQ_EDGES.filter(e => e.to.toLowerCase() === topic.title.trim().toLowerCase());
  const prereqEdgesFrom = PREREQ_EDGES.filter(e => e.from.toLowerCase() === topic.title.trim().toLowerCase());

  return (
    <div className="min-h-screen bg-transparent text-white">
      {/* Top Nav */}
      <div className="border-b border-[#262626] px-8 py-4 bg-[#0A0A0A] flex items-center gap-2 sticky top-0 z-10 text-[13px] font-medium">
        <Link href="/learn" className="text-[#A3A3A3] hover:text-white transition-colors">
          Learn
        </Link>
        <span className="text-[#525252]">/</span>
        <Link href={`/learn/${domain.slug}`} className="text-[#A3A3A3] hover:text-white transition-colors">
          {domain.name}
        </Link>
        <span className="text-[#525252]">/</span>
        <span className="text-white">{topic.title}</span>
      </div>

      <div className="max-w-4xl mx-auto px-8 py-12">
        {/* HEADER */}
        <div className="mb-12">
          <div className="text-[12px] font-bold text-[#60A5FA] uppercase tracking-wider mb-3">
            {domain.name} <span className="text-[#525252] mx-2">→</span> {topic.level}
          </div>
          
          <h1 className="text-[40px] font-bold text-white mb-6 leading-tight">{topic.title}</h1>
          
          <div className="flex items-center gap-3 mb-8">
            <span className="px-3 py-1 rounded-md text-[12px] font-bold uppercase tracking-wider" 
              style={{ backgroundColor: `${getDifficultyColor(topic.level)}15`, color: getDifficultyColor(topic.level) }}>
              {topic.level}
            </span>
            {topic.studyTime && (
              <span className="flex items-center gap-1.5 text-[12px] text-[#A3A3A3] bg-[#111111] border border-[#262626] px-3 py-1 rounded-md">
                <Clock size={14} /> {topic.studyTime}
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-[#0A0A0A] border border-[#262626] rounded-xl p-5">
              <div className="text-[12px] font-semibold text-[#A3A3A3] uppercase tracking-wider mb-4 flex items-center gap-2">
                <Key size={14} className="text-[#60A5FA]" /> Prerequisites
              </div>
              {topic.prerequisites.length > 0 ? (
                <div className="flex flex-wrap">
                  {topic.prerequisites.map(p => renderChip(p, true))}
                </div>
              ) : (
                <div className="text-[13px] text-[#525252] italic">None required</div>
              )}
            </div>

            <div className="bg-[#0A0A0A] border border-[#262626] rounded-xl p-5">
              <div className="text-[12px] font-semibold text-[#A3A3A3] uppercase tracking-wider mb-4 flex items-center gap-2">
                <ArrowRight size={14} className="text-[#4ADE80]" /> What This Unlocks
              </div>
              {topic.unlocks.length > 0 ? (
                <div className="flex flex-wrap">
                  {topic.unlocks.map(u => renderChip(u, false))}
                </div>
              ) : (
                <div className="text-[13px] text-[#525252] italic">End of path</div>
              )}
            </div>
          </div>
          
          {(() => {
            const dojoSlug = dojoSlugFor(topic.title);
            const arch = findArch(topic.title);
            
            const topicIndex = domain.topics.findIndex(t => t.slug === topic.slug);
            const prevTopic = topicIndex > 0 ? domain.topics[topicIndex - 1] : null;
            const nextTopic = topicIndex < domain.topics.length - 1 ? domain.topics[topicIndex + 1] : null;
            
            if (!dojoSlug && !arch && !prevTopic && !nextTopic) return null;

            return (
              <div className="mt-8 pt-8 border-t border-[#262626] flex flex-wrap gap-3 items-center">
                <span className="text-[12px] font-bold text-[#525252] uppercase tracking-wider mr-2">Continue</span>
                
                {prevTopic && (
                  <Link href={`/learn/${domain.slug}/${prevTopic.slug}`} className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#111111] border border-[#262626] rounded-md text-[12px] font-medium text-[#A3A3A3] hover:text-white hover:border-[#525252] transition-colors">
                    ← {prevTopic.title}
                  </Link>
                )}
                
                {nextTopic && (
                  <Link href={`/learn/${domain.slug}/${nextTopic.slug}`} className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#111111] border border-[#262626] rounded-md text-[12px] font-medium text-[#A3A3A3] hover:text-white hover:border-[#525252] transition-colors">
                    {nextTopic.title} →
                  </Link>
                )}

                {dojoSlug && (
                  <Link href={`/dojo/${dojoSlug}`} className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#111111] border border-[#A78BFA]/30 rounded-md text-[12px] font-medium text-[#A78BFA] hover:bg-[#A78BFA]/10 transition-colors ml-auto">
                    Practice: {dojoSlug} →
                  </Link>
                )}

                {arch && (
                  <Link href={`/architectures/${arch.slug}`} className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#111111] border border-[#A78BFA]/30 rounded-md text-[12px] font-medium text-[#A78BFA] hover:bg-[#A78BFA]/10 transition-colors">
                    Architecture: {arch.name} →
                  </Link>
                )}
              </div>
            );
          })()}
        </div>

        {/* PREREQUISITE CHAIN */}
        {(prereqEdgesTo.length > 0 || prereqEdgesFrom.length > 0) && (
          <div className="bg-[#111111] border border-[#262626] rounded-xl p-6 mb-16">
            <h3 className="text-[14px] font-bold text-white mb-4">Prerequisite Chain</h3>
            <div className="flex flex-col gap-4">
              {prereqEdgesTo.length > 0 && (
                <div>
                  <div className="text-[11px] font-medium text-[#525252] mb-2 uppercase">Learn these first</div>
                  <div className="flex flex-wrap">
                    {prereqEdgesTo.map(e => renderChip(e.from, true))}
                  </div>
                </div>
              )}
              {prereqEdgesFrom.length > 0 && (
                <div>
                  <div className="text-[11px] font-medium text-[#525252] mb-2 uppercase">This unlocks</div>
                  <div className="flex flex-wrap">
                    {prereqEdgesFrom.map(e => renderChip(e.to, false))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* DYNAMIC SECTIONS */}
        <div className="space-y-16">
          {LEARN_SECTIONS.map((section, idx) => (
            <section key={section} id={`section-${idx + 1}`} className="scroll-mt-16">
              <h2 className="text-[22px] font-bold text-white mb-6 border-b border-[#262626] pb-2">
                {idx + 1}. {section}
              </h2>
              <SectionPlaceholder />
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
