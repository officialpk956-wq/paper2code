'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, Clock, Key, ArrowRight } from 'lucide-react';
import { CURRICULUM } from '@/data/content/curriculum';
import { PREREQ_EDGES } from '@/data/content/prerequisites';
import { dojoSlugFor, findArch } from '@/lib/crosslinks';
import { FadeIn } from '@/components/FadeIn';
import AnimatedCard from '@/components/AnimatedCard';

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

          {topic.why && topic.why.trim() && (
            <div className="border-l-2 border-[#60A5FA] bg-[#60A5FA]/5 rounded-r-lg px-5 py-4 mb-8">
              <div className="text-[11px] font-bold text-[#60A5FA] uppercase tracking-wider mb-2">Why this matters</div>
              <p className="text-[15px] text-[#D4D4D4] leading-relaxed">{topic.why.trim()}</p>
            </div>
          )}

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
          <FadeIn>
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
          </FadeIn>
        )}

        {/* DYNAMIC SECTIONS - RENDERED FROM TS DATA */}
        <div className="space-y-12">
          {/* Motivation section */}
          {topic.why && (
            <FadeIn>
              <section id="motivation" className="bg-[#111111] border border-[#262626] rounded-xl p-8 relative overflow-hidden">
                <div className="absolute right-0 top-0 w-[300px] h-[300px] bg-gradient-to-bl from-[#60A5FA]/5 to-transparent pointer-events-none rounded-full filter blur-3xl" />
                <h2 className="text-[20px] font-bold text-white mb-4 border-b border-[#262626] pb-2 flex items-center gap-2">
                  <span className="w-1.5 h-6 rounded-full bg-[#60A5FA]" /> Why This Exists
                </h2>
                <div className="text-[16px] leading-[1.8] text-[#D4D4D4] italic border-l-2 border-[#60A5FA] pl-4">
                  "{topic.why}"
                </div>
              </section>
            </FadeIn>
          )}

          {/* Practice section */}
          {(() => {
            const dojoSlug = dojoSlugFor(topic.title);
            const arch = findArch(topic.title);
            if (!dojoSlug && !arch) return null;

            return (
              <FadeIn>
                <section id="practice" className="bg-[#111111] border border-[#262626] rounded-xl p-6">
                  <h2 className="text-[18px] font-bold text-white mb-4">Hands-on Practice & Reference Models</h2>
                  <p className="text-[13px] text-[#A3A3A3] mb-6">Apply your theoretical learning immediately with these practice items:</p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {dojoSlug && (
                      <AnimatedCard>
                        <Link href={`/dojo/${dojoSlug}`} className="bg-[#1A1A1A] border border-[#262626] p-5 rounded-xl block hover:border-[#60A5FA]/40 transition-colors h-full flex flex-col justify-between">
                          <div>
                            <div className="text-[10px] font-bold text-[#A78BFA] uppercase tracking-wider mb-1">Dojo Challenge</div>
                            <div className="text-[15px] font-bold text-white mb-2">{dojoSlug}</div>
                            <p className="text-[12px] text-[#A3A3A3] mb-4">Implement this concept from scratch in the browser coding dojo.</p>
                          </div>
                          <span className="text-[12px] text-[#60A5FA] font-medium flex items-center gap-1 mt-4">Practice Now <ArrowRight size={14} /></span>
                        </Link>
                      </AnimatedCard>
                    )}

                    {arch && (
                      <AnimatedCard>
                        <Link href={`/architectures/${arch.slug}`} className="bg-[#1A1A1A] border border-[#262626] p-5 rounded-xl block hover:border-[#A78BFA]/40 transition-colors h-full flex flex-col justify-between">
                          <div>
                            <div className="text-[10px] font-bold text-[#A78BFA] uppercase tracking-wider mb-1">Reference Architecture</div>
                            <div className="text-[15px] font-bold text-white mb-2">{arch.name}</div>
                            <p className="text-[12px] text-[#A3A3A3] mb-4">Read deep analysis, lineage, and implementation code of this architecture.</p>
                          </div>
                          <span className="text-[12px] text-[#A78BFA] font-medium flex items-center gap-1 mt-4">View Model Details <ArrowRight size={14} /></span>
                        </Link>
                      </AnimatedCard>
                    )}
                  </div>
                </section>
              </FadeIn>
            );
          })()}
        </div>
      </div>
    </div>
  );
}
