'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { useState } from 'react';
import { ArrowLeft, ArrowRight, CheckSquare, Key, Image as ImageIcon, BookOpen, PenTool, MessageCircle } from 'lucide-react';
import { SD_SYSTEMS } from '@/data/content/systemDesign';
import { findTopic, findArch } from '@/lib/crosslinks';

export default function SystemDesignSlugPage() {
  const params = useParams();
  const slug = params.slug as string;
  const sys = SD_SYSTEMS.find(s => s.slug === slug);
  const [activeTab, setActiveTab] = useState<'Beginner' | 'Intermediate' | 'Advanced' | 'Research'>('Beginner');

  if (!sys) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-white flex flex-col items-center justify-center">
        <h1 className="text-2xl font-bold mb-4">System not found</h1>
        <Link href="/system-design" className="text-[#F97316] hover:underline flex items-center gap-2">
          <ArrowLeft size={16} /> Back to System Design
        </Link>
      </div>
    );
  }

  const currentModule = sys.modules.find(m => m.level === activeTab);
  const tabs = ['Beginner', 'Intermediate', 'Advanced', 'Research'] as const;

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white flex flex-col">
      {/* HEADER */}
      <div className="border-b border-[#262626] bg-[#0D0D0D]">
        <div className="max-w-5xl mx-auto px-8 py-8">
          <Link href="/system-design" className="text-[#A3A3A3] hover:text-white text-[13px] flex items-center gap-1.5 mb-6 transition-colors inline-flex">
            <ArrowLeft size={16} /> Back to Systems
          </Link>
          <div className="text-[12px] font-bold text-[#F97316] uppercase tracking-wider mb-3">
            System {sys.number}
          </div>
          <h1 className="text-[40px] font-bold text-white mb-8 leading-tight">{sys.name}</h1>
          
          {/* TABS */}
          <div className="flex items-center gap-2 border-b border-[#262626] w-full">
            {tabs.map(tab => {
              const hasModule = sys.modules.some(m => m.level === tab);
              const isActive = activeTab === tab;
              return (
                <button
                  key={tab}
                  disabled={!hasModule}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-3 text-[14px] font-medium transition-colors border-b-2 ${
                    isActive 
                      ? 'border-[#F97316] text-[#F97316]' 
                      : hasModule 
                        ? 'border-transparent text-[#A3A3A3] hover:text-white' 
                        : 'border-transparent text-[#525252] cursor-not-allowed opacity-50'
                  }`}
                >
                  {tab}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* CONTENT */}
      <div className="flex-1 max-w-5xl mx-auto w-full px-8 py-12">
        {currentModule ? (
          <div className="grid grid-cols-1 gap-8">
            {/* Learning Objectives */}
            {currentModule.learningObjectives.length > 0 && (
              <div className="bg-[#111] border border-[#262626] rounded-xl p-6 md:p-8">
                <h2 className="text-[18px] font-bold text-white mb-6 flex items-center gap-2 pb-4 border-b border-[#262626]">
                  <CheckSquare size={20} className="text-[#4ADE80]" /> Learning Objectives
                </h2>
                <ul className="space-y-4">
                  {currentModule.learningObjectives.map((obj, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <div className="mt-1 flex-shrink-0 w-1.5 h-1.5 rounded-full bg-[#525252]" />
                      <span className="text-[15px] text-[#D4D4D4] leading-relaxed">{obj}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Prerequisites */}
            {currentModule.prerequisites.length > 0 && (
              <div className="bg-[#111] border border-[#262626] rounded-xl p-6 md:p-8">
                <h2 className="text-[18px] font-bold text-white mb-6 flex items-center gap-2 pb-4 border-b border-[#262626]">
                  <Key size={20} className="text-[#FACC15]" /> Prerequisites
                </h2>
                <div className="flex flex-wrap gap-2">
                  {currentModule.prerequisites.map((req, i) => {
                    const topic = findTopic(req);
                    if (topic) {
                      return (
                        <Link key={i} href={`/learn/${topic.domainSlug}/${topic.topicSlug}`} className="px-3 py-1.5 bg-[#1A1A1A] border border-[#333] rounded-md text-[13px] font-medium text-[#60A5FA] hover:border-[#60A5FA]/50 hover:bg-[#60A5FA]/10 transition-colors">
                          {req}
                        </Link>
                      );
                    }
                    const arch = findArch(req);
                    if (arch) {
                      return (
                        <Link key={i} href={`/architectures/${arch.slug}`} className="px-3 py-1.5 bg-[#1A1A1A] border border-[#333] rounded-md text-[13px] font-medium text-[#F97316] hover:border-[#F97316]/50 hover:bg-[#F97316]/10 transition-colors">
                          {req}
                        </Link>
                      );
                    }
                    return (
                      <span key={i} className="px-3 py-1.5 bg-[#1A1A1A] border border-[#333] rounded-md text-[13px] text-[#A3A3A3]">
                        {req}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Diagram Specs */}
            {currentModule.diagramsNeeded.length > 0 && (
              <div className="bg-[#111] border border-[#262626] rounded-xl p-6 md:p-8">
                <h2 className="text-[18px] font-bold text-white mb-6 flex items-center gap-2 pb-4 border-b border-[#262626]">
                  <ImageIcon size={20} className="text-[#A78BFA]" /> Architecture Diagrams
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {currentModule.diagramsNeeded.map((diag, i) => (
                    <div key={i} className="bg-[#1A1A1A] border border-[#333] p-4 rounded-lg flex flex-col h-full">
                      <div className="text-[10px] font-bold text-[#525252] uppercase tracking-wider mb-2">Diagram Spec</div>
                      <div className="text-[14px] text-[#D4D4D4] leading-relaxed">{diag}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Case Studies */}
            {currentModule.caseStudies.length > 0 && (
              <div className="bg-[#111] border border-[#262626] rounded-xl p-6 md:p-8">
                <h2 className="text-[18px] font-bold text-white mb-6 flex items-center gap-2 pb-4 border-b border-[#262626]">
                  <BookOpen size={20} className="text-[#60A5FA]" /> Case Studies
                </h2>
                <ul className="space-y-4">
                  {currentModule.caseStudies.map((cs, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <div className="mt-1 flex-shrink-0 w-1.5 h-1.5 rounded-full bg-[#525252]" />
                      <span className="text-[15px] text-[#D4D4D4] leading-relaxed">{cs}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Hands-On Projects */}
            {currentModule.handsOnProjects.length > 0 && (
              <div className="bg-[#111] border border-[#262626] rounded-xl p-6 md:p-8">
                <h2 className="text-[18px] font-bold text-white mb-6 flex items-center gap-2 pb-4 border-b border-[#262626]">
                  <PenTool size={20} className="text-[#F97316]" /> Hands-On Projects
                </h2>
                <div className="space-y-4">
                  {currentModule.handsOnProjects.map((proj, i) => (
                    <div key={i} className="flex gap-4 items-start bg-[#1A1A1A] border border-[#333] p-4 rounded-lg">
                      <div className="text-[#F97316] font-mono font-bold text-[14px] mt-0.5">{(i + 1).toString().padStart(2, '0')}</div>
                      <div className="text-[15px] text-[#D4D4D4] leading-relaxed">{proj}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Interview Questions */}
            {currentModule.interviewQuestions.length > 0 && (
              <div className="bg-[#111] border border-[#262626] rounded-xl p-6 md:p-8">
                <h2 className="text-[18px] font-bold text-white mb-6 flex items-center gap-2 pb-4 border-b border-[#262626]">
                  <MessageCircle size={20} className="text-[#F87171]" /> Interview Questions
                </h2>
                <ul className="space-y-4">
                  {currentModule.interviewQuestions.map((iq, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <div className="mt-1.5 flex-shrink-0 text-[#F87171]">Q:</div>
                      <span className="text-[15px] text-[#D4D4D4] font-medium leading-relaxed">{iq}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-20 text-[#A3A3A3]">
            No modules available for {activeTab} level yet.
          </div>
        )}
      </div>

      {/* FOOTER CTAs */}
      <div className="border-t border-[#262626] bg-[#0D0D0D] py-12 mt-auto">
        <div className="max-w-5xl mx-auto px-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <Link href="/architectures" className="bg-[#111] border border-[#262626] p-6 rounded-xl hover:border-[#F97316]/50 transition-colors group flex items-center justify-between">
            <div>
              <div className="text-[16px] font-bold text-white group-hover:text-[#F97316] transition-colors mb-1">Browse Architectures</div>
              <div className="text-[13px] text-[#A3A3A3]">Explore ML models behind the systems.</div>
            </div>
            <ArrowRight size={20} className="text-[#525252] group-hover:text-[#F97316] transition-colors" />
          </Link>

          <Link href="/dojo" className="bg-[#111] border border-[#262626] p-6 rounded-xl hover:border-[#F97316]/50 transition-colors group flex items-center justify-between">
            <div>
              <div className="text-[16px] font-bold text-white group-hover:text-[#F97316] transition-colors mb-1">Practice in Dojo</div>
              <div className="text-[13px] text-[#A3A3A3]">Implement core components from scratch.</div>
            </div>
            <ArrowRight size={20} className="text-[#525252] group-hover:text-[#F97316] transition-colors" />
          </Link>
        </div>
      </div>
    </div>
  );
}
