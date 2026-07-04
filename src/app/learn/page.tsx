'use client';

import Link from 'next/link';
import { CURRICULUM } from '@/data/content/curriculum';

export default function LearnPage() {
  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-[26px] font-bold text-white mb-2">Curriculum</h1>
        <p className="text-[13px] text-[#A3A3A3] mb-8">
          A structured path from basics to advanced research.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {CURRICULUM.map(domain => {
            const topicCount = domain.topics.length;
            const beginnerCount = domain.topics.filter(t => t.level === 'Beginner').length;
            const intermediateCount = domain.topics.filter(t => t.level === 'Intermediate').length;
            const advancedCount = domain.topics.filter(t => t.level === 'Advanced').length;
            const expertCount = domain.topics.filter(t => t.level === 'Expert').length;

            return (
              <Link 
                href={`/learn/${domain.slug}`}
                key={domain.slug}
                className="bg-[#111] border border-[#262626] rounded-xl p-5 hover:border-[#60A5FA]/40 transition-colors flex flex-col"
              >
                <div className="text-[11px] font-semibold text-[#60A5FA] mb-2 uppercase tracking-wider">
                  Domain {domain.number}
                </div>
                <h2 className="text-[18px] font-bold text-white mb-4 leading-tight">{domain.name}</h2>
                <div className="mt-auto">
                  <div className="flex justify-between items-end mb-2">
                    <span className="text-[12px] font-medium text-[#A3A3A3]">{topicCount} Topics</span>
                  </div>
                  
                  {/* 4-segment level bar */}
                  <div className="flex gap-1 h-1.5 w-full">
                    {topicCount > 0 ? (
                      <>
                        <div className="h-full bg-[#4ADE80] rounded-l-sm" style={{ width: `${(beginnerCount / topicCount) * 100}%` }} title={`Beginner: ${beginnerCount}`} />
                        <div className="h-full bg-[#FACC15]" style={{ width: `${(intermediateCount / topicCount) * 100}%` }} title={`Intermediate: ${intermediateCount}`} />
                        <div className="h-full bg-[#F87171]" style={{ width: `${(advancedCount / topicCount) * 100}%` }} title={`Advanced: ${advancedCount}`} />
                        <div className="h-full bg-[#A78BFA] rounded-r-sm" style={{ width: `${(expertCount / topicCount) * 100}%` }} title={`Expert: ${expertCount}`} />
                      </>
                    ) : (
                      <div className="h-full w-full bg-[#262626] rounded-sm" />
                    )}
                  </div>
                  <div className="flex justify-between text-[10px] text-[#525252] mt-1">
                    <span>Beg</span>
                    <span>Int</span>
                    <span>Adv</span>
                    <span>Exp</span>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
