'use client';

import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ChevronLeft, ArrowRight, Clock } from 'lucide-react';
import { CURRICULUM, CurriculumTopic } from '@/data/content/curriculum';

const getDifficultyColor = (diff: string) => {
  switch(diff) {
    case 'Beginner': return '#4ADE80';
    case 'Intermediate': return '#FACC15';
    case 'Advanced': return '#F87171';
    case 'Expert': return '#A78BFA';
    default: return '#525252';
  }
};

export default function DomainPage() {
  const params = useParams();
  const domainSlug = params.domain as string;
  
  const domain = CURRICULUM.find(d => d.slug === domainSlug);

  if (!domain) {
    return (
      <div className="min-h-screen bg-transparent text-white flex flex-col items-center justify-center">
        <h1 className="text-2xl font-bold mb-4">Domain not found</h1>
        <Link href="/learn" className="text-[#60A5FA] hover:underline">
          ← Back to Curriculum
        </Link>
      </div>
    );
  }

  // Group topics by level
  const groupedTopics: Record<string, CurriculumTopic[]> = {
    Beginner: [],
    Intermediate: [],
    Advanced: [],
    Expert: []
  };
  domain.topics.forEach(t => groupedTopics[t.level].push(t));

  return (
    <div className="min-h-screen bg-transparent text-white">
      {/* Top Nav */}
      <div className="border-b border-[#223429] px-8 py-6 bg-[#0E1811]">
        <Link href="/learn" className="text-[#A3A3A3] hover:text-white text-[13px] flex items-center gap-1.5 mb-6 transition-colors">
          <ChevronLeft size={16} /> Back to Curriculum
        </Link>
        <div className="text-[12px] font-semibold text-[#60A5FA] mb-2 tracking-wider">DOMAIN {domain.number}</div>
        <h1 className="text-[32px] font-bold text-white leading-tight">{domain.name}</h1>
      </div>

      <div className="max-w-5xl mx-auto px-8 py-12 space-y-16">
        {(['Beginner', 'Intermediate', 'Advanced', 'Expert'] as const).map(level => {
          const topics = groupedTopics[level];
          if (topics.length === 0) return null;
          return (
            <div key={level}>
              <div className="text-[12px] font-bold tracking-widest uppercase mb-6 flex items-center gap-3"
                style={{ color: getDifficultyColor(level) }}>
                {level} 
                <div className="flex-1 h-px bg-[#223429]"></div>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {topics.map(t => (
                  <Link
                    key={t.slug}
                    href={`/learn/${domain.slug}/${t.slug}`}
                    className="bg-[#121D16] border border-[#223429] p-6 rounded-xl block hover:border-[#60A5FA]/50 transition-colors group flex flex-col h-full"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-[18px] font-bold text-white group-hover:text-[#60A5FA] transition-colors">{t.title}</h3>
                        <div className="text-[14px] text-[#A3A3A3] mt-2 line-clamp-2 leading-relaxed">{t.why}</div>
                      </div>
                    </div>
                    <div className="mt-auto pt-4 flex items-center justify-between border-t border-[#223429]">
                      <div className="flex items-center gap-3">
                        <span className="px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider" 
                          style={{ backgroundColor: `${getDifficultyColor(t.level)}15`, color: getDifficultyColor(t.level) }}>
                          {t.level}
                        </span>
                        {t.studyTime && (
                          <span className="flex items-center gap-1.5 text-[11px] text-[#A3A3A3]">
                            <Clock size={12} /> {t.studyTime}
                          </span>
                        )}
                      </div>
                      <ArrowRight size={16} className="text-[#525252] group-hover:text-[#60A5FA] transition-colors" />
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
