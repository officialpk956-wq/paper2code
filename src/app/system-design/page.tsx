'use client';

import Link from 'next/link';
import { ArrowRight } from 'lucide-react';
import { SD_SYSTEMS } from '@/data/content/systemDesign';

export default function SystemDesignPage() {
  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white">
      {/* HEADER */}
      <div className="border-b border-[#262626] bg-[#0D0D0D] px-8 py-10 text-center">
        <h1 className="text-[32px] font-bold text-white mb-3">AI System Design</h1>
        <p className="text-[15px] text-[#A3A3A3] max-w-2xl mx-auto">
          Master the architecture of production AI systems. Learn how to design scalable, 
          robust infrastructure for vector search, LLM serving, RAG, and agentic workflows.
        </p>
      </div>

      {/* GRID */}
      <div className="max-w-6xl mx-auto px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {SD_SYSTEMS.map(sys => {
            // Count total hands-on projects across all modules
            const projectCount = sys.modules.reduce((acc, mod) => acc + mod.handsOnProjects.length, 0);
            
            return (
              <Link 
                key={sys.slug} 
                href={`/system-design/${sys.slug}`}
                className="group bg-[#111] border border-[#262626] rounded-2xl p-6 hover:border-[#F97316]/50 transition-all flex flex-col h-full"
              >
                <div className="text-[11px] font-bold text-[#F97316] uppercase tracking-wider mb-2">
                  System {sys.number}
                </div>
                <h2 className="text-[20px] font-bold text-white mb-3 group-hover:text-[#F97316] transition-colors leading-tight">
                  {sys.name}
                </h2>
                
                <div className="mt-auto space-y-3">
                  <div className="flex items-center text-[12px] text-[#A3A3A3]">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#4ADE80] mr-2"></span>
                    4 levels · Beginner → Research
                  </div>
                  <div className="flex items-center justify-between pt-4 border-t border-[#262626]">
                    <span className="text-[12px] font-medium text-[#525252]">
                      {projectCount} Hands-on Projects
                    </span>
                    <ArrowRight size={16} className="text-[#525252] group-hover:text-[#F97316] transition-colors" />
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
