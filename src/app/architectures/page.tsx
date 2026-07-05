'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ARCHITECTURES } from '@/data/content/architectures';

export default function ArchitecturesPage() {
  const [active, setActive] = useState('All');
  const [search, setSearch] = useState('');
  
  const categories = ['All', ...Array.from(new Set(ARCHITECTURES.map(a => a.category)))];
  
  const filtered = ARCHITECTURES.filter(a => {
    if (active !== 'All' && a.category !== active) return false;
    if (search && !a.name.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const getDifficultyColor = (diff: string) => {
    switch(diff) {
      case 'Beginner': return '#4ADE80';
      case 'Intermediate': return '#FACC15';
      case 'Advanced': return '#F87171';
      case 'Expert': return '#A78BFA';
      default: return '#525252';
    }
  };

  return (
    <div className="min-h-screen bg-transparent text-white">
      {/* HEADER */}
      <div className="border-b border-[#262626] bg-[#111111] px-8 py-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h1 className="text-[26px] font-bold text-white">Architecture Library</h1>
            <p className="mt-1 text-[13px] text-[#A3A3A3]">
              {filtered.length} of 207 architectures
            </p>
          </div>
          <input
            type="text"
            placeholder="Search architectures..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-[240px] bg-[#111111] border border-[#262626] rounded-lg px-3 py-2 text-[13px] text-white placeholder:text-[#525252] focus:outline-none focus:border-[#A3E635]/40"
          />
        </div>
        <div className="flex flex-wrap gap-2">
          {categories.map(c => (
            <button key={c} type="button" onClick={() => setActive(c)}
              className={'rounded-full px-3 py-1 text-[11px] font-medium border transition-colors ' +
                (c === active
                  ? 'bg-[#A3E635] border-[#A3E635] text-black'
                  : 'border-[#262626] text-[#A3A3A3] hover:text-white hover:border-[#2E4436]')}>
              {c}
            </button>
          ))}
        </div>
      </div>

      {/* GRID */}
      <div className="mx-auto max-w-7xl px-8 py-8 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {filtered.map(a => (
          <Link key={a.slug} href={`/architectures/${a.slug}`}
            className="flex flex-col rounded-xl border border-[#262626] bg-[#111111] p-4 hover:border-[#A3E635]/30 transition-colors">
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] text-[#525252]">{a.year}</span>
              <span className="text-[10px] px-2 py-0.5 rounded-md font-medium" style={{ backgroundColor: `${getDifficultyColor(a.difficulty)}20`, color: getDifficultyColor(a.difficulty) }}>
                {a.difficulty}
              </span>
            </div>
            <div className="text-[14px] font-bold text-white">{a.name}</div>
            <div className="mt-1 text-[11px] leading-relaxed text-[#A3A3A3] line-clamp-2 flex-1">{a.keyInnovation}</div>
            <div className="mt-3">
              <span className="rounded-md bg-[#111111] border border-[#262626] px-2 py-0.5 text-[10px] text-[#A3A3A3]">
                {a.category}
              </span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
