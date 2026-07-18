'use client';

import React, { useState } from 'react';
import { ARCHITECTURES } from '@/data/content/architectures';
import { ArrowRight, GitCompare } from 'lucide-react';
import Link from 'next/link';

export default function CompareTrigger({ currentSlug }: { currentSlug: string }) {
  const [isOpen, setIsOpen] = useState(false);
  const [selected, setSelected] = useState('');

  if (!isOpen) {
    return (
      <button 
        onClick={() => setIsOpen(true)}
        className="mt-6 flex items-center gap-2 text-[12px] text-[#A3A3A3] hover:text-white transition-colors p-2 rounded-md hover:bg-[#111111] w-full"
      >
        <GitCompare size={14} /> Compare Architecture
      </button>
    );
  }

  return (
    <div className="mt-6 bg-[#111111] border border-[#262626] rounded-md p-3">
      <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-2 flex items-center gap-2">
        <GitCompare size={12} /> Compare With
      </div>
      <select 
        className="w-full bg-[#0A0A0A] border border-[#262626] text-white text-[12px] p-2 rounded-md mb-3 focus:outline-none focus:border-[#A3E635]"
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
      >
        <option value="" disabled>Select architecture...</option>
        {ARCHITECTURES.map(a => {
          if (a.slug === currentSlug) return null;
          return (
            <option key={a.slug} value={a.slug}>
              {a.name} ({a.year})
            </option>
          );
        })}
      </select>
      
      <div className="flex justify-end gap-2">
        <button 
          onClick={() => setIsOpen(false)}
          className="text-[11px] text-[#A3A3A3] hover:text-white px-2 py-1"
        >
          Cancel
        </button>
        {selected ? (
          <Link 
            href={`/architectures/compare?aName=${currentSlug}&bName=${selected}`}
            className="text-[11px] bg-[#A3E635] text-black px-3 py-1 rounded-md font-medium hover:bg-[#bef264] flex items-center gap-1"
          >
            Compare <ArrowRight size={12} />
          </Link>
        ) : (
          <button disabled className="text-[11px] bg-[#262626] text-[#525252] px-3 py-1 rounded-md font-medium flex items-center gap-1 cursor-not-allowed">
            Compare <ArrowRight size={12} />
          </button>
        )}
      </div>
    </div>
  );
}
