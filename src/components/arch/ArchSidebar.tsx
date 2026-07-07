'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

function slugify(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^\w\-]+/g, '')
    .replace(/\-\-+/g, '-');
}

interface ArchSidebarProps {
  sections: string[];
}

export default function ArchSidebar({ sections }: ArchSidebarProps) {
  const [activeId, setActiveId] = useState<string>('');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntries = entries.filter((entry) => entry.isIntersecting);
        if (visibleEntries.length > 0) {
          setActiveId(visibleEntries[0].target.id);
        }
      },
      {
        rootMargin: '-80px 0px -60% 0px',
        threshold: 0.1,
      }
    );

    const headingElements = document.querySelectorAll('h2[id]');
    headingElements.forEach((el) => observer.observe(el));

    return () => {
      headingElements.forEach((el) => observer.unobserve(el));
    };
  }, [sections]);

  return (
    <div className="w-[260px] border-r border-[#262626] bg-[#0A0A0A] flex-shrink-0 flex flex-col h-full">
      <div className="p-4 border-b border-[#262626]">
        <Link href="/architectures" className="text-[12px] text-[#A3A3A3] hover:text-white flex items-center gap-1.5 transition-colors">
          <ArrowLeft size={14} /> Back to Library
        </Link>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-1">
        {sections.map((section, idx) => {
          const id = slugify(section);
          const isActive = activeId === id;
          return (
            <a
              key={section}
              href={`#${id}`}
              className={`block text-[12px] px-3 py-2 rounded-md transition-colors truncate ${
                isActive
                  ? 'text-[#A3E635] bg-[#A3E635]/10 font-medium'
                  : 'text-[#A3A3A3] hover:text-white hover:bg-[#111111]'
              }`}
            >
              {idx + 1}. {section}
            </a>
          );
        })}
      </div>
    </div>
  );
}
