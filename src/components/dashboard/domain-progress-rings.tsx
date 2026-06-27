'use client';

import { useEffect, useState } from 'react';
import { TOPIC_REGISTRY } from '@/data/topics';
import { isCompleted } from '@/lib/progress';

export function DomainProgressRings() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const domains = Object.entries(TOPIC_REGISTRY).map(([domain, topics]) => {
    // Count completed (unique)
    let completed = 0;
    
    // Convert to Map to handle aliased topics but we just need the values since uniqueness handles aliases
    // The keys don't matter as much for isCompleted, wait, the slug matters.
    // If we just loop through the original keys:
    for (const slug of Object.keys(topics)) {
      if (isCompleted('topic', slug)) {
        completed++;
      }
    }

    // Wait, if an alias is completed, the canonical is too? The progress is stored under the canonical slug if we use it, or under the aliased slug.
    // Usually progress is tied to the slug visited. Let's just check the unique slugs.
    const uniqueSlugs = new Set(Object.values(topics).map(t => t.meta.slug));
    const total = uniqueSlugs.size;
    completed = 0;
    for (const slug of uniqueSlugs) {
      if (slug && isCompleted('topic', slug)) {
        completed++;
      }
    }

    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    return {
      id: domain,
      title: domain === 'nlp' || domain === 'llms' 
        ? domain.toUpperCase() 
        : domain.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '),
      total,
      completed,
      percentage
    };
  });

  if (!mounted) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 px-6 mt-6 mb-2">
      {domains.map(domain => (
        <div key={domain.id} className="bg-[--bg-panel] border border-[--color-border] rounded-xl p-4 flex flex-col items-center justify-center">
          <div className="relative w-16 h-16 mb-2">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
              <path
                className="text-[--bg-surface]"
                strokeWidth="3"
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
              <path
                className="text-[--accent-primary]"
                strokeWidth="3"
                strokeDasharray={`${domain.percentage}, 100`}
                stroke="currentColor"
                fill="none"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-[--color-text-primary]">
              {domain.percentage}%
            </div>
          </div>
          <div className="text-xs font-semibold text-[--color-text-primary] text-center">
            {domain.title}
          </div>
          <div className="text-[10px] text-[--color-text-secondary]">
            {domain.completed} / {domain.total} Topics
          </div>
        </div>
      ))}
    </div>
  );
}
