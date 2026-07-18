'use client';

import { useEffect, useState } from 'react';
import { apiGet } from '@/lib/api';

type Node = {
  id: string;
  type: string;
  dimensionality: string;
};

type Constraint = {
  from: string;
  to: string;
  relation: string;
  reason: string;
};

type KAGData = {
  nodes: Node[];
  constraints: Constraint[];
};

export default function KAGRelations({ slug }: { slug: string }) {
  const [data, setData] = useState<KAGData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    apiGet<KAGData>(`/api/architectures/${slug}/knowledge-relations`)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [slug]);

  if (loading) {
    return (
      <div className="bg-[#111111] border border-[#1A1A1A] rounded-xl p-4 mb-4">
        <h3 className="text-[10px] uppercase font-bold text-[#A78BFA] tracking-widest mb-3">
          Architecture Constraints
        </h3>
        <div className="space-y-3">
          <div className="h-4 bg-[#1A1A1A] rounded w-3/4 animate-pulse"></div>
          <div className="h-4 bg-[#1A1A1A] rounded w-1/2 animate-pulse"></div>
          <div className="h-4 bg-[#1A1A1A] rounded w-5/6 animate-pulse"></div>
        </div>
      </div>
    );
  }

  if (!data || data.nodes.length === 0 || data.constraints.length === 0) {
    return null;
  }

  return (
    <div className="bg-[#111111] border border-[#1A1A1A] rounded-xl p-4 mb-4">
      <h3 className="text-[10px] uppercase font-bold text-[#A78BFA] tracking-widest mb-3">
        Architecture Constraints
      </h3>
      <div className="space-y-3">
        {data.constraints.map((c, i) => {
          const isCompatible = c.relation === 'COMPATIBLE';
          return (
            <div key={i} className="flex flex-col">
              <div className="flex items-center gap-2">
                <span 
                  className="w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: isCompatible ? '#22c55e' : '#ef4444' }}
                ></span>
                <span className="text-[12px] text-[#A3A3A3]">
                  {c.from} &rarr; {c.to}
                </span>
              </div>
              <span className="text-[11px] text-[#525252] ml-3.5 mt-0.5">
                {c.reason}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
