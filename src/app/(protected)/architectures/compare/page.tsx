'use client';

import React, { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import { apiGet } from '@/lib/api';
import { ARCHITECTURES } from '@/data/content/architectures';

type CompareData = {
  status?: string;
  message?: string;
  paper_a?: { id: number; title: string };
  paper_b?: { id: number; title: string };
  diff?: {
    added_nodes: string[];
    removed_nodes: string[];
    changed_params: any[];
    deltas: Record<string, number | null>;
    summary?: string;
  };
};

function CompareContent() {
  const searchParams = useSearchParams();
  const a = searchParams.get('a');
  const b = searchParams.get('b');
  const aName = searchParams.get('aName');
  const bName = searchParams.get('bName');

  const [data, setData] = useState<CompareData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const archA = ARCHITECTURES.find(x => x.slug === aName);
  const archB = ARCHITECTURES.find(x => x.slug === bName);

  useEffect(() => {
    if (!aName || !bName) {
      setLoading(false);
      setError("Missing architecture slugs in URL");
      return;
    }
    
    let url = `/api/architectures/compare?a_slug=${aName}&b_slug=${bName}`;
    if (a && b) {
      url = `/api/architectures/compare?paper_a=${a}&paper_b=${b}`;
    }

    setLoading(true);
    apiGet<CompareData>(url)
      .then(setData)
      .catch(e => setError(e.message || "Failed to load comparison data"))
      .finally(() => setLoading(false));
  }, [a, b, aName, bName]);

  return (
    <div className="flex flex-col h-full bg-[#050505] text-white overflow-y-auto">
      <div className="p-6 border-b border-[#1A1A1A]">
        <Link href={`/architectures/${aName || ''}`} className="text-[12px] text-[#A3A3A3] hover:text-white flex items-center gap-1.5 mb-4 w-fit">
          <ArrowLeft size={14} /> Back
        </Link>
        <h1 className="text-2xl font-bold">Compare Architectures</h1>
        <p className="text-[#A3A3A3] text-sm mt-1">Analyzing structural differences and metric deltas</p>
      </div>

      <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl w-full">
        {/* Arch A Info */}
        <div className="bg-[#111111] border border-[#262626] rounded-xl p-6">
          <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-2">Architecture A</div>
          <h2 className="text-xl font-bold text-white">{archA?.name || aName}</h2>
          <div className="text-sm text-[#525252] mt-1">{archA?.year} · {archA?.difficulty}</div>
          {data?.paper_a && <div className="mt-4 text-[11px] text-[#525252]">Source: {data.paper_a.title}</div>}
        </div>
        
        {/* Arch B Info */}
        <div className="bg-[#111111] border border-[#262626] rounded-xl p-6">
          <div className="text-[11px] font-bold text-[#A3A3A3] uppercase tracking-wider mb-2">Architecture B</div>
          <h2 className="text-xl font-bold text-white">{archB?.name || bName}</h2>
          <div className="text-sm text-[#525252] mt-1">{archB?.year} · {archB?.difficulty}</div>
          {data?.paper_b && <div className="mt-4 text-[11px] text-[#525252]">Source: {data.paper_b.title}</div>}
        </div>
      </div>

      <div className="px-6 pb-12 max-w-6xl w-full space-y-6">
        <h2 className="text-lg font-semibold border-b border-[#262626] pb-2">Differences</h2>
        
        {loading && (
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-[#262626] rounded w-1/4" />
            <div className="h-8 bg-[#262626] rounded w-1/2" />
          </div>
        )}

        {error && (
          <div className="text-red-400 bg-red-400/10 border border-red-400/20 p-4 rounded-xl text-sm">
            {error}
          </div>
        )}

        {!loading && !error && data?.status === 'incomplete' && (
          <div className="text-[#A3A3A3] bg-[#1A1A1A] border border-[#262626] p-4 rounded-xl text-sm">
            {data.message || "Architecture data unavailable for comparison"}
          </div>
        )}

        {!loading && !error && data?.diff && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="space-y-6">
              {/* Added Nodes */}
              <div className="bg-[#111111] border border-[#22c55e]/30 rounded-xl p-5">
                <h3 className="text-sm font-semibold text-[#22c55e] mb-3">Added in B</h3>
                {data.diff.added_nodes.length === 0 ? (
                  <p className="text-xs text-[#525252]">None</p>
                ) : (
                  <ul className="space-y-1.5">
                    {data.diff.added_nodes.map(n => (
                      <li key={n} className="text-[13px] text-white flex items-center gap-2">
                        <span className="text-[#22c55e]">+</span> {n}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {/* Removed Nodes */}
              <div className="bg-[#111111] border border-[#ef4444]/30 rounded-xl p-5">
                <h3 className="text-sm font-semibold text-[#ef4444] mb-3">Removed from A</h3>
                {data.diff.removed_nodes.length === 0 ? (
                  <p className="text-xs text-[#525252]">None</p>
                ) : (
                  <ul className="space-y-1.5">
                    {data.diff.removed_nodes.map(n => (
                      <li key={n} className="text-[13px] text-white flex items-center gap-2">
                        <span className="text-[#ef4444]">-</span> {n}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              
              {/* Changed Nodes */}
              <div className="bg-[#111111] border border-[#eab308]/30 rounded-xl p-5">
                <h3 className="text-sm font-semibold text-[#eab308] mb-3">Changed Nodes</h3>
                {(!data.diff.changed_params || data.diff.changed_params.length === 0) ? (
                  <p className="text-xs text-[#525252]">None</p>
                ) : (
                  <ul className="space-y-1.5">
                    {data.diff.changed_params.map((n, idx) => (
                      <li key={idx} className="text-[13px] text-white flex items-start gap-2">
                        <span className="text-[#eab308] mt-0.5">~</span> 
                        <div>
                          <div>{n.label}</div>
                          {n.type_changed && <div className="text-[11px] text-[#A3A3A3] mt-1">Type: {n.from.type} &rarr; {n.to.type}</div>}
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* Metrics */}
            <div className="bg-[#111111] border border-[#262626] rounded-xl p-5 h-fit">
              <h3 className="text-sm font-semibold text-white mb-4 border-b border-[#262626] pb-2">Metric Deltas (A &rarr; B)</h3>
              <div className="space-y-3">
                {Object.entries(data.diff.deltas).map(([k, v]) => {
                  if (v === null || v === undefined) return null;
                  const isPos = v > 0;
                  const isNeg = v < 0;
                  return (
                    <div key={k} className="flex justify-between items-center py-1">
                      <span className="text-[13px] text-[#A3A3A3] capitalize">{k}</span>
                      <span className={`text-[13px] font-mono ${isPos ? 'text-green-400' : isNeg ? 'text-blue-400' : 'text-[#525252]'}`}>
                        {v > 0 ? '+' : ''}{v}
                      </span>
                    </div>
                  );
                })}
              </div>
              {data.diff.summary && (
                <div className="mt-6 pt-4 border-t border-[#262626] text-[12px] leading-relaxed text-[#A3A3A3]">
                  {data.diff.summary}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ArchitectureComparePage() { return <Suspense fallback={<div>Loading compare...</div>}><CompareContent /></Suspense>; }