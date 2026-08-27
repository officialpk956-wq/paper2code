'use client';

import React, { useEffect, useState, Suspense } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
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

type ArchitectureComboboxProps = {
  id: string;
  label: string;
  selectedSlug: string;
  onSelect: (slug: string) => void;
  excludeSlug?: string;
};

function ArchitectureCombobox({ id, label, selectedSlug, onSelect, excludeSlug }: ArchitectureComboboxProps) {
  const selected = ARCHITECTURES.find((architecture) => architecture.slug === selectedSlug);
  const [query, setQuery] = useState(selected ? `${selected.name} (${selected.year})` : '');
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const architecture = ARCHITECTURES.find((entry) => entry.slug === selectedSlug);
    setQuery(architecture ? `${architecture.name} (${architecture.year})` : '');
  }, [selectedSlug]);

  const normalizedQuery = query.trim().toLowerCase();
  const options = ARCHITECTURES.filter((architecture) => {
    if (architecture.slug === excludeSlug) return false;
    if (!normalizedQuery || selectedSlug) return true;
    return `${architecture.name} ${architecture.slug} ${architecture.year}`
      .toLowerCase()
      .includes(normalizedQuery);
  }).slice(0, 12);

  return (
    <div className="relative">
      <label htmlFor={id} className="mb-2 block text-[11px] font-bold uppercase tracking-wider text-[#A3A3A3]">
        {label}
      </label>
      <input
        id={id}
        role="combobox"
        aria-autocomplete="list"
        aria-controls={`${id}-options`}
        aria-expanded={open}
        autoComplete="off"
        value={query}
        placeholder="Search by name, slug, or year"
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onChange={(event) => {
          setQuery(event.target.value);
          onSelect('');
          setOpen(true);
        }}
        className="w-full rounded-lg border border-[#333333] bg-[#0A0A0A] px-3 py-2.5 text-sm text-white outline-none placeholder:text-[#525252] focus:border-[#A78BFA]"
      />
      {open && (
        <div
          id={`${id}-options`}
          role="listbox"
          className="absolute z-20 mt-2 max-h-72 w-full overflow-y-auto rounded-xl border border-[#333333] bg-[#111111] p-1.5 shadow-2xl"
        >
          {options.length > 0 ? options.map((architecture) => (
            <button
              key={architecture.slug}
              type="button"
              role="option"
              aria-selected={architecture.slug === selectedSlug}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => {
                onSelect(architecture.slug);
                setQuery(`${architecture.name} (${architecture.year})`);
                setOpen(false);
              }}
              className="flex w-full items-center justify-between rounded-lg px-3 py-2 text-left hover:bg-white/[0.06]"
            >
              <span>
                <span className="block text-sm font-medium text-white">{architecture.name}</span>
                <span className="block text-[11px] text-[#525252]">{architecture.slug}</span>
              </span>
              <span className="text-xs text-[#A3A3A3]">{architecture.year}</span>
            </button>
          )) : (
            <div className="px-3 py-4 text-center text-xs text-[#737373]">No architectures found.</div>
          )}
        </div>
      )}
    </div>
  );
}

function CompareContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const a = searchParams.get('a');
  const b = searchParams.get('b');
  const aName = searchParams.get('aName');
  const bName = searchParams.get('bName');

  const [data, setData] = useState<CompareData | null>(null);
  const [loading, setLoading] = useState(Boolean(aName && bName));
  const [error, setError] = useState('');
  const [selectedA, setSelectedA] = useState(aName ?? '');
  const [selectedB, setSelectedB] = useState(bName ?? '');

  const archA = ARCHITECTURES.find(x => x.slug === aName);
  const archB = ARCHITECTURES.find(x => x.slug === bName);

  useEffect(() => {
    if (!aName || !bName) {
      setLoading(false);
      setData(null);
      setError('');
      return;
    }
    
    let url = `/api/architectures/compare?a_slug=${aName}&b_slug=${bName}`;
    if (a && b) {
      url = `/api/architectures/compare?paper_a=${a}&paper_b=${b}`;
    }

    setLoading(true);
    setError('');
    setData(null);
    apiGet<CompareData>(url)
      .then(setData)
      .catch(e => setError(e.message || "Failed to load comparison data"))
      .finally(() => setLoading(false));
  }, [a, b, aName, bName]);

  const compareSelected = () => {
    if (!selectedA || !selectedB || selectedA === selectedB) return;
    router.push(
      `/architectures/compare?aName=${encodeURIComponent(selectedA)}&bName=${encodeURIComponent(selectedB)}`,
    );
  };

  const hasComparison = Boolean(aName && bName);
  const canCompare = Boolean(selectedA && selectedB && selectedA !== selectedB);

  return (
    <div className="flex flex-col h-full bg-[#050505] text-white overflow-y-auto">
      <div className="p-6 border-b border-[#1A1A1A]">
        <Link href={aName ? `/architectures/${aName}` : '/architectures'} className="text-[12px] text-[#A3A3A3] hover:text-white flex items-center gap-1.5 mb-4 w-fit">
          <ArrowLeft size={14} /> Back
        </Link>
        <h1 className="text-2xl font-bold">Compare Architectures</h1>
        <p className="text-[#A3A3A3] text-sm mt-1">Analyzing structural differences and metric deltas</p>
      </div>

      <div className="w-full max-w-6xl p-6">
        <div className="rounded-xl border border-[#262626] bg-[#111111] p-6">
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            <ArchitectureCombobox
              id="architecture-a"
              label="Architecture A"
              selectedSlug={selectedA}
              onSelect={setSelectedA}
              excludeSlug={selectedB}
            />
            <ArchitectureCombobox
              id="architecture-b"
              label="Architecture B"
              selectedSlug={selectedB}
              onSelect={setSelectedB}
              excludeSlug={selectedA}
            />
          </div>
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-[#737373]">
              {selectedA && selectedB && selectedA === selectedB
                ? 'Choose two different architectures.'
                : 'Pick two architectures to compare'}
            </p>
            <button
              type="button"
              disabled={!canCompare}
              onClick={compareSelected}
              className="rounded-lg bg-[#A78BFA] px-4 py-2 text-sm font-semibold text-black transition hover:bg-[#C4B5FD] disabled:cursor-not-allowed disabled:opacity-40"
            >
              Compare selected
            </button>
          </div>
        </div>
      </div>

      {hasComparison && <div className="p-6 pt-0 grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl w-full">
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
      </div>}

      <div className="px-6 pb-12 max-w-6xl w-full space-y-6">
        {hasComparison && <h2 className="text-lg font-semibold border-b border-[#262626] pb-2">Differences</h2>}
        
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

        {!hasComparison && (
          <div className="rounded-xl border border-dashed border-[#333333] bg-[#0A0A0A] p-8 text-center text-sm text-[#737373]">
            Pick two architectures to compare
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
