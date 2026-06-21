"use client";

import { useSearchParams } from "next/navigation";
import { Breadcrumb } from "@/components/breadcrumb";
import { EVOLUTION_NODES } from "@/data/evolution";
import { Check, X, ArrowRight, SkipForward } from "lucide-react";
import Link from "next/link";
import { Suspense, useState } from "react";

function ComparisonContent() {
  const searchParams = useSearchParams();
  const initialA = searchParams.get("a");
  const initialB = searchParams.get("b");

  const [slugA, setSlugA] = useState(initialA || "resnet");
  const [slugB, setSlugB] = useState(initialB || "densenet");

  const nodeA = EVOLUTION_NODES.find((n) => n.slug === slugA);
  const nodeB = EVOLUTION_NODES.find((n) => n.slug === slugB);

  return (
    <div>
      <div className="border-b border-[--color-border] bg-[--bg-surface]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Breadcrumb items={[{ label: "Compare", current: true }]} />
          <h1 className="text-3xl font-bold mb-2 mt-4">Comparison Engine</h1>
          <p className="text-[--color-text-secondary] max-w-3xl">
            Select two architectures or papers to compare their innovations, solved problems, and lineage side-by-side.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center gap-4 mt-8">
            <select 
              value={slugA} 
              onChange={(e) => setSlugA(e.target.value)}
              className="w-full sm:w-1/2 p-3 rounded-lg bg-[--bg-surface]/50 border border-[--color-border] text-[--color-text-primary] font-bold outline-none focus:border-[--accent-primary]"
            >
              {EVOLUTION_NODES.map(n => (
                <option key={n.slug} value={n.slug}>{n.title}</option>
              ))}
            </select>
            <div className="text-[--color-text-tertiary] font-bold">VS</div>
            <select 
              value={slugB} 
              onChange={(e) => setSlugB(e.target.value)}
              className="w-full sm:w-1/2 p-3 rounded-lg bg-[--bg-surface]/50 border border-[--color-border] text-[--color-text-primary] font-bold outline-none focus:border-[--accent-primary]"
            >
              {EVOLUTION_NODES.map(n => (
                <option key={n.slug} value={n.slug}>{n.title}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {!nodeA || !nodeB ? (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 text-center text-[--color-text-secondary]">
          Please select two valid nodes to compare.
        </div>
      ) : (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-16">
            
            {/* Column A */}
            <div className="space-y-12">
              <div className="sticky top-14 bg-[--bg-surface] py-4 z-10 border-b border-[--color-border]">
                <h2 className="text-2xl font-bold text-[--accent-primary]">{nodeA.title}</h2>
                <Link href={`/papers/${nodeA.slug}`} className="text-xs font-semibold text-[--color-text-tertiary] hover:text-[--accent-cyan] flex items-center gap-1 mt-1">
                  Read full breakdown <ArrowRight className="w-3 h-3" />
                </Link>
              </div>

              <section>
                <h3 className="text-lg font-bold text-[--color-text-primary] mb-4 flex items-center gap-2">
                  <Check className="w-5 h-5 text-emerald-400" /> Innovations Introduced
                </h3>
                <ul className="space-y-3">
                  {nodeA.innovationsIntroduced.length > 0 ? nodeA.innovationsIntroduced.map((inv, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-[--accent-cyan] mt-2 flex-shrink-0" />
                      <span className="text-[--color-text-secondary]">{inv}</span>
                    </li>
                  )) : <li className="text-[--color-text-tertiary] italic">None documented.</li>}
                </ul>
              </section>

              <section>
                <h3 className="text-lg font-bold text-[--color-text-primary] mb-4 flex items-center gap-2">
                  <X className="w-5 h-5 text-red-400" /> Problems Solved
                </h3>
                <ul className="space-y-3">
                  {nodeA.problemsSolved.length > 0 ? nodeA.problemsSolved.map((prob, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-2 flex-shrink-0" />
                      <span className="text-[--color-text-secondary]">{prob}</span>
                    </li>
                  )) : <li className="text-[--color-text-tertiary] italic">None documented.</li>}
                </ul>
              </section>

              <section>
                <h3 className="text-lg font-bold text-[--color-text-primary] mb-4 flex items-center gap-2">
                  <SkipForward className="w-5 h-5 text-purple-400" /> Replaced Predecessors
                </h3>
                <div className="space-y-4">
                  {nodeA.replaced.length > 0 ? nodeA.replaced.map((rep, idx) => (
                    <div key={idx} className="p-4 rounded-lg bg-[--bg-surface]/50 border border-[--color-border]">
                      <div className="font-bold text-[--color-text-primary] mb-1">Replaced: {rep.targetSlug}</div>
                      <div className="text-sm text-[--color-text-secondary]">{rep.reason}</div>
                    </div>
                  )) : <div className="text-[--color-text-tertiary] italic">Did not explicitly replace a predecessor in this index.</div>}
                </div>
              </section>

              <section>
                <h3 className="text-lg font-bold text-[--color-text-primary] mb-4">Lineage</h3>
                <div className="text-sm text-[--color-text-secondary]">
                  <span className="font-bold text-[--color-text-primary]">Parents:</span> {nodeA.parents.length > 0 ? nodeA.parents.join(", ") : "None"}
                </div>
                <div className="text-sm text-[--color-text-secondary] mt-2">
                  <span className="font-bold text-[--color-text-primary]">Children:</span> {nodeA.children.length > 0 ? nodeA.children.join(", ") : "None"}
                </div>
              </section>
            </div>

            {/* Column B */}
            <div className="space-y-12">
              <div className="sticky top-14 bg-[--bg-surface] py-4 z-10 border-b border-[--color-border]">
                <h2 className="text-2xl font-bold text-[--accent-primary]">{nodeB.title}</h2>
                <Link href={`/papers/${nodeB.slug}`} className="text-xs font-semibold text-[--color-text-tertiary] hover:text-[--accent-cyan] flex items-center gap-1 mt-1">
                  Read full breakdown <ArrowRight className="w-3 h-3" />
                </Link>
              </div>

              <section>
                <h3 className="text-lg font-bold text-[--color-text-primary] mb-4 flex items-center gap-2">
                  <Check className="w-5 h-5 text-emerald-400" /> Innovations Introduced
                </h3>
                <ul className="space-y-3">
                  {nodeB.innovationsIntroduced.length > 0 ? nodeB.innovationsIntroduced.map((inv, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-[--accent-cyan] mt-2 flex-shrink-0" />
                      <span className="text-[--color-text-secondary]">{inv}</span>
                    </li>
                  )) : <li className="text-[--color-text-tertiary] italic">None documented.</li>}
                </ul>
              </section>

              <section>
                <h3 className="text-lg font-bold text-[--color-text-primary] mb-4 flex items-center gap-2">
                  <X className="w-5 h-5 text-red-400" /> Problems Solved
                </h3>
                <ul className="space-y-3">
                  {nodeB.problemsSolved.length > 0 ? nodeB.problemsSolved.map((prob, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-orange-400 mt-2 flex-shrink-0" />
                      <span className="text-[--color-text-secondary]">{prob}</span>
                    </li>
                  )) : <li className="text-[--color-text-tertiary] italic">None documented.</li>}
                </ul>
              </section>

              <section>
                <h3 className="text-lg font-bold text-[--color-text-primary] mb-4 flex items-center gap-2">
                  <SkipForward className="w-5 h-5 text-purple-400" /> Replaced Predecessors
                </h3>
                <div className="space-y-4">
                  {nodeB.replaced.length > 0 ? nodeB.replaced.map((rep, idx) => (
                    <div key={idx} className="p-4 rounded-lg bg-[--bg-surface]/50 border border-[--color-border]">
                      <div className="font-bold text-[--color-text-primary] mb-1">Replaced: {rep.targetSlug}</div>
                      <div className="text-sm text-[--color-text-secondary]">{rep.reason}</div>
                    </div>
                  )) : <div className="text-[--color-text-tertiary] italic">Did not explicitly replace a predecessor in this index.</div>}
                </div>
              </section>

              <section>
                <h3 className="text-lg font-bold text-[--color-text-primary] mb-4">Lineage</h3>
                <div className="text-sm text-[--color-text-secondary]">
                  <span className="font-bold text-[--color-text-primary]">Parents:</span> {nodeB.parents.length > 0 ? nodeB.parents.join(", ") : "None"}
                </div>
                <div className="text-sm text-[--color-text-secondary] mt-2">
                  <span className="font-bold text-[--color-text-primary]">Children:</span> {nodeB.children.length > 0 ? nodeB.children.join(", ") : "None"}
                </div>
              </section>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="p-20 text-center">Loading comparison...</div>}>
      <ComparisonContent />
    </Suspense>
  );
}
