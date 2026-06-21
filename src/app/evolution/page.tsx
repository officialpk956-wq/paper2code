"use client";

import { useState } from "react";
import Link from "next/link";
import { Breadcrumb } from "@/components/breadcrumb";
import { 
  EVOLUTION_NODES, 
  RESEARCH_JOURNEYS, 
  FAMILY_TREES, 
  EvolutionNode 
} from "@/data/evolution";
import { ArrowRight, GitMerge, Lightbulb, SkipForward, Map } from "lucide-react";

function NodeCard({ node }: { node: EvolutionNode }) {
  return (
    <div className="p-4 rounded-lg border border-[--color-border] bg-[--bg-surface] hover:border-[--accent-primary] transition-colors relative group">
      <Link href={`/papers/${node.slug}`} className="absolute inset-0" />
      <h3 className="font-bold text-[--color-text-primary] mb-1 group-hover:text-[--accent-light]">
        {node.title}
      </h3>
      <div className="flex flex-col gap-2 mt-3">
        {node.innovationsIntroduced.length > 0 && (
          <div className="text-xs">
            <span className="text-[--accent-cyan] font-semibold flex items-center gap-1"><Lightbulb className="w-3 h-3"/> Introduced:</span>
            <ul className="list-disc pl-4 mt-1 text-[--color-text-secondary]">
              {node.innovationsIntroduced.map((i, idx) => (
                <li key={idx}>{i}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function JourneyView() {
  return (
    <div className="space-y-8">
      {RESEARCH_JOURNEYS.map(journey => (
        <div key={journey.id} className="p-6 rounded-xl border border-[--color-border] bg-[--bg-surface]/50">
          <h2 className="text-xl font-bold text-[--color-text-primary] mb-2">{journey.title}</h2>
          <p className="text-sm text-[--color-text-secondary] mb-6">{journey.description}</p>
          
          <div className="flex flex-wrap items-center gap-3">
            {journey.nodes.map((nodeSlug, idx) => {
              const node = EVOLUTION_NODES.find(n => n.slug === nodeSlug);
              if (!node) return null;
              
              return (
                <div key={nodeSlug} className="flex items-center gap-3">
                  <Link href={`/papers/${node.slug}`} className="px-4 py-2 rounded-lg bg-[--bg-surface] border border-[--color-border] hover:border-[--accent-primary] hover:text-[--accent-light] font-semibold text-sm transition-colors">
                    {node.title}
                  </Link>
                  {idx < journey.nodes.length - 1 && (
                    <ArrowRight className="w-4 h-4 text-[--color-text-tertiary]" />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

function FamilyTreeView() {
  // A simple recursive renderer for the trees
  const renderTree = (nodeSlug: string, visited: Set<string> = new Set()) => {
    if (visited.has(nodeSlug)) return null;
    visited.add(nodeSlug);
    
    const node = EVOLUTION_NODES.find(n => n.slug === nodeSlug);
    if (!node) return null;

    return (
      <div key={nodeSlug} className="flex flex-col items-center">
        <div className="w-48 relative z-10">
          <NodeCard node={node} />
        </div>
        {node.children.length > 0 && (
          <div className="flex flex-col items-center mt-2">
            <div className="w-px h-6 bg-[--color-border]"></div>
            <div className="flex items-start justify-center gap-4 border-t border-[--color-border] pt-4">
              {node.children.map(childSlug => renderTree(childSlug, new Set(visited)))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-12">
      {FAMILY_TREES.map(tree => (
        <div key={tree.id} className="p-6 rounded-xl border border-[--color-border] bg-[--bg-surface]/50 overflow-x-auto">
          <h2 className="text-xl font-bold text-[--color-text-primary] mb-2">{tree.title}</h2>
          <p className="text-sm text-[--color-text-secondary] mb-8">{tree.description}</p>
          <div className="flex justify-center min-w-max">
            {tree.roots.map(root => renderTree(root))}
          </div>
        </div>
      ))}
    </div>
  );
}

function ReplacementAnalysisView() {
  const replacements = EVOLUTION_NODES.flatMap(node => 
    node.replaced.map(r => ({
      source: node,
      targetSlug: r.targetSlug,
      reason: r.reason
    }))
  );

  return (
    <div className="grid md:grid-cols-2 gap-6">
      {replacements.map((rep, idx) => {
        const targetNode = EVOLUTION_NODES.find(n => n.slug === rep.targetSlug) || { title: rep.targetSlug };
        return (
          <div key={idx} className="p-6 rounded-xl border border-[--color-border] bg-[--bg-surface]/50 relative group">
            <div className="flex items-center gap-3 mb-4">
              <span className="font-bold text-[--accent-primary]">{rep.source.title}</span>
              <SkipForward className="w-4 h-4 text-[--color-text-tertiary]" />
              <span className="font-bold text-[--color-text-secondary]">{targetNode.title}</span>
            </div>
            <h3 className="text-sm font-bold text-[--color-text-primary] mb-2">Why it was replaced:</h3>
            <p className="text-sm text-[--color-text-secondary] leading-relaxed">
              {rep.reason}
            </p>
            <div className="mt-4 pt-4 border-t border-[--color-border]">
              <Link href={`/compare?a=${rep.source.slug}&b=${rep.targetSlug}`} className="text-xs font-semibold text-[--accent-cyan] hover:underline flex items-center gap-1">
                Compare these architectures <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function EvolutionEnginePage() {
  const [activeTab, setActiveTab] = useState<"journeys" | "families" | "replacements">("journeys");

  return (
    <div>
      <div className="border-b border-[--color-border] bg-[--bg-surface]">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Breadcrumb items={[{ label: "Research Evolution", current: true }]} />
          <h1 className="text-3xl font-bold mb-2 mt-4">Research Evolution Engine</h1>
          <p className="text-[--color-text-secondary] max-w-3xl">
            Understand how AI evolved. Trace the lineage of ideas, explore family trees, and learn why certain paradigms replaced others.
          </p>
        </div>
      </div>

      <div className="sticky top-14 z-20 border-b border-[--color-border] bg-[rgba(13,13,20,0.92)] backdrop-blur-md">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center gap-6">
          <button
            onClick={() => setActiveTab("journeys")}
            className={`py-4 text-sm font-bold border-b-2 transition-colors ${
              activeTab === "journeys" ? "border-[--accent-primary] text-[--accent-light]" : "border-transparent text-[--color-text-tertiary] hover:text-[--color-text-secondary]"
            }`}
          >
            <span className="flex items-center gap-2"><Map className="w-4 h-4"/> Research Journeys</span>
          </button>
          <button
            onClick={() => setActiveTab("families")}
            className={`py-4 text-sm font-bold border-b-2 transition-colors ${
              activeTab === "families" ? "border-[--accent-primary] text-[--accent-light]" : "border-transparent text-[--color-text-tertiary] hover:text-[--color-text-secondary]"
            }`}
          >
            <span className="flex items-center gap-2"><GitMerge className="w-4 h-4"/> Family Trees</span>
          </button>
          <button
            onClick={() => setActiveTab("replacements")}
            className={`py-4 text-sm font-bold border-b-2 transition-colors ${
              activeTab === "replacements" ? "border-[--accent-primary] text-[--accent-light]" : "border-transparent text-[--color-text-tertiary] hover:text-[--color-text-secondary]"
            }`}
          >
            <span className="flex items-center gap-2"><SkipForward className="w-4 h-4"/> Replacement Analysis</span>
          </button>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        {activeTab === "journeys" && <JourneyView />}
        {activeTab === "families" && <FamilyTreeView />}
        {activeTab === "replacements" && <ReplacementAnalysisView />}
      </div>
    </div>
  );
}
