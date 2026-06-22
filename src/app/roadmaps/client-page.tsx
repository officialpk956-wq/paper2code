"use client";

import { useState } from "react";
import { Breadcrumb } from "@/components/breadcrumb";
import { ROADMAP_LIST, ROADMAPS, RoadmapId, RoadmapNode } from "@/data/roadmaps";
import { RoadmapGraph } from "@/components/roadmap-graph";
import { NodeDetailPanel } from "@/components/node-detail-panel";
import { RoadmapProgress } from "@/components/roadmap-progress";
import { LearningGraph } from "@/components/learning-graph";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { resolveTopicPath } from "@/data/topics";
import { ArrowRight, Zap } from "lucide-react";

export function RoadmapsClientPage({ validSlugs }: { validSlugs: Record<string, string[]> }) {
  const router = useRouter();
  const [selectedRoadmapId, setSelectedRoadmapId] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<RoadmapNode | null>(null);
  const [activeTab, setActiveTab] = useState<"path" | "graph">("path");

  const selectedRoadmap = selectedRoadmapId
    ? ROADMAPS[selectedRoadmapId as RoadmapId]
    : null;

  const graphNodes = selectedRoadmap
    ? selectedRoadmap.phases
        .flatMap((phase) =>
          phase.nodes.map((node) => ({
            id: node.id,
            label: node.title,
            type: "topic" as const,
            color: node.state === "completed" ? "#10B981" : "#7C3AED",
          }))
        )
    : [];

  const graphConnections = selectedRoadmap
    ? selectedRoadmap.phases
        .flatMap((phase) =>
          phase.nodes.flatMap((node) =>
            node.prerequisites.map((prereq) => ({
              from: prereq,
              to: node.id,
              type: "requires" as const,
              fromType: "topic",
              toType: "topic",
            }))
          )
        )
    : [];

  return (
    <div>
      {/* Header */}
      <div className="border-b border-[--color-border] bg-[--bg-surface] sticky top-14 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <Breadcrumb items={[{ label: "Roadmaps", current: true }]} />
          <h1 className="text-3xl font-bold mb-2">Learning Roadmaps</h1>
          <p className="text-[--color-text-secondary]">
            Choose a specialized path and master your AI engineering journey
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Roadmap selector */}
        {!selectedRoadmap ? (
          <div>
            <h2 className="text-2xl font-bold mb-6">Choose Your Path</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {ROADMAP_LIST.map((roadmap) => (
                <button
                  key={roadmap.id}
                  onClick={() => setSelectedRoadmapId(roadmap.id)}
                  className="group p-6 rounded-lg bg-[--bg-surface] border border-[--color-border] hover:border-[--accent-primary] transition-all text-left hover:shadow-lg"
                >
                  <div className="flex items-start justify-between mb-4">
                    <span className="text-4xl">{roadmap.icon}</span>
                    <span className="text-sm font-semibold text-[--accent-primary] bg-[rgb(124,58,237,0.1)] px-2 py-1 rounded-full">
                      {roadmap.progress}%
                    </span>
                  </div>

                  <h3 className="text-lg font-bold mb-2 group-hover:text-[--accent-primary] transition-colors">
                    {roadmap.title}
                  </h3>
                  <p className="text-sm text-[--color-text-tertiary] mb-4">
                    {roadmap.description}
                  </p>

                  <div className="h-2 bg-[--bg-body] rounded-full overflow-hidden mb-4">
                    <div
                      className={`h-full bg-gradient-to-r ${roadmap.color}`}
                      style={{ width: `${roadmap.progress}%` }}
                    />
                  </div>

                  <div className="flex items-center gap-2 text-[--accent-primary] font-medium group-hover:gap-3 transition-all">
                    View Roadmap
                    <ArrowRight className="w-4 h-4" />
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div>
            {/* Back button and header */}
            <div className="flex items-center justify-between mb-8">
              <div>
                <button
                  onClick={() => {
                    setSelectedRoadmapId(null);
                    setSelectedNode(null);
                  }}
                  className="text-[--color-text-tertiary] hover:text-[--color-text-secondary] font-medium mb-2 flex items-center gap-1"
                >
                  ← Back to Roadmaps
                </button>
                <h2 className="text-3xl font-bold">
                  {selectedRoadmap.title} Roadmap
                </h2>
              </div>
              <span className="text-4xl">{selectedRoadmap.icon}</span>
            </div>

            <div className="grid lg:grid-cols-4 gap-8">
              {/* Main content */}
              <div className="lg:col-span-3">
                {/* Tabs */}
                <div className="flex gap-2 mb-6 border-b border-[--color-border]">
                  <button
                    onClick={() => setActiveTab("path")}
                    className={`px-4 py-3 font-medium border-b-2 transition-colors ${
                      activeTab === "path"
                        ? "text-[--accent-primary] border-[--accent-primary]"
                        : "text-[--color-text-secondary] border-transparent hover:text-[--color-text-primary]"
                    }`}
                  >
                    Learning Path
                  </button>
                  <button
                    onClick={() => setActiveTab("graph")}
                    className={`px-4 py-3 font-medium border-b-2 transition-colors ${
                      activeTab === "graph"
                        ? "text-[--accent-primary] border-[--accent-primary]"
                        : "text-[--color-text-secondary] border-transparent hover:text-[--color-text-primary]"
                    }`}
                  >
                    Knowledge Graph
                  </button>
                </div>

                {/* Tab content */}
                {activeTab === "path" ? (
                  <RoadmapGraph
                    phases={selectedRoadmap.phases}
                    onNodeClick={setSelectedNode}
                  />
                ) : (
                  <LearningGraph nodes={graphNodes} connections={graphConnections} />
                )}

                {/* Guided learning CTA */}
                {activeTab === "path" && (
                  <div className="mt-12 p-6 rounded-lg bg-gradient-to-r from-[rgb(124,58,237,0.1)] to-[rgb(6,182,212,0.1)] border border-[--color-border]">
                    <h3 className="font-bold mb-2 flex items-center gap-2">
                      <Zap className="w-5 h-5 text-[--accent-primary]" />
                      Ready to start learning?
                    </h3>
                    <p className="text-sm text-[--color-text-secondary] mb-4">
                      Begin with the first available lesson in this roadmap. We&apos;ll guide you
                      step by step through the curriculum.
                    </p>
                    <Link
                      href="/learn"
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-[--accent-primary] to-[--accent-cyan] text-white font-semibold hover:opacity-90 transition-opacity"
                    >
                      Start Learning
                      <ArrowRight className="w-4 h-4" />
                    </Link>
                  </div>
                )}
              </div>

              {/* Sidebar - Progress tracking */}
              <div className="lg:col-span-1">
                <div className="sticky top-24">
                  <RoadmapProgress roadmap={selectedRoadmap} />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Node detail panel */}
      <NodeDetailPanel
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onStart={
          selectedNode && (() => {
            const resource = selectedNode.linkedResources[0];
            if (!resource) return undefined;

            let targetUrl: string | null = null;
            if (resource.type === "topic") {
              const slug = resource.url.split("/").pop() || "";
              targetUrl = resolveTopicPath(slug);
            } else {
              const slug = resource.url.split("/").pop() || "";
              if (validSlugs[resource.type]?.includes(slug)) {
                targetUrl = resource.url;
              }
            }
            return targetUrl ? () => router.push(targetUrl as string) : undefined;
          })() || undefined
        }
      />
    </div>
  );
}
