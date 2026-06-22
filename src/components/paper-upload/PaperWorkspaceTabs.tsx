'use client';

import { useState } from 'react';
import { Shapes, Sigma, Clock3, Share2, Network, Cpu } from 'lucide-react';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { PaperKnowledgeGraph } from '@/components/paper-upload/PaperKnowledgeGraph';
import type { KGNode, KGEdge } from '@/components/paper-upload/PaperKnowledgeGraph';
import { ArchitectureBlueprintViewer } from '@/components/paper-upload/ArchitectureBlueprintViewer';
import type { ArchitectureBlueprint } from '@/components/paper-upload/ArchitectureBlueprintViewer';
import { ExecutableGraphViewer } from '@/components/paper-upload/ExecutableGraphViewer';
import type { ExecutableGraph } from '@/components/paper-upload/ExecutableGraphViewer';

// ---------------------------------------------------------------------------
// Types (mirroring the server-side PaperDetailResponse)
// ---------------------------------------------------------------------------

type FigureItem  = { id: string; page: number; caption?: string | null; has_binary?: boolean };
type EquationItem = { id: string; page: number; text: string };

export interface PaperWorkspaceTabsProps {
  fullTitle: string;
  abstract: string | null;
  architectureType: string;
  depth: number;
  parameterCount: number;
  flops: number;
  figures: FigureItem[];
  equations: EquationItem[];
  textExtractionMethod: string | undefined;
  pageCount: number | undefined;
  detectedComponents: string[] | undefined;
  moduleCount: number;
  knowledgeGraph: { nodes: KGNode[]; edges: KGEdge[] };
  architectureBlueprint: ArchitectureBlueprint | null;
  executableGraph: ExecutableGraph | null;
}

// ---------------------------------------------------------------------------
// Inner components (no 'use client' needed — they live inside one)
// ---------------------------------------------------------------------------

function SectionCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-3xl border border-[--color-border] bg-[--bg-panel] p-5 shadow-[0_24px_80px_rgba(0,0,0,0.12)]">
      <div className="mb-4 flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-[--color-border] bg-[--bg-surface] text-[--accent-cyan]">
          {icon}
        </div>
        <div>
          <SectionLabel>{title}</SectionLabel>
        </div>
      </div>
      {children}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

type TabId =
  | 'overview'
  | 'figures'
  | 'equations'
  | 'knowledge-graph'
  | 'architecture-blueprint'
  | 'executable-graph';

const TABS: { id: TabId; label: string }[] = [
  { id: 'overview',               label: 'Overview'             },
  { id: 'figures',                label: 'Figures'              },
  { id: 'equations',              label: 'Equations'            },
  { id: 'knowledge-graph',        label: 'Knowledge Graph'      },
  { id: 'architecture-blueprint', label: 'Architecture Blueprint' },
  { id: 'executable-graph',       label: 'Executable Graph'     },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function PaperWorkspaceTabs({
  fullTitle,
  abstract,
  architectureType,
  depth,
  parameterCount,
  flops,
  figures,
  equations,
  textExtractionMethod,
  pageCount,
  detectedComponents,
  moduleCount,
  knowledgeGraph,
  architectureBlueprint,
  executableGraph,
}: PaperWorkspaceTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>('overview');

  return (
    <div className="h-full overflow-y-auto p-5">
      <div className="rounded-[28px] border border-[--color-border] bg-[linear-gradient(180deg,rgba(14,18,29,0.98),rgba(8,10,16,0.98))] p-6 shadow-[0_30px_90px_rgba(0,0,0,0.18)]">
        <SectionLabel>Generated Paper Workspace</SectionLabel>
        <h1 className="mt-2 text-3xl font-semibold text-[--color-text-primary]">{fullTitle}</h1>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-[--color-text-secondary]">
          {abstract ?? 'This workspace was generated directly from the uploaded PDF and persisted in the paper database.'}
        </p>

        {/* Tab bar */}
        <div className="mt-6 flex gap-1 border-b border-[--color-border]" role="tablist" aria-label="Paper workspace tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`tabpanel-${tab.id}`}
              id={`tab-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-[--accent-primary] text-[--accent-primary]'
                  : 'border-transparent text-[--color-text-secondary] hover:text-[--color-text-primary]'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Overview */}
        <div
          id="tabpanel-overview"
          role="tabpanel"
          aria-labelledby="tab-overview"
          hidden={activeTab !== 'overview'}
        >
          <div className="mt-6 grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Architecture</div>
              <div className="mt-2 text-lg font-semibold text-[--color-text-primary]">{architectureType}</div>
              <div className="mt-1 text-sm text-[--color-text-secondary]">{depth} layers deep</div>
            </div>
            <div className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Compute</div>
              <div className="mt-2 text-lg font-semibold text-[--color-text-primary]">{parameterCount.toLocaleString()}</div>
              <div className="mt-1 text-sm text-[--color-text-secondary]">Estimated parameters</div>
            </div>
            <div className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-4">
              <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">FLOPs</div>
              <div className="mt-2 text-lg font-semibold text-[--color-text-primary]">{flops.toLocaleString()}</div>
              <div className="mt-1 text-sm text-[--color-text-secondary]">Normalized analysis score</div>
            </div>
          </div>

          <div className="mt-6">
            <SectionCard title="Ingestion Timeline" icon={<Clock3 className="h-5 w-5" />}>
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-3 text-sm text-[--color-text-secondary]">
                  <div className="font-medium text-[--color-text-primary]">Text extraction</div>
                  <div className="mt-1">{textExtractionMethod ?? 'unknown'}</div>
                </div>
                <div className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-3 text-sm text-[--color-text-secondary]">
                  <div className="font-medium text-[--color-text-primary]">Pages</div>
                  <div className="mt-1">{pageCount ?? 0}</div>
                </div>
                <div className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-3 text-sm text-[--color-text-secondary]">
                  <div className="font-medium text-[--color-text-primary]">Detected components</div>
                  <div className="mt-1">{detectedComponents?.join(', ') ?? 'None'}</div>
                </div>
                <div className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-3 text-sm text-[--color-text-secondary]">
                  <div className="font-medium text-[--color-text-primary]">Modules</div>
                  <div className="mt-1">{moduleCount}</div>
                </div>
              </div>
            </SectionCard>
          </div>
        </div>

        {/* Figures */}
        <div
          id="tabpanel-figures"
          role="tabpanel"
          aria-labelledby="tab-figures"
          hidden={activeTab !== 'figures'}
          className="mt-6"
        >
          <SectionCard title="Extracted Figures" icon={<Shapes className="h-5 w-5" />}>
            {figures.length > 0 ? (
              <div className="space-y-3">
                {figures.map((figure) => (
                  <div key={figure.id} className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-3">
                    <div className="text-sm font-medium text-[--color-text-primary]">Figure {figure.page}</div>
                    <div className="mt-1 text-xs text-[--color-text-secondary]">
                      {figure.caption ?? 'Caption not detected'}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-[--color-text-secondary]">No extractable figures were detected in this PDF.</div>
            )}
          </SectionCard>
        </div>

        {/* Equations */}
        <div
          id="tabpanel-equations"
          role="tabpanel"
          aria-labelledby="tab-equations"
          hidden={activeTab !== 'equations'}
          className="mt-6"
        >
          <SectionCard title="Extracted Equations" icon={<Sigma className="h-5 w-5" />}>
            {equations.length > 0 ? (
              <div className="space-y-3">
                {equations.slice(0, 16).map((equation) => (
                  <div key={equation.id} className="rounded-2xl border border-[--color-border] bg-[--bg-body] p-3 font-mono text-xs leading-6 text-[--color-text-primary]">
                    {equation.text}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-[--color-text-secondary]">No equation-like spans were detected in the extracted text.</div>
            )}
          </SectionCard>
        </div>

        {/* Knowledge Graph */}
        <div
          id="tabpanel-knowledge-graph"
          role="tabpanel"
          aria-labelledby="tab-knowledge-graph"
          hidden={activeTab !== 'knowledge-graph'}
          className="mt-6"
          style={{ height: 520 }}
        >
          <SectionCard title="Knowledge Graph" icon={<Share2 className="h-5 w-5" />}>
            <div style={{ height: 400 }}>
              <PaperKnowledgeGraph
                nodes={knowledgeGraph.nodes}
                edges={knowledgeGraph.edges}
              />
            </div>
          </SectionCard>
        </div>

        {/* Architecture Blueprint */}
        <div
          id="tabpanel-architecture-blueprint"
          role="tabpanel"
          aria-labelledby="tab-architecture-blueprint"
          hidden={activeTab !== 'architecture-blueprint'}
          className="mt-6"
        >
          <SectionCard title="Architecture Blueprint" icon={<Network className="h-5 w-5" />}>
            <ArchitectureBlueprintViewer blueprint={architectureBlueprint} />
          </SectionCard>
        </div>

        {/* Executable Graph */}
        <div
          id="tabpanel-executable-graph"
          role="tabpanel"
          aria-labelledby="tab-executable-graph"
          hidden={activeTab !== 'executable-graph'}
          className="mt-6"
        >
          <SectionCard title="Executable Graph" icon={<Cpu className="h-5 w-5" />}>
            <ExecutableGraphViewer graph={executableGraph} />
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
