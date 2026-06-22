import { notFound } from 'next/navigation';
import type { ReactNode } from 'react';
import { ThreeColumnLayout } from '@/components/layout/three-column-layout';
import { SectionLabel } from '@/components/ui/SectionLabel';
import { PaperWorkspaceTabs } from '@/components/paper-upload/PaperWorkspaceTabs';
import type { KGNode, KGEdge } from '@/components/paper-upload/PaperKnowledgeGraph';
import type { ArchitectureBlueprint } from '@/components/paper-upload/ArchitectureBlueprintViewer';
import type { ExecutableGraph } from '@/components/paper-upload/ExecutableGraphViewer';
import { getBackendUrl } from '@/lib/backend';
import { FileText, Layers3, Clock3 } from 'lucide-react';

type PaperDetailResponse = {
  metadata: {
    id: number;
    title: string;
    full_title: string;
    authors: string | null;
    abstract: string | null;
    architecture_type: string;
    status: string;
    source_filename?: string | null;
    figure_count?: number;
    equation_count?: number;
  };
  module_summary: Array<{
    id: number;
    order_index: number;
    layer_name: string;
    module_type: string;
    explanation: string | null;
    tensor_flow: unknown;
    graph_nodes: unknown;
    flops_context: unknown;
  }>;
  architecture_statistics: {
    depth: number;
    node_count: number;
    edge_count: number;
  };
  architecture_graph: {
    nodes: Array<{ id: string; type: string; label: string }>;
    edges: Array<{ source: string; target: string; type: string }>;
    ingestion?: {
      source_filename?: string;
      page_count?: number;
      text_extraction_method?: string;
      figure_count?: number;
      equation_count?: number;
      figures?: Array<{ id: string; page: number; caption?: string | null; has_binary?: boolean }>;
      equations?: Array<{ id: string; page: number; text: string }>;
      detected_components?: string[];
      module_count?: number;
      raw_text_excerpt?: string;
      knowledge_graph?: { nodes: KGNode[]; edges: KGEdge[] };
      architecture_blueprint?: ArchitectureBlueprint;
      executable_graph?: ExecutableGraph;
    };
  };
  flops: number;
  parameter_count: number;
  ingestion?: PaperDetailResponse['architecture_graph']['ingestion'];
};

async function loadPaper(paperId: string): Promise<PaperDetailResponse | null> {
  const response = await fetch(getBackendUrl(`/api/papers/${paperId}`), { cache: 'no-store' });
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error(`Failed to load paper ${paperId}`);
  }
  return (await response.json()) as PaperDetailResponse;
}

function SectionCard({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
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

export default async function GeneratedPaperPage({ params }: { params: Promise<{ paperId: string }> }) {
  const { paperId } = await params;
  const paper = await loadPaper(paperId);

  if (!paper) {
    notFound();
  }

  const ingestion = paper.ingestion ?? paper.architecture_graph.ingestion ?? {};
  const authors = paper.metadata.authors ? paper.metadata.authors.split(',').map((author) => author.trim()).filter(Boolean) : [];
  const figures = ingestion.figures ?? [];
  const equations = ingestion.equations ?? [];
  const modules = paper.module_summary;

  const left = (
    <div className="h-full overflow-y-auto border-r border-[--color-border] bg-[--bg-body] p-4">
      <SectionCard
        title="Paper Record"
        icon={<FileText className="h-5 w-5" />}
      >
        <div className="space-y-3 text-sm text-[--color-text-secondary]">
          <div>
            <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Title</div>
            <div className="mt-1 text-[--color-text-primary]">{paper.metadata.full_title}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Source</div>
            <div className="mt-1">{ingestion.source_filename ?? paper.metadata.source_filename ?? 'Uploaded PDF'}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Status</div>
            <div className="mt-1">{paper.metadata.status}</div>
          </div>
          {authors.length > 0 && (
            <div>
              <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Authors</div>
              <div className="mt-1">{authors.join(', ')}</div>
            </div>
          )}
        </div>
      </SectionCard>

      <div className="mt-4 grid gap-3">
        <div className="rounded-2xl border border-[--color-border] bg-[--bg-panel] p-4">
          <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Modules</div>
          <div className="mt-2 text-2xl font-semibold text-[--color-text-primary]">{modules.length}</div>
        </div>
        <div className="rounded-2xl border border-[--color-border] bg-[--bg-panel] p-4">
          <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Figures</div>
          <div className="mt-2 text-2xl font-semibold text-[--color-text-primary]">{ingestion.figure_count ?? figures.length}</div>
        </div>
        <div className="rounded-2xl border border-[--color-border] bg-[--bg-panel] p-4">
          <div className="text-xs uppercase tracking-[0.16em] text-[--color-text-tertiary]">Equations</div>
          <div className="mt-2 text-2xl font-semibold text-[--color-text-primary]">{ingestion.equation_count ?? equations.length}</div>
        </div>
      </div>

      <SectionCard
        title="Module Outline"
        icon={<Layers3 className="h-5 w-5" />}
      >
        <div className="space-y-3">
          {modules.map((module) => (
            <div key={module.id} className="rounded-2xl border border-[--color-border] bg-[--bg-surface] p-3">
              <div className="text-sm font-medium text-[--color-text-primary]">{module.layer_name}</div>
              <div className="mt-1 text-xs text-[--color-text-tertiary]">{module.module_type}</div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );

  const knowledgeGraph = ingestion.knowledge_graph ?? { nodes: [], edges: [] };
  const architectureBlueprint = ingestion.architecture_blueprint ?? null;
  const executableGraph = ingestion.executable_graph ?? null;

  const center = (
    <PaperWorkspaceTabs
      fullTitle={paper.metadata.full_title}
      abstract={paper.metadata.abstract}
      architectureType={paper.metadata.architecture_type}
      depth={paper.architecture_statistics.depth}
      parameterCount={paper.parameter_count}
      flops={paper.flops}
      figures={figures}
      equations={equations}
      textExtractionMethod={ingestion.text_extraction_method}
      pageCount={ingestion.page_count}
      detectedComponents={ingestion.detected_components}
      moduleCount={ingestion.module_count ?? modules.length}
      knowledgeGraph={knowledgeGraph}
      architectureBlueprint={architectureBlueprint}
      executableGraph={executableGraph}
    />
  );

  const right = (
    <div className="h-full overflow-y-auto border-l border-[--color-border] bg-[--bg-body] p-4">
      <SectionCard title="Workspace Notes" icon={<Clock3 className="h-5 w-5" />}>
        <div className="space-y-3 text-sm text-[--color-text-secondary]">
          <p>
            This workspace was generated from a real uploaded PDF and backed by the persisted Paper record.
          </p>
          <p>
            The raw extraction payload is stored with the paper so the workspace can be reloaded without rerunning the parser.
          </p>
        </div>
      </SectionCard>

      <SectionCard title="Paper Graph" icon={<Layers3 className="h-5 w-5" />}>
        <div className="space-y-3 text-sm text-[--color-text-secondary]">
          <div>Nodes: {paper.architecture_statistics.node_count}</div>
          <div>Edges: {paper.architecture_statistics.edge_count}</div>
          <div>Modules: {modules.length}</div>
        </div>
      </SectionCard>

      <SectionCard title="Raw Source" icon={<FileText className="h-5 w-5" />}>
        <div className="max-h-[18rem] overflow-y-auto rounded-2xl border border-[--color-border] bg-[--bg-surface] p-3 text-xs leading-6 text-[--color-text-secondary]">
          {ingestion.raw_text_excerpt ?? 'No raw excerpt persisted.'}
        </div>
      </SectionCard>
    </div>
  );

  return <ThreeColumnLayout left={left} center={center} right={right} leftWidth="w-[24rem]" rightWidth="w-[24rem]" />;
}
