'use client';

import { useState, useCallback, useMemo, useEffect, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Share2, Check, Loader2 } from 'lucide-react';
import { formatParams, formatFlops, saveGraph, fetchGraph } from '@/lib/model-viz-api';
import type { ParsedGraph, ParsedNode } from '@/lib/model-viz-api';
import UploadZone from '@/components/model-viz/UploadZone';
import GraphCanvas from '@/components/model-viz/GraphCanvas';
import InspectPanel from '@/components/model-viz/InspectPanel';

// ── inner page (needs useSearchParams → must be inside <Suspense>) ────────────

function ModelVizContent() {
  const searchParams = useSearchParams();
  const graphIdParam = searchParams.get('g');

  const [graph, setGraph] = useState<ParsedGraph | null>(null);
  const [modelName, setModelName] = useState('');
  const [modelFormat, setModelFormat] = useState<'onnx' | 'pytorch'>('onnx');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Save / share state
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [shareUrl, setShareUrl] = useState('');

  // ── load by URL param (?g=ID) ─────────────────────────────────────────────
  useEffect(() => {
    if (!graphIdParam) return;
    const id = parseInt(graphIdParam, 10);
    if (isNaN(id)) return;

    fetchGraph(id)
      .then((saved) => {
        setGraph(saved.graph_data as ParsedGraph);
        setModelName(saved.name);
        setModelFormat(saved.format);
        setSelectedNodeId(null);
      })
      .catch(() => {
        // Graph not found or fetch failed — just show the upload screen
      });
  }, [graphIdParam]);

  // ── callbacks ──────────────────────────────────────────────────────────────

  const handleParsed = useCallback(
    (g: ParsedGraph, filename: string, format: 'onnx' | 'pytorch') => {
      setGraph(g);
      setModelName(filename);
      setModelFormat(format);
      setSelectedNodeId(null);
      setSaveState('idle');
      setShareUrl('');
    },
    [],
  );

  const handleNewUpload = useCallback(() => {
    setGraph(null);
    setSelectedNodeId(null);
    setSaveState('idle');
    setShareUrl('');
    // Clear the ?g= param from URL without a full navigation
    const url = new URL(window.location.href);
    url.searchParams.delete('g');
    window.history.replaceState(null, '', url.toString());
  }, []);

  const handleSave = useCallback(async () => {
    if (!graph) return;
    setSaveState('saving');
    try {
      const { id } = await saveGraph(graph, modelName, modelFormat);
      const url = `${window.location.origin}/model-viz?g=${id}`;
      setShareUrl(url);
      await navigator.clipboard.writeText(url).catch(() => {});
      setSaveState('saved');
    } catch {
      setSaveState('error');
    }
  }, [graph, modelName, modelFormat]);

  // ── derived ────────────────────────────────────────────────────────────────

  const selectedNode = useMemo<ParsedNode | null>(() => {
    if (!selectedNodeId || !graph) return null;
    return graph.nodes.find((n) => n.id === selectedNodeId) ?? null;
  }, [selectedNodeId, graph]);

  // ── render: upload state ──────────────────────────────────────────────────

  if (!graph) {
    return (
      <div
        style={{
          height: 'calc(100vh - 56px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#0a0a0a',
          overflowY: 'auto',
        }}
      >
        <UploadZone onParsed={handleParsed} />
      </div>
    );
  }

  // ── render: graph view ────────────────────────────────────────────────────

  return (
    <div
      style={{
        height: 'calc(100vh - 56px)',
        display: 'flex',
        flexDirection: 'column',
        background: '#0a0a0a',
        overflow: 'hidden',
      }}
    >
      {/* Top bar */}
      <div
        style={{
          height: 48,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '0 16px',
          borderBottom: '1px solid #1a1a1a',
          flexShrink: 0,
        }}
      >
        {/* Model name */}
        <span
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: '#f5f5f5',
            maxWidth: 220,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {modelName}
        </span>

        {/* Format badge */}
        <span
          style={{
            fontSize: 9,
            fontFamily: 'monospace',
            color: modelFormat === 'pytorch' ? '#f97316' : '#A78BFA',
            background: modelFormat === 'pytorch' ? 'rgba(249,115,22,0.1)' : 'rgba(167,139,250,0.1)',
            border: `1px solid ${modelFormat === 'pytorch' ? 'rgba(249,115,22,0.2)' : 'rgba(167,139,250,0.2)'}`,
            padding: '2px 8px',
            borderRadius: 4,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}
        >
          {modelFormat === 'pytorch'
            ? (graph.meta.method === 'named_modules' ? 'PyTorch (tree)' : 'PyTorch (fx)')
            : 'ONNX'}
        </span>

        <StatBadge label="Layers" value={graph.meta.total_nodes} />
        <StatBadge label="Params" value={formatParams(graph.meta.total_params)} />
        {!!graph.meta.total_flops && graph.meta.total_flops > 0 && (
          <StatBadge label="FLOPs" value={formatFlops(graph.meta.total_flops)} />
        )}
        {graph.meta.opset_version > 0 && (
          <StatBadge label="Opset" value={`v${graph.meta.opset_version}`} />
        )}

        <div style={{ flex: 1 }} />

        {/* Share / save button */}
        <ShareButton state={saveState} shareUrl={shareUrl} onSave={handleSave} />

        {/* New upload button */}
        <button
          onClick={handleNewUpload}
          style={{
            fontSize: 11,
            color: '#737373',
            background: 'none',
            border: '1px solid #2a2a2a',
            padding: '4px 12px',
            borderRadius: 6,
            cursor: 'pointer',
          }}
        >
          New Upload
        </button>
      </div>

      {/* Canvas + Inspect Panel */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <GraphCanvas
            nodes={graph.nodes}
            edges={graph.edges}
            selectedNodeId={selectedNodeId}
            onNodeClick={setSelectedNodeId}
          />
        </div>

        {selectedNode && (
          <InspectPanel
            node={selectedNode}
            onClose={() => setSelectedNodeId(null)}
          />
        )}
      </div>
    </div>
  );
}

// ── page export (Suspense required for useSearchParams in Next.js 15) ─────────

export default function ModelVizPage() {
  return (
    <Suspense fallback={
      <div style={{
        height: 'calc(100vh - 56px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0a0a0a',
      }}>
        <Loader2 size={24} style={{ color: '#525252' }} className="animate-spin" />
      </div>
    }>
      <ModelVizContent />
    </Suspense>
  );
}

// ── small components ──────────────────────────────────────────────────────────

function StatBadge({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 5,
        padding: '3px 8px',
        background: '#111',
        borderRadius: 6,
        border: '1px solid #1a1a1a',
      }}
    >
      <span style={{ fontSize: 9, color: '#404040', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        {label}
      </span>
      <span style={{ fontSize: 11, fontWeight: 600, color: '#a3a3a3', fontFamily: 'monospace' }}>
        {value}
      </span>
    </div>
  );
}

type ShareState = 'idle' | 'saving' | 'saved' | 'error';

function ShareButton({
  state,
  shareUrl,
  onSave,
}: {
  state: ShareState;
  shareUrl: string;
  onSave: () => void;
}) {
  if (state === 'saved') {
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <Check size={12} style={{ color: '#22c55e' }} />
        <span style={{ fontSize: 11, color: '#22c55e' }}>Link copied!</span>
        <a
          href={shareUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            fontSize: 11,
            color: '#737373',
            fontFamily: 'monospace',
            maxWidth: 160,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            textDecoration: 'none',
          }}
        >
          {shareUrl.replace(/^https?:\/\//, '')}
        </a>
      </div>
    );
  }

  if (state === 'error') {
    return (
      <span style={{ fontSize: 11, color: '#f87171' }}>Save failed</span>
    );
  }

  return (
    <button
      onClick={onSave}
      disabled={state === 'saving'}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        fontSize: 11,
        color: state === 'saving' ? '#525252' : '#A78BFA',
        background: 'none',
        border: `1px solid ${state === 'saving' ? '#2a2a2a' : 'rgba(167,139,250,0.3)'}`,
        padding: '4px 12px',
        borderRadius: 6,
        cursor: state === 'saving' ? 'default' : 'pointer',
        transition: 'all 0.1s',
      }}
    >
      {state === 'saving' ? (
        <Loader2 size={11} className="animate-spin" />
      ) : (
        <Share2 size={11} />
      )}
      {state === 'saving' ? 'Saving…' : 'Share'}
    </button>
  );
}
