'use client';

import React, { useMemo, useCallback } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  type Node,
  type Edge,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import * as dagre from '@dagrejs/dagre';

import NodeCard from './NodeCard';
import { getOpColor } from '@/lib/model-viz-api';
import type { ParsedNode, ParsedEdge } from '@/lib/model-viz-api';

// Stable reference — must be outside component or React Flow re-mounts on every render
const NODE_TYPES = { modelNode: NodeCard };

const NODE_W = 200;
const NODE_H = 72;

function applyDagreLayout(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: 'TB', ranksep: 80, nodesep: 40, marginx: 40, marginy: 40 });
  g.setDefaultEdgeLabel(() => ({}));

  nodes.forEach((n) => g.setNode(n.id, { width: NODE_W, height: NODE_H }));
  edges.forEach((e) => {
    try {
      g.setEdge(e.source, e.target);
    } catch {
      // skip malformed edges (disconnected nodes in unusual ONNX models)
    }
  });

  dagre.layout(g);

  return nodes.map((n) => {
    const pos = g.node(n.id);
    return {
      ...n,
      position: pos
        ? { x: pos.x - NODE_W / 2, y: pos.y - NODE_H / 2 }
        : { x: 0, y: 0 },
    };
  });
}

type Props = {
  nodes: ParsedNode[];
  edges: ParsedEdge[];
  selectedNodeId: string | null;
  onNodeClick: (nodeId: string) => void;
};

export default function GraphCanvas({ nodes, edges, selectedNodeId, onNodeClick }: Props) {
  // Dagre layout — only recomputes when the graph topology changes, not on selection
  const laidOutNodes = useMemo<Node[]>(() => {
    const raw: Node[] = nodes.map((n) => ({
      id: n.id,
      type: 'modelNode',
      position: { x: 0, y: 0 },
      data: n,
      draggable: false,
      selectable: true,
    }));
    return applyDagreLayout(raw, edges.map((e) => ({ id: e.id, source: e.source, target: e.target })));
  }, [nodes, edges]);

  // Selection is a cheap overlay — does not re-run layout
  const rfNodes = useMemo<Node[]>(
    () => laidOutNodes.map((n) => ({ ...n, selected: n.id === selectedNodeId })),
    [laidOutNodes, selectedNodeId],
  );

  // Convert ParsedEdge[] → React Flow Edge[] (style comes from defaultEdgeOptions)
  const rfEdges = useMemo<Edge[]>(
    () => edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
    [edges],
  );

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => onNodeClick(node.id),
    [onNodeClick],
  );

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        nodeTypes={NODE_TYPES}
        onNodeClick={handleNodeClick}
        fitView
        fitViewOptions={{ padding: 0.12, maxZoom: 1.5 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        panOnScroll={false}
        zoomOnScroll
        style={{ background: '#0a0a0a' }}
        defaultEdgeOptions={{ type: 'smoothstep', style: { stroke: '#2a2a2a', strokeWidth: 1.5 }, animated: false }}
      >
        <Background
          color="#1c1c1c"
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
        />
        <Controls
          showInteractive={false}
          style={{
            background: '#111',
            border: '1px solid #2a2a2a',
            borderRadius: 8,
          }}
        />
        <MiniMap
          style={{
            background: '#111',
            border: '1px solid #2a2a2a',
            borderRadius: 8,
          }}
          nodeColor={(n) => getOpColor((n.data as ParsedNode)?.op_type ?? '')}
          maskColor="rgba(0,0,0,0.65)"
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  );
}
