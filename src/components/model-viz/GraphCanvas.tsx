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
import GroupNode from './GroupNode';
import { getOpColor, collapseGraph, severityColor } from '@/lib/model-viz-api';
import type { ParsedNode, ParsedEdge, MotifGroup } from '@/lib/model-viz-api';

// Stable reference — must be outside component or React Flow re-mounts on every render
const NODE_TYPES = { modelNode: NodeCard, groupNode: GroupNode };

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
  groups?: MotifGroup[];
  grouping?: boolean;
  expandedGroups?: Set<string>;
  onToggleGroup?: (groupId: string) => void;
  // when set, nodes NOT in this set (of ORIGINAL node ids) are dimmed; null = no dimming
  highlightIds?: Set<string> | null;
};

export default function GraphCanvas({
  nodes,
  edges,
  selectedNodeId,
  onNodeClick,
  groups = [],
  grouping = false,
  expandedGroups,
  onToggleGroup,
  highlightIds = null,
}: Props) {
  // Collapse repeated blocks into super-nodes when grouping is on (pure, memoized)
  const display = useMemo(() => {
    if (grouping && groups.length) {
      return collapseGraph(nodes, edges, groups, expandedGroups ?? new Set());
    }
    return {
      nodes: nodes.map((n) => ({ kind: 'node' as const, id: n.id, node: n })),
      edges: edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
    };
  }, [nodes, edges, groups, grouping, expandedGroups]);

  // Dagre layout — only recomputes when the (possibly collapsed) topology changes
  const laidOutNodes = useMemo<Node[]>(() => {
    const raw: Node[] = display.nodes.map((dn) =>
      dn.kind === 'group'
        ? { id: dn.id, type: 'groupNode', position: { x: 0, y: 0 }, data: { group: dn.group }, draggable: false, selectable: true }
        : { id: dn.id, type: 'modelNode', position: { x: 0, y: 0 }, data: dn.node, draggable: false, selectable: true },
    );
    return applyDagreLayout(raw, display.edges.map((e) => ({ id: e.id, source: e.source, target: e.target })));
  }, [display]);

  // Which DISPLAY ids are "active" (a group is active if any member is). null = all.
  const activeIds = useMemo<Set<string> | null>(() => {
    if (!highlightIds) return null;
    const active = new Set<string>();
    for (const dn of display.nodes) {
      const on = dn.kind === 'group' ? dn.group.node_ids.some((id) => highlightIds.has(id)) : highlightIds.has(dn.id);
      if (on) active.add(dn.id);
    }
    return active;
  }, [display, highlightIds]);

  // Selection + highlight dimming are cheap overlays — they do not re-run layout
  const rfNodes = useMemo<Node[]>(
    () =>
      laidOutNodes.map((n) => ({
        ...n,
        selected: n.id === selectedNodeId,
        style: { ...(n.style ?? {}), opacity: activeIds && !activeIds.has(n.id) ? 0.18 : 1 },
      })),
    [laidOutNodes, selectedNodeId, activeIds],
  );

  const rfEdges = useMemo<Edge[]>(
    () =>
      display.edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        style: activeIds && !(activeIds.has(e.source) && activeIds.has(e.target)) ? { opacity: 0.1 } : undefined,
      })),
    [display, activeIds],
  );

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      // clicking a collapsed super-node expands it; clicking a real node selects it
      if (node.type === 'groupNode') onToggleGroup?.(node.id);
      else onNodeClick(node.id);
    },
    [onNodeClick, onToggleGroup],
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
          nodeColor={(n) => {
            const d = n.data as { group?: MotifGroup } & Partial<ParsedNode>;
            return d?.group ? severityColor(d.group.severity) : getOpColor(d?.op_type ?? '');
          }}
          maskColor="rgba(0,0,0,0.65)"
          pannable
          zoomable
        />
      </ReactFlow>
    </div>
  );
}
