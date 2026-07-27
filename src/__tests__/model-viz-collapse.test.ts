import { describe, it, expect } from 'vitest';
import { collapseGraph } from '@/lib/model-viz-api';
import type { ParsedNode, ParsedEdge, MotifGroup } from '@/lib/model-viz-api';

function node(id: string, op = 'Op'): ParsedNode {
  return {
    id, op_type: op, label: id, inputs: [], outputs: [],
    input_shapes: {}, output_shapes: {}, primary_out_shape: [], params: 0, attrs: {},
  };
}
function edge(source: string, target: string): ParsedEdge {
  return { id: `${source}-${target}`, source, target, shape: [], tensor: '' };
}
function group(id: string, node_ids: string[]): MotifGroup {
  return {
    id, signature: 'Conv → BN → Relu', label: 'Conv → BN → Relu', node_ids,
    member_ops: ['Conv', 'BatchNormalization', 'Relu'], repeat_index: 0, repeat_count: 2,
    flops: 0, params: 0, memory_mb: 0, severity: 'low',
  };
}

// linear chain: 0 -> [1,2,3] -> [4,5,6] -> 7, with two Conv→BN→Relu blocks
const NODES = ['node_0', 'node_1', 'node_2', 'node_3', 'node_4', 'node_5', 'node_6', 'node_7'].map((id) => node(id));
const EDGES = [
  edge('node_0', 'node_1'), edge('node_1', 'node_2'), edge('node_2', 'node_3'),
  edge('node_3', 'node_4'), edge('node_4', 'node_5'), edge('node_5', 'node_6'),
  edge('node_6', 'node_7'),
];
const GROUPS = [group('grp_0', ['node_1', 'node_2', 'node_3']), group('grp_1', ['node_4', 'node_5', 'node_6'])];

describe('collapseGraph', () => {
  it('collapses each group into one super-node and remaps edges', () => {
    const { nodes, edges } = collapseGraph(NODES, EDGES, GROUPS, new Set());
    expect(nodes.map((n) => n.id)).toEqual(['node_0', 'grp_0', 'grp_1', 'node_7']);
    expect(nodes.find((n) => n.id === 'grp_0')!.kind).toBe('group');
    const pairs = edges.map((e) => `${e.source}->${e.target}`).sort();
    expect(pairs).toEqual(['grp_0->grp_1', 'grp_1->node_7', 'node_0->grp_0']);
  });

  it('shows member nodes for an expanded group', () => {
    const { nodes, edges } = collapseGraph(NODES, EDGES, GROUPS, new Set(['grp_0']));
    expect(nodes.map((n) => n.id)).toEqual(['node_0', 'node_1', 'node_2', 'node_3', 'grp_1', 'node_7']);
    const pairs = edges.map((e) => `${e.source}->${e.target}`).sort();
    expect(pairs).toEqual(['grp_1->node_7', 'node_0->node_1', 'node_1->node_2', 'node_2->node_3', 'node_3->grp_1']);
  });

  it('with no groups the graph passes through unchanged in shape', () => {
    const { nodes, edges } = collapseGraph(NODES, EDGES, [], new Set());
    expect(nodes).toHaveLength(NODES.length);
    expect(nodes.every((n) => n.kind === 'node')).toBe(true);
    expect(edges).toHaveLength(EDGES.length);
  });
});
