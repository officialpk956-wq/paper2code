import { describe, it, expect } from 'vitest';
import { matchNodes, dataPath } from '@/lib/model-viz-api';
import type { ParsedNode } from '@/lib/model-viz-api';

function node(id: string, op: string, label = ''): ParsedNode {
  return {
    id, op_type: op, label: label || id, inputs: [], outputs: [],
    input_shapes: {}, output_shapes: {}, primary_out_shape: [], params: 0, attrs: {},
  };
}

const NODES = [
  node('n0', 'Conv', 'backbone.conv1'),
  node('n1', 'Relu', 'backbone.act1'),
  node('n2', 'Conv', 'backbone.conv2'),
  node('n3', 'Gemm', 'classifier.fc'),
];

// diamond: n0 -> n1 -> n3 ; n0 -> n2 -> n3
const EDGES = [
  { source: 'n0', target: 'n1' },
  { source: 'n0', target: 'n2' },
  { source: 'n1', target: 'n3' },
  { source: 'n2', target: 'n3' },
];

describe('matchNodes', () => {
  it('matches by op_type (case-insensitive)', () => {
    expect(matchNodes(NODES, 'conv')).toEqual(new Set(['n0', 'n2']));
  });
  it('matches by label substring', () => {
    expect(matchNodes(NODES, 'classifier')).toEqual(new Set(['n3']));
  });
  it('empty query matches nothing', () => {
    expect(matchNodes(NODES, '   ')).toEqual(new Set());
  });
});

describe('dataPath', () => {
  it('includes the node, all ancestors and all descendants', () => {
    // path through n1: itself + ancestor n0 + descendant n3 (NOT the sibling n2)
    expect(dataPath(EDGES, 'n1')).toEqual(new Set(['n1', 'n0', 'n3']));
  });
  it('a source node yields itself + everything downstream', () => {
    expect(dataPath(EDGES, 'n0')).toEqual(new Set(['n0', 'n1', 'n2', 'n3']));
  });
  it('a sink node yields itself + everything upstream', () => {
    expect(dataPath(EDGES, 'n3')).toEqual(new Set(['n3', 'n1', 'n2', 'n0']));
  });
  it('an isolated node yields just itself', () => {
    expect(dataPath(EDGES, 'lonely')).toEqual(new Set(['lonely']));
  });
});
