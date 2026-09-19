// ── Types matching the backend onnx_parser / pytorch_parser response ───────────

export type Severity = 'low' | 'medium' | 'high' | 'critical';

export type ParsedNode = {
  id: string;
  op_type: string;
  label: string;
  inputs: string[];
  outputs: string[];
  input_shapes: Record<string, number[]>;
  output_shapes: Record<string, number[]>;
  primary_out_shape: number[];
  params: number;
  attrs: Record<string, unknown>;
  // per-node compute cost (added by the backend estimator; optional so older
  // saved graphs without them still render)
  flops?: number;
  memory_mb?: number;
  severity?: Severity;
};

export type ParsedEdge = {
  id: string;
  source: string;
  target: string;
  shape: number[];
  tensor: string;
};

export type MotifGroup = {
  id: string;
  signature: string;
  label: string;
  node_ids: string[];
  member_ops: string[];
  repeat_index: number;
  repeat_count: number;
  flops: number;
  params: number;
  memory_mb: number;
  severity: Severity;
};

export type GraphMeta = {
  total_nodes: number;
  total_params: number;
  total_flops?: number;
  total_edges: number;
  motif_count?: number;
  grouped_nodes?: number;
  motifs?: { signature: string; count: number }[];
  graph_inputs: Record<string, number[]>;
  graph_outputs: Record<string, number[]>;
  ir_version: number;
  opset_version: number;
  method?: string;   // "symbolic_trace" | "named_modules" | "torchscript_onnx" — pytorch only
};

export type ParsedGraph = {
  nodes: ParsedNode[];
  edges: ParsedEdge[];
  groups?: MotifGroup[];
  meta: GraphMeta;
};

export type SavedGraph = {
  id: number;
  name: string;
  format: 'onnx' | 'pytorch';
  graph_data: ParsedGraph;
  created_at: string;
};

// ── Color map for layer types ──────────────────────────────────────────────────

const OP_COLOR_MAP: Record<string, string> = {
  // Convolutions
  Conv: '#6366f1',
  Conv2d: '#6366f1',
  ConvTranspose: '#6366f1',
  ConvTranspose2d: '#6366f1',
  // Pooling
  MaxPool: '#8b5cf6',
  MaxPool2d: '#8b5cf6',
  AveragePool: '#8b5cf6',
  AvgPool2d: '#8b5cf6',
  GlobalAveragePool: '#8b5cf6',
  GlobalMaxPool: '#8b5cf6',
  AdaptiveAvgPool2d: '#8b5cf6',
  // Normalization
  BatchNormalization: '#ec4899',
  BatchNorm2d: '#ec4899',
  LayerNormalization: '#ec4899',
  LayerNorm: '#ec4899',
  InstanceNormalization: '#ec4899',
  InstanceNorm2d: '#ec4899',
  GroupNormalization: '#ec4899',
  GroupNorm: '#ec4899',
  // Activations
  Relu: '#22c55e',
  ReLU: '#22c55e',
  LeakyRelu: '#22c55e',
  LeakyReLU: '#22c55e',
  Sigmoid: '#22c55e',
  Tanh: '#22c55e',
  Gelu: '#22c55e',
  GELU: '#22c55e',
  Selu: '#22c55e',
  Elu: '#22c55e',
  ELU: '#22c55e',
  HardSwish: '#22c55e',
  Hardswish: '#22c55e',
  Mish: '#22c55e',
  Softmax: '#22c55e',
  LogSoftmax: '#22c55e',
  SiLU: '#22c55e',
  // Linear / fully-connected
  Gemm: '#f59e0b',
  MatMul: '#f59e0b',
  Linear: '#f59e0b',
  // Attention
  Attention: '#f97316',
  MultiHeadAttention: '#f97316',
  MultiheadAttention: '#f97316',
  // Dropout / regularisation
  Dropout: '#64748b',
  Dropout2d: '#64748b',
  // Reshape ops
  Flatten: '#94a3b8',
  Reshape: '#94a3b8',
  Transpose: '#94a3b8',
  Squeeze: '#94a3b8',
  Unsqueeze: '#94a3b8',
};

export function getOpColor(opType: string): string {
  return OP_COLOR_MAP[opType] ?? '#64748b';
}

// ── Compute-cost helpers (FLOPs / memory / severity) ────────────────────────

const SEVERITY_COLOR: Record<Severity, string> = {
  low: '#525252',
  medium: '#eab308',
  high: '#f97316',
  critical: '#ef4444',
};

export function severityColor(s?: Severity): string {
  return SEVERITY_COLOR[s ?? 'low'];
}

/** Human-readable FLOPs (empty string for zero/undefined). */
export function formatFlops(n?: number): string {
  if (!n || n <= 0) return '';
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)} TFLOPs`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GFLOPs`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MFLOPs`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KFLOPs`;
  return `${n} FLOPs`;
}

// ── Motif collapse (repeated-block super-nodes) ─────────────────────────────

export type DisplayNode =
  | { kind: 'node'; id: string; node: ParsedNode }
  | { kind: 'group'; id: string; group: MotifGroup };

export type DisplayEdge = { id: string; source: string; target: string };

/**
 * Collapse each repeated-block group (unless expanded) into a single super-node,
 * remapping edges through it. Pure — the same inputs always give the same graph,
 * so it is unit-testable and safe to memoize. Preserves node order (a group is
 * emitted at the position of its first member).
 */
export function collapseGraph(
  nodes: ParsedNode[],
  edges: ParsedEdge[],
  groups: MotifGroup[],
  expandedGroupIds: Set<string>,
): { nodes: DisplayNode[]; edges: DisplayEdge[] } {
  const collapsed = groups.filter((g) => !expandedGroupIds.has(g.id));
  const groupById = new Map(collapsed.map((g) => [g.id, g]));
  const nodeToGroup = new Map<string, string>();
  for (const g of collapsed) for (const nid of g.node_ids) nodeToGroup.set(nid, g.id);

  const displayId = (nid: string) => nodeToGroup.get(nid) ?? nid;

  const dNodes: DisplayNode[] = [];
  const emitted = new Set<string>();
  for (const n of nodes) {
    const gid = nodeToGroup.get(n.id);
    if (gid) {
      if (!emitted.has(gid)) {
        emitted.add(gid);
        dNodes.push({ kind: 'group', id: gid, group: groupById.get(gid)! });
      }
    } else {
      dNodes.push({ kind: 'node', id: n.id, node: n });
    }
  }

  const seen = new Set<string>();
  const dEdges: DisplayEdge[] = [];
  for (const e of edges) {
    const s = displayId(e.source);
    const t = displayId(e.target);
    if (s === t) continue; // edge internal to one collapsed group
    const key = `${s}->${t}`;
    if (seen.has(key)) continue; // many member edges fold into one super-edge
    seen.add(key);
    dEdges.push({ id: `de_${key}`, source: s, target: t });
  }
  return { nodes: dNodes, edges: dEdges };
}

// ── Search + data-path highlighting ─────────────────────────────────────────

/** Node ids whose op_type or label contains `query` (case-insensitive). Empty
 *  query → empty set (caller treats "no highlight" as "show everything"). */
export function matchNodes(nodes: ParsedNode[], query: string): Set<string> {
  const q = query.trim().toLowerCase();
  if (!q) return new Set();
  const hits = new Set<string>();
  for (const n of nodes) {
    if (n.op_type.toLowerCase().includes(q) || (n.label ?? '').toLowerCase().includes(q)) {
      hits.add(n.id);
    }
  }
  return hits;
}

/** The data path through `nodeId`: the node itself plus every ancestor (upstream)
 *  and every descendant (downstream), via BFS over the directed edges. */
export function dataPath(edges: { source: string; target: string }[], nodeId: string): Set<string> {
  const succ = new Map<string, string[]>();
  const pred = new Map<string, string[]>();
  for (const e of edges) {
    (succ.get(e.source) ?? succ.set(e.source, []).get(e.source)!).push(e.target);
    (pred.get(e.target) ?? pred.set(e.target, []).get(e.target)!).push(e.source);
  }
  const path = new Set<string>([nodeId]);
  const walk = (adj: Map<string, string[]>) => {
    const stack = [nodeId];
    while (stack.length) {
      const cur = stack.pop()!;
      for (const nxt of adj.get(cur) ?? []) {
        if (!path.has(nxt)) {
          path.add(nxt);
          stack.push(nxt);
        }
      }
    }
  };
  walk(succ); // descendants
  walk(pred); // ancestors
  return path;
}

// ── HTTP calls ────────────────────────────────────────────────────────────────
// All calls go through src/lib/api.ts, which refreshes an expired access token
// on 401 and retries. Access tokens live 15 minutes; this module used raw
// fetch with a hand-built Authorization header, so fifteen minutes after
// login every upload here failed with "Could not validate credentials" while
// the rest of the app quietly refreshed and kept working.
//
// Timeouts are explicit: the shared default is 15s, which would abort the
// PyTorch parse -- ~30s in a warm E2B sandbox, up to 300s cold (the server
// side gives it _PYTORCH_SANDBOX_TIMEOUT = 300).

import { apiGet, apiPost, apiPostForm } from '@/lib/api';

const ONNX_PARSE_TIMEOUT_MS = 120_000;      // 50 MB upload + parse
const PYTORCH_PARSE_TIMEOUT_MS = 330_000;   // sandbox limit (300s) + margin

/** Parse an .onnx file. */
export async function parseModel(file: File): Promise<ParsedGraph> {
  const form = new FormData();
  form.append('file', file);
  return apiPostForm<ParsedGraph>('/api/model/parse', form, undefined, ONNX_PARSE_TIMEOUT_MS);
}

/** Parse a .pt / .pth file via the E2B sandbox. */
export async function parsePytorchModel(
  file: File,
  inputShape: number[],
): Promise<ParsedGraph> {
  const form = new FormData();
  form.append('file', file);
  form.append('input_shape', JSON.stringify(inputShape));
  return apiPostForm<ParsedGraph>('/api/model/parse-pytorch', form, undefined, PYTORCH_PARSE_TIMEOUT_MS);
}

/** Save a parsed graph to the DB and return { id }. */
export async function saveGraph(
  graph: ParsedGraph,
  name: string,
  format: 'onnx' | 'pytorch',
): Promise<{ id: number }> {
  return apiPost<{ id: number }>('/api/model/save', { name, format, graph_data: graph });
}

/** Fetch a previously saved graph by ID (public — no auth needed). */
export async function fetchGraph(id: number): Promise<SavedGraph> {
  return apiGet<SavedGraph>(`/api/model/${id}`);
}

// ── Param formatter ────────────────────────────────────────────────────────────

export function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return n.toString();
}
