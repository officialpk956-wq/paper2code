// ── Types matching the backend onnx_parser / pytorch_parser response ───────────

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
};

export type ParsedEdge = {
  id: string;
  source: string;
  target: string;
  shape: number[];
  tensor: string;
};

export type GraphMeta = {
  total_nodes: number;
  total_params: number;
  total_edges: number;
  graph_inputs: Record<string, number[]>;
  graph_outputs: Record<string, number[]>;
  ir_version: number;
  opset_version: number;
  method?: string;   // "symbolic_trace" | "named_modules" — pytorch only
};

export type ParsedGraph = {
  nodes: ParsedNode[];
  edges: ParsedEdge[];
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

// ── HTTP helpers ──────────────────────────────────────────────────────────────

const BASE =
  (typeof process !== 'undefined' && process.env.NEXT_PUBLIC_API_URL) ||
  'http://127.0.0.1:8000';

function getAuthHeader(): Record<string, string> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function throwOnError(res: Response): Promise<void> {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = Array.isArray(err.detail)
      ? err.detail.map((e: { msg?: string }) => e?.msg ?? JSON.stringify(e)).join('; ')
      : err.detail;
    throw new Error(detail ?? 'Request failed');
  }
}

// ── API calls ─────────────────────────────────────────────────────────────────

/** Parse an .onnx file. */
export async function parseModel(file: File): Promise<ParsedGraph> {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${BASE}/api/model/parse`, {
    method: 'POST',
    headers: getAuthHeader(),
    body: form,
  });
  await throwOnError(res);
  return res.json() as Promise<ParsedGraph>;
}

/** Parse a .pt / .pth file via the E2B sandbox. */
export async function parsePytorchModel(
  file: File,
  inputShape: number[],
): Promise<ParsedGraph> {
  const form = new FormData();
  form.append('file', file);
  form.append('input_shape', JSON.stringify(inputShape));

  const res = await fetch(`${BASE}/api/model/parse-pytorch`, {
    method: 'POST',
    headers: getAuthHeader(),
    body: form,
  });
  await throwOnError(res);
  return res.json() as Promise<ParsedGraph>;
}

/** Save a parsed graph to the DB and return { id }. */
export async function saveGraph(
  graph: ParsedGraph,
  name: string,
  format: 'onnx' | 'pytorch',
): Promise<{ id: number }> {
  const res = await fetch(`${BASE}/api/model/save`, {
    method: 'POST',
    headers: { ...getAuthHeader(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, format, graph_data: graph }),
  });
  await throwOnError(res);
  return res.json() as Promise<{ id: number }>;
}

/** Fetch a previously saved graph by ID (public — no auth needed). */
export async function fetchGraph(id: number): Promise<SavedGraph> {
  const res = await fetch(`${BASE}/api/model/${id}`);
  await throwOnError(res);
  return res.json() as Promise<SavedGraph>;
}

// ── Param formatter ────────────────────────────────────────────────────────────

export function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return n.toString();
}
