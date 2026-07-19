# Model Visualizer — Build Log

## What we're building
Upload an ONNX or PyTorch model file → backend parses the computation graph → frontend renders it
as an interactive node-edge graph using React Flow.  Users can save graphs and share them via a
permanent URL.

---

## Phase 1: ONNX + ephemeral (no DB)

### Files created / modified

#### Backend (new)
| File | What it does |
|------|--------------|
| `backend/services/onnx_parser.py` | Loads ONNX protobuf, runs shape inference, returns nodes+edges+meta JSON |
| `backend/routers/model_viz.py` | `POST /api/model/parse` — validates file, calls parser, returns graph JSON |

#### Backend (edited)
| File | Change |
|------|--------|
| `requirements.txt` | Added `onnx>=1.16.0` |
| `backend/server.py` | Imported and registered `model_viz.router` |

#### Frontend (new)
| File | What it does |
|------|--------------|
| `src/lib/model-viz-api.ts` | TypeScript types + `parseModel()` API call |
| `src/components/model-viz/NodeCard.tsx` | Custom React Flow node: color stripe, op name, output shape, param count |
| `src/components/model-viz/InspectPanel.tsx` | Right sidebar — full node details on click (shapes, attrs, params) |
| `src/components/model-viz/GraphCanvas.tsx` | ReactFlow canvas with dagre layout, minimap, controls, invisible handles |
| `src/components/model-viz/UploadZone.tsx` | Drag-drop zone, file picker, loading/error states, example model links |
| `src/app/(protected)/model-viz/page.tsx` | Page orchestrator — upload state vs graph state, top info bar |

#### Frontend (edited)
| File | Change |
|------|--------|
| `package.json` | Added `@xyflow/react ^12.0.0` and `@dagrejs/dagre ^1.0.0` |
| `next.config.mjs` | Added `transpilePackages: ['@dagrejs/dagre', '@dagrejs/graphlib']` to fix CJS webpack error |
| `src/components/TopNavbar.tsx` | Added "Model Viz" nav link after Labs |

---

## Phase 2: PyTorch (.pt/.pth) + save/share

### New backend files
| File | What it does |
|------|--------------|
| `backend/services/pytorch_parser.py` | Spins up an E2B sandbox, uploads the .pt file, runs `torch.fx.symbolic_trace` (or named_modules fallback), returns same JSON format as onnx_parser |

### Edited backend files
| File | Change |
|------|--------|
| `backend/models.py` | Added `ModelGraph` table (id, user_id, name, format, graph_data JSON, created_at) + `User.model_graphs` relationship |
| `backend/routers/model_viz.py` | Added `POST /parse-pytorch`, `POST /save`, `GET /{id}` endpoints |

### Migration
| File | What it creates |
|------|----------------|
| `migrations/versions/j1k2l3m4n5o6_add_model_graphs.py` | `model_graphs` table + 3 indexes |

Run to apply: `alembic upgrade head`

### New frontend files
| File | What it does |
|------|--------------|
| `src/components/model-viz/InputShapeForm.tsx` | Shown after user picks .pt/.pth: collects input tensor shape (spatial dims, no batch), provides common presets, warns about first-run latency |

### Edited frontend files
| File | Change |
|------|--------|
| `src/lib/model-viz-api.ts` | Added `parsePytorchModel()`, `saveGraph()`, `fetchGraph()`, `SavedGraph` type, extra PyTorch op colors |
| `src/components/model-viz/UploadZone.tsx` | Accepts .pt/.pth; ONNX parses immediately; .pt shows InputShapeForm first; onParsed now passes format |
| `src/app/(protected)/model-viz/page.tsx` | Share button → `POST /save` → copies URL; reads `?g=ID` query param to load a shared graph on mount; Suspense wrapper for useSearchParams |

---

## How Phase 2 works end-to-end

### PyTorch path
```
User drops .pt file
  → UploadZone detects .pt extension
  → Shows InputShapeForm (e.g. "3, 224, 224")
  → User submits shape → UploadZone calls parsePytorchModel(file, [3,224,224])
  → POST /api/model/parse-pytorch  (multipart: file + input_shape form field)
      → backend validates extension + size
      → parses input_shape JSON string → [3, 224, 224]
      → calls pytorch_parser.parse_pytorch(bytes, [3,224,224])
          → creates E2B sandbox (using E2B_SANDBOX_TEMPLATE env if set)
          → writes model.pt + parse.py into sandbox
          → sandbox script:
              1. import torch (or pip install torch CPU if absent — slow on cold start)
              2. torch.load("/home/user/model.pt", map_location="cpu", weights_only=False)
              3. detect state_dict vs nn.Module
              4. STRATEGY 1: torch.fx.symbolic_trace(model)
                  - exact computation graph with actual op connections
                  - forward hooks collect per-module output shapes
                  - edges from node.args (torch.fx.Node references)
              5. STRATEGY 2 (fallback): named_modules() leaf walk
                  - works for models with dynamic control flow
                  - sequential edges between adjacent leaf modules
                  - same hook-based shape inference
              6. print(json.dumps({nodes, edges, meta}))
          → backend reads stdout, finds last JSON line, parses it
          → returns same {nodes, edges, meta} format as ONNX
  → Frontend renders with same GraphCanvas, InspectPanel, NodeCard
```

### Save / share path
```
User views a graph → clicks "Share"
  → page.tsx calls saveGraph(graph, filename, format)
  → POST /api/model/save  { name, format, graph_data }
      → inserts ModelGraph row in DB
      → returns { id: 42 }
  → frontend builds URL: /model-viz?g=42
  → copies to clipboard
  → shows "Link copied!" with the URL

Anyone visits /model-viz?g=42
  → page.tsx useEffect reads ?g= param
  → GET /api/model/42  (no auth required — public read)
  → returns { id, name, format, graph_data, created_at }
  → sets graph state → renders graph directly (no upload needed)
```

---

## Design decisions

### Color coding by layer type (extended for PyTorch)
- `#6366f1` indigo — Conv2d, ConvTranspose2d
- `#8b5cf6` purple — MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- `#ec4899` pink — BatchNorm2d, LayerNorm, GroupNorm
- `#22c55e` green — ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax (all activations)
- `#f59e0b` amber — Linear, Gemm, MatMul
- `#f97316` orange — MultiheadAttention
- `#94a3b8` slate — Flatten, Reshape, Transpose
- `#64748b` gray — everything else

### PyTorch E2B sandbox strategy
- **Why E2B?** torch is a 2 GB dependency; we don't install it on the main backend.
- **Why two strategies?** `torch.fx.symbolic_trace` is exact but fails for ~30% of real models (dynamic loops, conditional branches, non-traceable ops like `torch.nonzero`). The `named_modules` fallback always works.
- **Timeout**: 300 seconds. First run with `pip install torch` takes 2–4 min. Subsequent runs are ~10–30 s once the template has torch cached.
- **Custom template**: Set `E2B_SANDBOX_TEMPLATE` env var to a custom E2B template ID with torch pre-installed. This eliminates the cold-start delay completely.

### ModelGraph table
- `graph_data` is a JSON column (JSONB on Postgres) — the full `{nodes, edges, meta}` dict.
- No size limit enforced at DB level; typical graphs are 50–500 KB as JSON.
- GET /api/model/{id} is **public** (no auth) — the graph contains architecture info only, no personal data.
- POST /api/model/save requires auth — each graph is owned by a user.

---

## What we learned while building

### Phase 1
- **ONNX internals**: computation graph is `graph.node[]` (ops) + `graph.initializer[]` (weights as separate tensors, not inline). Connections are by tensor name, not node ID.
- **Shape inference**: `onnx.shape_inference.infer_shapes()` walks the graph symbolically to fill in all intermediate tensor shapes. Without it, you only get output tensor names, not their dimensions.
- **Parameter counting**: for each operator node, iterate its inputs; if an input name is in the initializer set, multiply its dims to get param count. Conv weight = [out_channels, in_channels, kH, kW].
- **Dagre layout**: handles DAGs with residual connections gracefully by assigning ranks. Nodes with skip connections get placed at the "later" rank (where they're consumed), creating long edge spans.
- **React Flow v12**: nodeTypes must be defined OUTSIDE the component render function (stable reference = no flicker). Edges need source/target to match real Handle IDs or React Flow defaults to node bounds.
- **CJS transpile**: `@dagrejs/dagre` and `@dagrejs/graphlib` are CommonJS modules. Next.js / webpack 5 needs `transpilePackages` in next.config.mjs or it throws `__webpack_modules__[moduleId] is not a function` at runtime.

### Phase 2
- **torch.fx vs named_modules**: `symbolic_trace` gives you the real forward-pass graph (which nodes call which), while `named_modules` gives you the module hierarchy. For most CNNs they look the same; for models with skip connections (ResNet) symbolic_trace shows the actual Add connections while named_modules shows sequential flow.
- **forward hooks for shapes**: `register_forward_hook` is the only reliable way to get output shapes from a PyTorch model without a CUDA setup or explicit shape tracking. You run one dummy forward pass and collect shapes from every module's output.
- **state_dict vs full model**: `torch.save(model.state_dict(), f)` saves only weights (a plain dict). `torch.save(model, f)` saves the full Module. Only the latter can be loaded and traced. Always check `isinstance(obj, nn.Module)` after loading.
- **Leaf modules**: `model.named_modules()` returns ALL modules including parent containers (Sequential, ResNet, etc.). Filtering to `.children() == []` (leaf nodes) gives one node per actual operation, not the container hierarchy.
- **useSearchParams in Next.js 15**: Client Components using `useSearchParams()` need a `<Suspense>` boundary wrapping them, otherwise Next.js throws a build-time error about missing Suspense.

---

## Potential issues & mitigations

| Issue | Mitigation |
|-------|-----------|
| torch not in E2B template | Script auto-runs `pip install torch --index-url .../cpu` — slow but works. Set `E2B_SANDBOX_TEMPLATE` to skip. |
| User saves a state_dict `.pt` | Sandbox detects `isinstance(obj, dict)` → returns clear error message with fix instructions |
| Custom model class not importable in sandbox | `torch.load` with `weights_only=False` needs the class definition; fails if it's in the user's local code. Error is surfaced as 422 with the exception text. |
| Very large graphs (GPT-2 = ~800 nodes) | Dagre and React Flow handle it, but pan/zoom UX degrades. Phase 3: add collapsible block groups. |
| `torch.fx.symbolic_trace` fails (dynamic ops) | Falls back to `named_modules` leaf walk automatically. The meta.method field tells the frontend which was used. |
| E2B_API_KEY not set | 503 with clear message: "E2B_API_KEY is not configured" |
| Graph too large for JSON column | Unlikely in practice; typical ResNet-50 is ~180 KB as JSON. |

---

## Commit guidance (Phase 2)

Files to stage:
```
backend/models.py                                         # ModelGraph + User.model_graphs
backend/services/pytorch_parser.py                        # new — E2B torch.fx runner
backend/routers/model_viz.py                              # new endpoints: parse-pytorch, save, GET /{id}
migrations/versions/j1k2l3m4n5o6_add_model_graphs.py     # new migration
src/lib/model-viz-api.ts                                  # parsePytorchModel, saveGraph, fetchGraph
src/components/model-viz/InputShapeForm.tsx               # new component
src/components/model-viz/UploadZone.tsx                   # .pt/.pth support
src/app/(protected)/model-viz/page.tsx                    # share button + ?g=ID loading
MODEL_VIZ_BUILD.md                                        # updated docs
```

Commit message:
```
feat: model-viz phase 2 — PyTorch parser (E2B sandbox) + save/share

- POST /api/model/parse-pytorch: accepts .pt/.pth + input_shape, runs
  torch.fx.symbolic_trace in E2B sandbox with named_modules() fallback
- POST /api/model/save + GET /api/model/{id}: persist graph to ModelGraph
  table; public read for shareable links (?g=ID)
- InputShapeForm: collect spatial dims before PyTorch parse; presets for
  ImageNet / MNIST; warns about E2B cold-start latency
- UploadZone: accepts .onnx / .pt / .pth; ONNX parses immediately, .pt
  shows shape form first
- Share button: saves graph, builds /model-viz?g=ID, copies to clipboard
- Suspense wrapper on page for useSearchParams (Next.js 15 requirement)
```

Run on backend after merge: `alembic upgrade head`
