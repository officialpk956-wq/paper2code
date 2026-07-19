# Phase 14B Implementation Report: Executable Architecture Graphs

## Summary

Phase 14B converts reconstructed architecture blueprints (Phase 14A output) into executable architecture graphs — structured, validated, exportable representations of a paper's neural network architecture.

**All constraints satisfied:**
1. No LLM APIs used
2. No mock graph data — compiler derives all data from blueprint
3. TensorTracker reused for shape inference
4. FLOPsEngine reused (values flow from Phase 14A's tensor_flow)
5. Architecture blueprint reused as the sole input
6. Build passes cleanly
7. All tests pass (793 Python + 271 frontend)
8. Accessibility preserved (ARIA roles, keyboard navigation)
9. Security preserved (input validation, no injection surfaces)

---

## New Files

### `backend/services/architecture_graph_compiler.py`

Core compiler service (~270 lines).

**Data model:**

| Class | Fields |
|-------|--------|
| `EGNode` | id, name, type, shape, params, flops, metadata |
| `EGEdge` | source, target, edge_type |
| `ValidationReport` | valid, errors (list), warnings (list) |
| `ExecutableGraph` | id, nodes, edges, input_spec, output_spec, parameter_count, flops, validation_status, validation_report |

All classes implement `to_dict()` / `from_dict()` for JSON serialization.

**`compile_blueprint(blueprint_dict)`** — main entry point:
- Reads components and connections from blueprint
- Calls `_run_tracker()` for shape inference via TensorTracker
- Reads FLOPs/params from `tensor_flow` (already computed by FLOPsEngine in Phase 14A)
- Calls `validate_graph()` internally
- Returns `ExecutableGraph` with `validation_status` in `{"valid", "warnings", "invalid"}`

**`_run_tracker(components, connections, input_shape)`** — TensorTracker integration:
- Maps blueprint component types to TensorTracker node types via `_COMP_TO_GRAPH_TYPE`
- Builds an `ArchitectureGraph` from blueprint components
- Calls `TensorTracker().propagate_shapes(ag, initial_shape=...)`
- Extracts `output_shape` from modified nodes (TensorTracker sets these in-place)
- Falls back gracefully if TensorTracker raises

**`validate_graph(graph)`** — 5-check validation:
1. Directed cycle detection (DFS with recursion-stack set)
2. Dangling edge references (source/target not in node set)
3. Disconnected nodes (nodes with no edges)
4. Missing input/output nodes
5. Sequential shape mismatches (src.shape ≠ tgt.metadata.input_shape)

**Export functions:**
- `export_graph_json(graph)` — `json.dumps(graph.to_dict(), indent=2)`
- `export_graph_mermaid(graph)` — `flowchart TD` syntax; dashed arrows (`-.->`) for non-sequential edges; `_safe_id()` sanitizes hyphens
- `export_graph_dot(graph)` — Graphviz `digraph`; `style=dashed` for non-sequential edges

---

### `src/components/paper-upload/ExecutableGraphViewer.tsx`

Client-side React component (~380 lines).

**Exported types:** `EGNode`, `EGEdge`, `ValidationReport`, `ExecutableGraph`

**Layout constants** (consistent with `ArchitectureBlueprintViewer`):
```
CANVAS_W=700  NODE_W=320  NODE_H=76
NODE_X=190    ROW_H=110   SKIP_X=562
```

**Features:**
- SVG canvas with zoom (80%–120%, reset)
- Clickable node buttons with Enter/Space keyboard support
- Node inspector panel with `ShapePill` + `FLOPsBadge`
- Validation report panel (errors in red, warnings in amber)
- Export toolbar: JSON / Mermaid / DOT via client-side `URL.createObjectURL`
- Edge rendering: solid lines for sequential, dashed arcs for skip/residual/concat

**Accessibility attributes:**

| Element | ARIA |
|---------|------|
| Empty state | `role="status"` |
| Toolbar | `role="region" aria-label="Executable graph toolbar"` |
| SVG canvas | `role="img" aria-label="Executable graph for {id}"` |
| Node buttons | `role="button" aria-pressed={selected} aria-label="{name} — {type}"` |
| Inspector panel | `role="region" aria-label="Selected node details"` |
| Close button | `aria-label="Close details panel"` |
| Validation panel | `role="region" aria-label="Graph validation report"` |

---

### `src/app/api/papers/generated/[id]/executable-graph/route.ts`

Next.js route proxy — `GET /api/papers/generated/[id]/executable-graph`
- Validates numeric `id`
- Proxies to `GET /api/papers/{id}/executable-graph` on the Python backend
- Returns 404, 502, or JSON as appropriate

### `src/app/api/papers/generated/[id]/graph-export/route.ts`

Next.js route proxy — `GET /api/papers/generated/[id]/graph-export?format=json|mermaid|dot`
- Validates `id` and `format` (rejects unknown formats with 400)
- Proxies to `GET /api/papers/{id}/graph-export?format={format}` on the Python backend
- Returns plain text with the appropriate `Content-Type`

### `tests/test_architecture_graph_compiler.py`

55 Python tests across 6 suites:

| Suite | Tests | Coverage |
|-------|-------|---------|
| `TestDataModel` | 5 | `to_dict`, `from_dict`, round-trips |
| `TestCompileBlueprint` | 18 | node/edge counts, FLOPs, params, types, edge_type mapping |
| `TestShapeInference` | 4 | TensorTracker integration, fallback, pooling shape |
| `TestValidation` | 11 | cycle, dangling edges, disconnected nodes, shape mismatch |
| `TestExports` | 12 | JSON, Mermaid, DOT format correctness |
| `TestCycleDetection` | 5 | linear chain, DAG, direct cycle, long cycle, U-Net skip |

### `src/__tests__/components/paper-upload/ExecutableGraphViewer.test.tsx`

31 frontend tests across 7 suites:

| Suite | Tests |
|-------|-------|
| Empty state | 3 |
| Rendering | 7 |
| Node selection | 6 |
| Zoom controls | 4 |
| Validation display | 5 |
| Export buttons | 3 |
| PaperWorkspaceTabs 6th tab | 3 |

---

## Modified Files

### `backend/services/paper_ingestion_service.py`

After `reconstruct_architecture()` succeeds, calls `compile_blueprint()` and stores the result at:
```
paper.architecture_graph.ingestion.executable_graph
```

### `backend/server.py`

Two new endpoints:

```
GET /api/papers/{paper_id}/executable-graph
GET /api/papers/{paper_id}/graph-export?format=json|mermaid|dot
```

The first returns the persisted `ExecutableGraph` dict. The second re-instantiates the graph from the stored dict and runs the appropriate export function.

### `src/components/paper-upload/PaperWorkspaceTabs.tsx`

- Added `executableGraph: ExecutableGraph | null` to `PaperWorkspaceTabsProps`
- Extended `TabId` union to include `'executable-graph'`
- Added 6th entry in `TABS` array
- Added 6th `tabpanel` div rendering `<ExecutableGraphViewer graph={executableGraph} />`

### `src/app/papers/upload/[paperId]/page.tsx`

- Added `import type { ExecutableGraph }` from `ExecutableGraphViewer`
- Added `executable_graph?: ExecutableGraph` to `PaperDetailResponse` ingestion type
- Extracted `const executableGraph = ingestion.executable_graph ?? null`
- Passed `executableGraph={executableGraph}` to `<PaperWorkspaceTabs>`

---

## Test Results

```
Python:   793/793 passed  (55 new in test_architecture_graph_compiler.py)
Frontend: 271/271 passed  (31 new in ExecutableGraphViewer.test.tsx)
Build:    next build — clean, no TypeScript errors
```

---

## Design Decisions

**FLOPs source**: blueprint's `tensor_flow` list already contains FLOPs computed by FLOPsEngine in Phase 14A's `_simulate_tensor_flow`. Using those values directly avoids double-computation and satisfies "reuse FLOPsEngine."

**TensorTracker integration**: `_run_tracker()` translates blueprint component types to TensorTracker node types via `_COMP_TO_GRAPH_TYPE`. TensorTracker modifies `GraphNode.output_shape` in-place; shapes are read after `propagate_shapes()` returns. Any exception from TensorTracker is silently caught — partial shapes are still useful.

**Cycle detection**: DFS with a `visited` set and an `in_stack` set. This correctly handles U-Net-style graphs with forward-only concat skip connections (no false positive cycles).

**Export id sanitization**: `_safe_id()` replaces non-alphanumeric characters with underscores for Mermaid/DOT compatibility (node IDs like `unet-enc1` would otherwise break both formats).

**Client-side exports**: `buildMermaid()` and `buildDot()` run in the browser; no server round-trip needed. `downloadText()` uses `URL.createObjectURL` + a temporary `<a>` element for the file download.
