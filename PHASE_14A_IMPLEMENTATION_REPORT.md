# Phase 14A — Automatic Architecture Reconstruction Engine

## Summary

Phase 14A extends the paper ingestion pipeline with a deterministic architecture reconstruction engine. When a PDF is uploaded, the system extracts numeric signals from the text, identifies the primary architecture via the knowledge graph (Phase 13B), selects the matching template, simulates tensor flow, estimates FLOPs and parameters using FLOPsEngine, and persists the resulting blueprint. A new "Architecture Blueprint" tab in the paper workspace renders an interactive SVG flow diagram.

---

## Deliverables

### Backend

| File | Role |
|------|------|
| `backend/services/architecture_reconstruction_service.py` | Core service: signal extraction, template library, blueprint generation, FLOPs simulation |
| `backend/services/paper_ingestion_service.py` | Updated: calls `reconstruct_architecture()` during ingestion |
| `backend/server.py` | New endpoint: `GET /api/papers/{id}/blueprint` |

### Frontend

| File | Role |
|------|------|
| `src/components/paper-upload/ArchitectureBlueprintViewer.tsx` | SVG viewer: layer flow, skip connections, zoom/pan, node selection, confidence bar |
| `src/components/paper-upload/PaperWorkspaceTabs.tsx` | Updated: 5th tab "Architecture Blueprint" |
| `src/app/papers/upload/[paperId]/page.tsx` | Updated: passes `architecture_blueprint` to workspace tabs |
| `src/app/api/papers/generated/[id]/blueprint/route.ts` | Next.js proxy to FastAPI endpoint |

### Tests

| File | Count |
|------|-------|
| `tests/test_architecture_reconstruction.py` | 68 Python tests |
| `src/__tests__/components/paper-upload/ArchitectureBlueprintViewer.test.tsx` | 26 frontend tests |

---

## Architecture

### Signal Extraction (no LLM)

`_extract_signals()` scans the combined title + abstract + first 6 sections text for numeric architecture parameters using regex patterns:

| Signal | Pattern examples | Detected value |
|--------|-----------------|----------------|
| `depth` | "50-layer", "L = 12" | Layer/block count |
| `num_heads` | "12 attention heads" | Attention head count |
| `d_model` | "d_model = 512", "hidden_dim = 768" | Hidden dimension |
| `ffn_dim` | "ffn_dim = 2048", "intermediate_size = 3072" | FFN dimension |
| `patch_size` | "patch_size = 16" | ViT patch size |
| `image_size` | "224×224 image" | Input resolution |
| `channels` | "64 channels" | Channel count |
| `num_classes` | "1000 classes" | Output class count |
| `vocab_size` | "vocab = 30522" | Token vocabulary |
| `seq_len` | "max sequence length = 512" | Sequence length |

### Template Library

7 complete architecture templates, each parameterized by extracted signals:

| Template | Architectures | Default dims |
|----------|---------------|-------------|
| `_resnet_template` | ResNet 18/34/50/101/152, VGG, AlexNet, DenseNet, EfficientNet | depth=50, 1000 classes |
| `_vit_template` | ViT, Swin, CLIP, DINO | embed=768, heads=12, layers=12, patch=16 |
| `_unet_template` | U-Net, FCN | 64 base channels, 2 output classes |
| `_transformer_template` | Transformer, BERT, GPT, T5, LLaMA, Seq2Seq, MoE | embed=512, heads=8, layers=6 |
| `_gan_template` | GAN | latent=100, 64×64 output |
| `_vae_template` | VAE, Autoencoder | latent=128, 64×64 |
| `_diffusion_template` | DDPM, Stable Diffusion, Latent Diffusion | U-Net denoiser, 64×64 |
| `_lstm_template` | LSTM, GRU, RNN | hidden=256, 2 layers |
| `_generic_template` | Unknown/fallback | embed=256 |

### FLOPs Estimation

Reuses `FLOPsEngine` (Phase 12 / core/rag/flops_engine.py). Component types map to engine dispatch methods:

| Component type | FLOPs method | Notes |
|----------------|-------------|-------|
| `conv` | `conv2d` or `residualblock` | Stages use `residualblock` via `metadata.flops_type` |
| `attention` | `multiheadattention` + `feedforward` | Encoder blocks estimate both MHSA and FFN per layer |
| `embedding` | `patchembedding` or `token_embedding` | Dispatched via `metadata.flops_type` |
| `pooling` | (none) | 0 FLOPs |
| `mlp` | `feedforward` | LSTM body uses this |
| `normalization` | `layernorm` | |
| `upsample` | `upsample` | Bilinear: 4 × C × H × W |
| `head` | `linear` | |

For components with `repeat_count`, FLOPs × params are multiplied by the repeat count.

### Tensor Flow Simulation

`_simulate_tensor_flow()` iterates components in declaration order, tracking sequential predecessor shapes. The first component receives the blueprint's `input_shape` ([1, 3, 224, 224] for vision, etc.). Each step records `{component_id, input_shape, output_shape, flops_mflops, params_M}`.

### Confidence Scoring

Evidence-weighted float in [0.0, 1.0]:

| Evidence | Weight |
|----------|--------|
| Architecture in title (introduces edge) | +0.35 |
| Architecture in body (uses edge) | +0.15 |
| Methods/Approach section detected | +0.20 |
| Each extracted signal (depth, dim, heads, …) | +0.05 each, max +0.25 |
| ≥3 equations | +0.10 |
| 1–2 equations | +0.05 |

### Persistence

Blueprint stored inside the existing `Paper.architecture_graph` JSON column, no migration needed:

```
paper.architecture_graph.ingestion.architecture_blueprint = { id, name, components, connections, ... }
```

### Connection Types

| Type | Visual | Meaning |
|------|--------|---------|
| `sequential` | Solid vertical arrow | Main data flow |
| `residual` | Dashed curved arrow (right) | Skip/residual connection |
| `concat` | Solid curved arrow (right) | Concat-skip (U-Net style) |
| `cross_attention` | Dotted curved arrow (right) | Encoder→Decoder cross-attn |

---

## Component: ArchitectureBlueprintViewer

- SVG canvas with `viewBox="0 0 700 {n×110+40}"` — grows with component count
- Grid background pattern
- Pan: `onMouseDown/Move/Up` drag on the container div
- Zoom: buttons ±0.2 per click, range 0.4–3.0, shown as percentage
- Node selection: click to select/deselect, `Enter`/`Space` keyboard support
- Info panel: shows input/output shapes, FLOPs, params for selected node
- Confidence meter: colored progress bar + percentage
- Architecture link: "Open Architecture →" button linking to `/architectures/{slug}`
- Connection legend (sequential / residual / concat / cross-attention)
- Empty state: "No architecture blueprint available"

**Accessibility**: `role="img"` on SVG, `role="button"` on each node group, `aria-pressed`, `aria-label`, `role="region"` on info panel, `aria-label` on confidence group.

---

## Test Results

```
Frontend:  240 / 240  (23 test files)
Python:    738 / 738
Build:     ✓  Next.js production build passes
```

Phase 14A new tests: 68 Python + 26 frontend = 94 tests

---

## Constraints Satisfied

1. No LLM APIs — all reconstruction is regex + template-based
2. No mock blueprint data — blueprint built from real extracted signals + KG
3. Deterministic reconstruction — same input always produces the same blueprint
4. Reuses TensorTracker patterns (sequential shape propagation)
5. Reuses FLOPsEngine — all per-component estimates use the existing engine
6. Reuses graph visualizer patterns — same SVG pan/zoom/select as PaperKnowledgeGraph
7. Build passes — zero TypeScript errors
8. Tests pass — 94 new tests, all green
9. Accessibility preserved — ARIA roles, keyboard navigation, screen-reader labels
10. Security preserved — no new input boundaries, no eval, no injection vectors
