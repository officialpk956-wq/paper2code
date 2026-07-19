# AI Labs — Implementation Report

**Date:** 2026-06-18
**Build status:** ✅ TypeScript clean · 38/38 Python tests pass · API live at /labs

---

## Files Created

### Backend

| File | Purpose |
|------|---------|
| `backend/services/lab_service.py` | Real PyTorch models + FLOPsEngine + forward hooks for all 4 labs |

### API Routes (Next.js)

| File | Endpoint | Method |
|------|----------|--------|
| `src/app/api/labs/route.ts` | GET /api/labs | Labs metadata + param schemas |
| `src/app/api/labs/transformer/route.ts` | POST /api/labs/transformer | Transformer metrics |
| `src/app/api/labs/cnn/route.ts` | POST /api/labs/cnn | CNN metrics |
| `src/app/api/labs/vit/route.ts` | POST /api/labs/vit | ViT metrics |
| `src/app/api/labs/diffusion/route.ts` | POST /api/labs/diffusion | Diffusion metrics |

### Frontend Components (`src/components/labs/`)

| File | Purpose |
|------|---------|
| `LabSelector.tsx` | Left-panel lab chooser with active state |
| `ParameterControls.tsx` | Sliders + number inputs with range labels |
| `MetricsPanel.tsx` | Right-panel live metrics display |
| `ArchitecturePreview.tsx` | Center tensor flow visualization |
| `ExperimentHistory.tsx` | localStorage experiment tracking (max 20/lab) |

### Page

| File | Route |
|------|-------|
| `src/app/labs/page.tsx` | `/labs` — main orchestrator with ThreeColumnLayout |

### Tests

| File | Coverage |
|------|---------|
| `tests/test_lab_service.py` | 38 tests across 6 test classes |

## Files Modified

| File | Change |
|------|--------|
| `src/components/layout/left-rail.tsx` | Added "AI Labs" nav item under Explore section |

---

## Architecture

### Data Flow

```
Browser → POST /api/labs/transformer {"d_model":512,"num_heads":8,...}
  └─ Next.js route (5-min in-process cache by param key)
       └─ python lab_service.py --lab transformer --d_model 512 ...
            ├─ _TransformerLab(d_model, num_heads, num_layers, vocab_size)
            │    ├─ core/blocks_transformer.py  TransformerEncoderBlock (real module)
            │    └─ register_forward_hook → captures real input/output shapes
            ├─ count_parameters(model)         from core/param_counter.py
            ├─ FLOPsEngine.estimate()          from core/rag/flops_engine.py
            └─ GPU_SPECS latency formula       from core/implementation/cost_estimator.py
                 ↓ JSON to stdout → cached → NextResponse.json
```

### Critical Constraint Met: No Fake Data

Every metric originates from real code:

| Metric | Source |
|--------|--------|
| `params_M` | `count_parameters(model)` from `core/param_counter.py` |
| `total_flops_mflops` | Sum of `FLOPsEngine.estimate()` per node |
| `memory_mb` | `FLOPsEngine.estimate().memory_mb` per node |
| `latency_ms` | `total_flops / (gpu_tflops × efficiency × 1e3)` using `GPU_SPECS["RTX4090"]` |
| `input_shape` / `output_shape` | `register_forward_hook` on real PyTorch forward pass |
| `formula` | Returned by `FLOPsEngine.estimate().formula` |
| `severity` | `FLOPsEngine._severity(flops, memory_mb)` |

---

## Labs

### 1. Transformer Lab

**Model:** `_TransformerLab` — encoder-only using `TransformerEncoderBlock` from `core/blocks_transformer.py`

**Controls:**
- `d_model`: 64 – 1024 (must be divisible by `num_heads`)
- `num_heads`: 1 – 16
- `num_layers`: 1 – 24
- `seq_len`: 16 – 512 (capped at pos_embed size of 512)
- `vocab_size`: 1000 – 50000

**Extra metrics:** `head_dim`, `token_count`, `attention_cost_mflops`, `attention_note` (O(N²·D) formula)

**Validation:** returns `{"error": "..."}` if `d_model % num_heads != 0`

### 2. CNN Lab

**Model:** `_CNNLab` — VGG-style with configurable depth/width/kernel, MaxPool every 2 layers

**Controls:**
- `base_channels`: 8 – 256
- `depth`: 1 – 8
- `kernel_size`: 1 – 7 (auto-adjusted to odd)
- `image_resolution`: 32 – 512

**Extra metrics:** `receptive_field` (estimated as `k + (k-1)×(d-1)`), `feature_maps` (spatial + channel per conv layer)

### 3. ViT Lab

**Model:** `_ViTLab` — patch embedding + `TransformerEncoderBlock` stack

**Controls:**
- `image_size`: 32 – 512
- `patch_size`: 4 – 32 (auto-corrected to largest divisor of `image_size` if incompatible)
- `hidden_dim`: 64 – 1024
- `num_blocks`: 1 – 24

**Extra metrics:** `token_count`, `num_patches`, `num_heads` (auto-derived), `head_dim`, `attention_cost_mflops`

**Auto-head selection:** `_find_valid_heads(hidden_dim)` — returns largest divisor of `hidden_dim` ≤ 16

### 4. Diffusion Lab

**Model:** `DDPMBuilder` from `core/ddpm_builder.py`

**Controls:**
- `latent_size`: 8 – 128 (spatial resolution)
- `channels`: 1 – 4
- `diffusion_steps`: 100 – 2000

**Special handling:** `DDPMBuilder.forward(x, time)` takes TWO arguments — uses `_capture_shapes_ddpm` which passes `(dummy_x, dummy_t)` separately

**Extra metrics:** `step_flops_mflops`, `inference_flops_mflops` (= step × steps), `training_flops_per_iter_mflops`, `training_note`

---

## Frontend

### Three-Column Layout

```
Left (w-72)           Center (flex-1)        Right (w-80)
─────────────         ──────────────         ────────────
LabSelector           ArchitecturePreview    MetricsPanel
ParameterControls     (tensor flow nodes)    (params, FLOPs,
ExperimentHistory                             memory, latency,
                                              lab-specific extras)
```

### Debounced Parameter Updates

- Parameters debounce 800ms before triggering POST
- Loading spinner in center panel during request
- Experiment auto-saved to localStorage on each successful response

### Experiment History

- Per-lab history (localStorage key: `paper2code:lab:experiments`)
- Max 20 experiments per storage key (shared across all labs, most-recent-first)
- Each entry shows: params snapshot, params_M, FLOPs, memory_mb, latency_ms, relative timestamp
- Most recent entry highlighted with accent border

---

## API Response Format

### GET /api/labs

```json
{
  "labs": [
    {
      "id": "transformer",
      "name": "Transformer Lab",
      "description": "...",
      "icon": "⚡",
      "params": [
        { "key": "d_model", "label": "Model Dimension", "min": 64, "max": 1024, "step": 64, "default": 512 }
      ],
      "endpoint": "/api/labs/transformer"
    }
  ]
}
```

### POST /api/labs/transformer (example response)

```json
{
  "lab": "transformer",
  "config": { "d_model": 512, "num_heads": 8, "num_layers": 6, "seq_len": 128, "vocab_size": 10000 },
  "params_M": 65.1234,
  "total_flops_mflops": 2450.32,
  "memory_mb": 18.4,
  "latency_ms": 0.0988,
  "token_count": 128,
  "head_dim": 64,
  "attention_cost_mflops": 310.5,
  "attention_note": "O(N²·D) per layer: 128²×512=8388608 ops × 6 layers = 50 MFLOPs",
  "flow_steps": [
    {
      "id": "embedding",
      "name": "Token Embedding",
      "type": "Embedding",
      "input_shape": [1, 128],
      "output_shape": [1, 128, 512],
      "flops_mflops": 0.0,
      "params_M": 5.12,
      "memory_mb": 0.25,
      "formula": "...",
      "severity": "info"
    }
  ]
}
```

---

## Bug Fixed During Implementation

### Latency Formula Unit Error

**Before:** `ms = total_flops_mflops / (tflops * efficiency * 1e6)` → units off by 1000×, giving µs instead of ms

**Derivation:**
```
latency_s = (total_flops_mflops × 10^6) / (gpu_tflops × 10^12 × eff)
latency_ms = latency_s × 1000
           = total_flops_mflops / (gpu_tflops × eff × 10^3)
```

**After:** `ms = total_flops_mflops / (tflops * efficiency * 1e3)` ✓

**Verified:** ResNet-50 (8178 MFLOPs) on RTX4090 @ 30% = **0.33 ms** (matches published benchmarks ~0.3-0.5ms for batch=1)

---

## Test Coverage

```
tests/test_lab_service.py — 38/38 PASS (10.5s)

TestTransformerLab       (11) — keys, params, FLOPs, scaling, head validation, latency, flow, notes
TestCNNLab                (7) — keys, params, FLOPs, depth/res scaling, receptive field, latency
TestViTLab                (8) — keys, token math, patch scaling, head derivation, note, autocorrect
TestDiffusionLab          (6) — keys, inference=step×T, step scaling, params, note, flow
TestLatencyEstimation     (3) — zero FLOPs, proportionality, ResNet-50 sanity
TestFlowSteps             (3) — required fields, null shapes, severity values
```

---

## Performance

- First request (cold): ~3–8s (PyTorch model build + forward pass)
- Subsequent requests with same params: <5ms (5-min in-process cache keyed by param hash)
- Debounce: 800ms after last param change → no API spam during slider drag
