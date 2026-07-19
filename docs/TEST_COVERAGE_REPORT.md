# Test Coverage Report — Phase 11B

**Date:** 2026-06-18  
**Status:** Backend 603/603 pass; frontend coverage measured by file inspection

---

## 1. Backend Python Tests

### Total: 603 tests — all pass

| Test File | Tests | Coverage Area |
|-----------|-------|--------------|
| `tests/test_lab_service.py` | 38 | AI Labs — all 4 models, latency, caching |
| `tests/test_block_viz.py` | 34 | Block hierarchy, forward pass, PyTorch hooks |
| `tests/test_phase10_impl.py` | ~45 | Training cost estimation, GPU specs |
| `tests/test_architecture_builder.py` | ~60 | ResNet, ViT, UNet, Transformer builders |
| `tests/test_flops_engine.py` | ~40 | FLOPs estimation, severity, memory |
| `tests/test_tensor_tracker.py` | ~30 | Hook registration, shape capture |
| `tests/test_content_graph.py` | ~25 | Content slug validation, relationship graph |
| `tests/test_dojo_runner.py` | ~20 | Python code execution, test case evaluation |
| Other test files | ~311 | Remaining coverage |

### Backend Coverage Estimate: ~85%

Key covered paths:
- All PyTorch model forward passes (Transformer, CNN, ViT, DDPM)
- FLOPs calculation for all layer types
- Shape capture via register_forward_hook
- Latency estimation formula
- Architecture builder parameter validation

Key gaps:
- Edge cases in `flops_engine.py` for unusual layer types
- Error handling paths in `tensor_tracker.py`

---

## 2. Frontend Test Coverage

No frontend unit/integration tests exist currently. This is the most significant coverage gap.

### Missing Tests — High Priority

#### AI Labs (Feature 3)
| Component | What to Test |
|-----------|-------------|
| `ParameterControls` | Range input updates state; number input validates bounds; disabled state |
| `MetricsPanel` | Renders loading state; renders metrics; renders error; labId-specific sections |
| `LabSelector` | Active state; onSelect callback; lab icons rendered |
| `ArchitecturePreview` | Empty state; flow steps render; severity colors |
| `ExperimentHistory` | localStorage read/write; clear function; max 20 entries |
| Labs page | Debounce behavior; lab switch clears metrics; API error handling |

#### Block Visualization (Feature 4)
| Component | What to Test |
|-----------|-------------|
| `BlockBox` | Expand/collapse; selected state; aria-expanded |
| `BlockGraph` | Stage grouping; onBlockSelect callback |
| `ForwardPassPlayer` | Play/pause; step scrubber; speed selector; aria attributes |

#### DS Coding Dojo (Feature 2)
| Component | What to Test |
|-----------|-------------|
| `DojoProblemPage` | Tab switching; run results display; submit status; localStorage persistence |
| `DojoEditor` | Code updates; run/submit button states; Monaco mount |
| `TestResultPanel` | Pass/fail display; error display |

#### API Route Tests (Next.js)
| Route | What to Test |
|-------|-------------|
| `POST /api/labs/transformer` | Valid params; out-of-range clamping; invalid JSON; timeout simulation |
| `POST /api/dojo/run` | Valid code; invalid functionName; missing testCases |
| `POST /api/dojo/submit` | Code too long; invalid identifier rejected |
| `GET /api/papers/[id]/block-hierarchy` | Valid arch; unknown arch returns 404 |

---

## 3. Recommended Testing Tools

| Layer | Recommended Tool | Why |
|-------|-----------------|-----|
| React components | `@testing-library/react` + `vitest` | Fast, DOM-based, accessible queries |
| API routes | `supertest` + `jest` or route handler unit tests | Direct handler invocation |
| E2E | `playwright` | Full flow (open problem → write code → run → submit) |

---

## 4. Coverage Targets

| Layer | Current | Target (Phase 12) |
|-------|---------|-------------------|
| Backend Python | ~85% | 90% |
| Frontend React components | ~0% | 70% |
| API routes (Next.js) | ~0% | 80% |
| E2E flows | 0 | 3 critical flows |

---

## 5. Summary

| Category | Status |
|----------|--------|
| Backend Python tests | 603/603 ✅ |
| Frontend component tests | 0 — gap |
| API route tests | 0 — gap |
| E2E tests | 0 — gap |

Highest-priority gap: frontend component tests for the three new features (Labs, Block Viz, Dojo). Recommend adding `@testing-library/react` + `vitest` in Phase 12.
