# Paper2Code Content Backlog
**Generated:** 2026-06-21 | **Phase:** Content Validation Audit

Ranked missing work by priority. **This is what to build next — in order.**

Every item is validated to be missing from the current codebase. No guesswork.

**Priority definitions:**
- **P0** — Platform-broken: wrong/missing data causing incorrect behavior right now
- **P1** — High-value: unlocks a major feature or completes a critical learning path
- **P2** — Medium-value: improves coverage significantly, but platform works without it
- **P3** — Nice-to-have: fills gaps in completeness, low urgency

---

## P0 — MUST FIX (Platform Bugs)

These cause incorrect platform behavior and should be fixed before any new content is written.

### P0-001: LLaMA Architecture Catalog Status Bug
- **Name:** Fix `llama` catalog entry status
- **Type:** Data fix (1 line)
- **Why it matters:** LLaMA architecture MDX (`src/content/architectures/llama/content.mdx`, 167 lines, full 15-section content) is fully authored. But `src/data/architecture-catalog.ts` has `status: "coming-soon"` for the `llama` entry, causing Architecture Explorer to display it as unavailable. Users cannot access complete content.
- **Effort:** 1 line change
- **Files:** `src/data/architecture-catalog.ts` — change `status: "coming-soon"` to `status: "complete"` for the llama entry
- **Dependencies:** None
- **Verified:** `CONTENT_DUPLICATES.md` group 9, `CONTENT_GAP_REPORT.md` GAP-003

---

### P0-002: VGG Catalog-MDX Slug Mismatch
- **Name:** Resolve `vgg` catalog stub vs `vgg16`/`vgg19` MDX
- **Type:** Data fix
- **Why it matters:** Architecture Explorer has a `vgg` slug in the catalog that has no matching MDX. Actual MDX content lives at `vgg16` and `vgg19`. Users clicking VGG in Explorer hit a dead end.
- **Effort:** 2–4 hours (either rename MDX dirs or update catalog to point to vgg16)
- **Files:** `src/data/architecture-catalog.ts` (update vgg stub to redirect to vgg16) OR rename `src/content/architectures/vgg16/` to `src/content/architectures/vgg/` and consolidate
- **Dependencies:** None
- **Verified:** `CONTENT_DUPLICATES.md` group 5, `CONTENT_GAP_REPORT.md` GAP-004

---

### P0-003: 9 Domain Pages Show Placeholder Content
- **Name:** Author 9 missing domain data files
- **Type:** Content authoring (TypeScript data files)
- **Why it matters:** `/learn/computer-vision`, `/learn/nlp`, `/learn/reinforcement-learning`, and 6 other domain pages show a generic scaffold with no real content. These are major entry points — a user browsing Computer Vision or NLP sees an empty template.
- **Effort:** 2–4 hours per domain × 9 domains = 18–36 hours total
- **Files to create:**
  - `src/data/domains/computer-vision.ts`
  - `src/data/domains/nlp.ts`
  - `src/data/domains/reinforcement-learning.ts`
  - `src/data/domains/statistics.ts`
  - `src/data/domains/mathematics.ts`
  - `src/data/domains/mlops.ts`
  - `src/data/domains/ai-systems.ts`
  - `src/data/domains/research-methods.ts`
  - `src/data/domains/robotics.ts`
- **Dependencies:** Existing domain data pattern in `src/data/domains/deep-learning.ts`
- **Verified:** `CONTENT_GAP_REPORT.md` GAP-002, `INVENTORY_VALIDATION_REPORT.md` Section 7B

---

### P0-004: 5 Problem MDX Deep-Dives Are 7-Line Stubs
- **Name:** Fill problem MDX deep-dives for 5 problems
- **Type:** Content authoring (MDX)
- **Why it matters:** 5 of 8 problem MDX files are frontmatter only (7 lines each with no body content). Users clicking on a problem expecting a detailed explanation get an empty page.
- **Effort:** 2–4 hours per problem × 5 problems = 10–20 hours
- **Files to fill:**
  - `src/content/problems/vit-patch-size/content.mdx` (7 lines → ~40+)
  - `src/content/problems/stable-diffusion-cfg/content.mdx` (7 lines → ~40+)
  - `src/content/problems/moe-routing/content.mdx` (7 lines → ~40+)
  - `src/content/problems/llama-rope/content.mdx` (7 lines → ~40+)
  - `src/content/problems/clip-batch-size/content.mdx` (7 lines → ~40+)
- **Dependencies:** Knowledge of each topic (ViT patches, CFG, MoE, RoPE, CLIP)
- **Verified:** `INVENTORY_VALIDATION_REPORT.md` Section 5C

---

### P0-005: Knowledge Graph Metadata Discrepancy
- **Name:** Fix KNOWLEDGE_GRAPH.json metadata counts
- **Type:** Data fix
- **Why it matters:** `KNOWLEDGE_GRAPH.json` claims 95 nodes and 187 edges. Actual count is 94 nodes and 164 edges. Any tooling that reads the metadata field gets wrong numbers.
- **Effort:** 30 minutes
- **Files:** `KNOWLEDGE_GRAPH.json` — update `"total_nodes": 95` to `94` and `"total_edges": 187` to `164`
- **Dependencies:** None
- **Verified:** `KNOWLEDGE_GRAPH_AUDIT.md` Section 1

---

## P1 — HIGH PRIORITY (Unlocks Major Features)

### P1-001: InstructGPT / RLHF Paper MDX
- **Name:** Add `instructgpt` paper to paper library
- **Type:** Content authoring (paper MDX, 15 sections)
- **Why it matters:** RLHF is a P1 topic (listed in COURSE_GENERATION_MAP.md), identified as 10% covered. There is no RLHF paper in the 19-paper library despite the platform having an RLHF concept node and a chatgpt-system-design case that references it. This is the foundational alignment paper.
- **Effort:** 3–6 hours (15-section paper MDX following the standard format)
- **Files to create:** `src/content/papers/instructgpt/content.mdx`
- **Dependencies:** None
- **Verified:** `PAPER_COVERAGE_MATRIX.md` (missing papers table), `CONTENT_COVERAGE_MATRIX.md` rlhf row

---

### P1-002: Flash Attention Paper MDX
- **Name:** Add `flash-attention` paper to paper library
- **Type:** Content authoring (paper MDX)
- **Why it matters:** Flash Attention is referenced by a dead link in the platform (confirmed in Phase 16 reports). It enables the KV cache discussion and is a core inference optimization technique. The KG has a `flash-attention` concept node with no paper pointing to it.
- **Effort:** 3–6 hours
- **Files to create:** `src/content/papers/flash-attention/content.mdx`
- **Dependencies:** None
- **Verified:** `CONTENT_GAP_REPORT.md` Section 3 (papers missing), `KNOWLEDGE_GRAPH_AUDIT.md` Section 7B

---

### P1-003: Top 10 Missing Topic Pages
- **Name:** Author top 10 topic data files
- **Type:** Content authoring (TypeScript data files with 13 sections each)
- **Why it matters:** Only 1 of ~100 topic pages is authored. The `/learn/[domain]/[topic]` route returns "Topic Not Found" for all 17 specified topics in COURSE_GENERATION_MAP.md. The Learn → Practice loop works only for `attention`. These 10 have supporting content already in the codebase (architecture MDX, paper MDX, problem slugs) that can be referenced.
- **Effort:** 4–8 hours per topic × 10 topics = 40–80 hours
- **Priority sub-order within P1 (most supporting content first):**
  1. `transformers` — full arch MDX + paper + 510-line implementation
  2. `convolutional-networks` — 3 arch MDX + paper + 279-line impl + 3 problems
  3. `kv-cache` — covered in attention topic + problem MDX
  4. `mixture-of-experts` — full arch MDX + paper + problem
  5. `vision-transformers` — full arch MDX + paper + lab
  6. `diffusion-models` — 2 arch MDX + 2 papers + 433-line impl + lab
  7. `backpropagation` — problem + interview content
  8. `gradient-descent` — interview MDX + problem
  9. `rope-embeddings` — llama arch MDX + problem
  10. `tokenization` — referenced in gpt-2/bert papers
- **Files to create:** `src/data/topics/[slug].ts` for each, registered in `src/data/topics/index.ts`
- **Dependencies:** COURSE_GENERATION_MAP.md (YAML specs already written for all 10)
- **Verified:** `COURSE_GENERATION_MAP.md`, `CONTENT_COVERAGE_MATRIX.md`

---

### P1-004: Fill 10 Architecture Scaffold Bodies
- **Name:** Author content bodies for 10 scaffold architecture MDX files
- **Type:** Content authoring (MDX, fill empty section bodies)
- **Why it matters:** 10 architecture MDX files have all 17 section H2 headers present but all section bodies are empty (confirmed by direct file read of clip MDX). These are 35-line placeholder files. Users see headings with no content.
- **Effort:** 3–6 hours per file × 10 files = 30–60 hours
- **Priority sub-order (most-referenced first):**
  1. `clip` — central to contrastive learning topic + used in Stable Diffusion
  2. `unet` — backbone of Stable Diffusion, high pedagogical value
  3. `swin` — key modern vision architecture referenced in vision-transformers topic
  4. `dino` — self-supervised learning, referenced in contrastive-learning topic
  5. `lstm` — foundational sequence modeling; prerequisite for many concepts
  6. `gru` — natural pair with LSTM
  7. `rnn` — prerequisite for LSTM/GRU understanding
  8. `seq2seq` — important encoder-decoder concept
  9. `fcn` — foundational semantic segmentation
  10. `deeplabv3plus` — state-of-art segmentation
- **Files:** `src/content/architectures/[slug]/content.mdx` — fill all section bodies
- **Dependencies:** Each file's header structure already exists (scaffold)
- **Verified:** `INVENTORY_VALIDATION_REPORT.md` Section 1B, `ARCHITECTURE_COVERAGE_MATRIX.md`

---

### P1-005: DPO and LoRA Papers
- **Name:** Add `dpo` and `lora` papers to paper library
- **Type:** Content authoring (paper MDX, 2 papers)
- **Why it matters:** DPO has a KG node but no paper. LoRA is the dominant fine-tuning technique for LLMs referenced throughout the platform. Both are required for the RLHF/alignment learning path to be coherent.
- **Effort:** 3–6 hours each = 6–12 hours total
- **Files to create:**
  - `src/content/papers/dpo/content.mdx`
  - `src/content/papers/lora/content.mdx`
- **Dependencies:** None
- **Verified:** `PAPER_COVERAGE_MATRIX.md` missing papers table, `KNOWLEDGE_GRAPH_AUDIT.md` Section 7B

---

## P2 — MEDIUM PRIORITY (Significant Coverage Improvement)

### P2-001: CLIP Architecture MDX Content
- **Name:** Write CLIP architecture MDX body content
- **Type:** Content authoring (MDX)
- **Why it matters:** CLIP architecture scores 2/10 in the Architecture Coverage Matrix — the only P1 architecture with near-zero content. It has 17 section headers (scaffold) but all bodies are empty. CLIP is central to the contrastive-learning topic and is used by Stable Diffusion. The implementation MDX has real PyTorch code that can inform this.
- **Effort:** 4–8 hours (fill all 17 section bodies)
- **Files:** `src/content/architectures/clip/content.mdx` — replace scaffold bodies with real content
- **Dependencies:** `src/content/implementations/clip/content.mdx` (PyTorch code exists, can be referenced)
- **Verified:** `ARCHITECTURE_COVERAGE_MATRIX.md` CLIP entry, `INVENTORY_VALIDATION_REPORT.md` row 22

---

### P2-002: LLaMA and ViT Implementation MDX
- **Name:** Improve LLaMA and ViT implementation MDX files
- **Type:** Content authoring (MDX)
- **Why it matters:** LLaMA implementation is 89-line stub. ViT implementation is 122-line file with no H2 sections. Both are "complete" architectures in the catalog but their implementation walk-throughs are underdeveloped compared to the 274–510-line implementations for Transformer, BERT, GPT, ResNet, GAN, Stable Diffusion.
- **Effort:** 6–10 hours each = 12–20 hours
- **Files:**
  - `src/content/implementations/llama/content.mdx` — expand to ~250+ lines with milestones: RoPE, RMSNorm, SwiGLU, GQA
  - `src/content/implementations/vision-transformer/content.mdx` — restructure with H2 sections: patch embedding, CLS token, position encoding, classification head
- **Dependencies:** Architecture MDX for each (already complete)
- **Verified:** `INVENTORY_VALIDATION_REPORT.md` Section 3B, `ARCHITECTURE_COVERAGE_MATRIX.md`

---

### P2-003: Architecture Explorer Diagram Data for 16 Architectures
- **Name:** Add `keyFacts`, `diagram`, and `animatedSVG` to 16 architecture catalog entries
- **Type:** Data authoring (`architecture-catalog.ts`)
- **Why it matters:** The Architecture Explorer shows rich interactive diagrams for only 5 architectures (Transformer, BERT, GPT, ResNet, ViT). 16 architectures have full MDX but no catalog diagram. The Explorer shows a blank/placeholder card for all 16.
- **Effort:** 2–4 hours per architecture × 16 = 32–64 hours (or batch-author 3–4 at a time)
- **Priority sub-order:** AlexNet, VGG, GAN, VAE, CLIP, LLaMA, Stable Diffusion, DenseNet, EfficientNet, GoogLeNet, LeNet, MoE, LSTM, GRU, VGG16, VGG19
- **Files:** `src/data/architecture-catalog.ts` — add `diagram: [...]` and `keyFacts: [...]` arrays for each
- **Dependencies:** Architecture MDX (all 16 have full content; use as source for facts/diagram)
- **Verified:** `ARCHITECTURE_COVERAGE_MATRIX.md` Diagram column, `CONTENT_GAP_REPORT.md` Section 2

---

### P2-004: Knowledge Graph — Missing Architecture and Paper Nodes
- **Name:** Add 11 missing architecture nodes and 7 missing paper nodes to KNOWLEDGE_GRAPH.json
- **Type:** Data authoring (JSON)
- **Why it matters:** The KG covers 15 of 31 architectures (48%) and 11 of 19 papers (58%). Missing nodes leave concepts disconnected in the graph — the Research Hub's KG visualization doesn't show important relationships (DenseNet → ResNet → AlexNet evolution chain; SAM → ViT → CLIP relationship).
- **Effort:** 2–4 hours
- **Files:** `KNOWLEDGE_GRAPH.json` — add nodes and edges for:
  - Architecture nodes: densenet, efficientnet, googlenet, lenet, swin, dino, unet, t5, vgg16, vgg19, inceptionv3
  - Paper nodes: paper-gpt, paper-gpt2, paper-vgg, paper-alexnet, paper-batch-norm, paper-segment-anything, paper-palm
- **Dependencies:** Existing KG structure
- **Verified:** `KNOWLEDGE_GRAPH_AUDIT.md` Section 7A and 7B

---

### P2-005: 8 Remaining Topic Pages
- **Name:** Author 8 remaining topic data files (after P1-003)
- **Type:** Content authoring (TypeScript data files)
- **Why it matters:** After completing P1-003 (top 10), these 8 topics complete the full COURSE_GENERATION_MAP spec. Includes transfer-learning, batch-normalization, loss-functions, residual-networks, rlhf, rag, contrastive-learning, backpropagation (if not in P1 batch).
- **Effort:** 4–8 hours per topic × 8 = 32–64 hours
- **Files:** `src/data/topics/[slug].ts` for: transfer-learning, batch-normalization, loss-functions, residual-networks, rlhf, rag, contrastive-learning (and any remaining from P1 that weren't completed)
- **Dependencies:** P1-003 completed first (to learn the data file pattern)
- **Verified:** `COURSE_GENERATION_MAP.md`, `CONTENT_COVERAGE_MATRIX.md`

---

### P2-006: Fix Knowledge Graph Circular Dependency and Isolated Nodes
- **Name:** Fix KG: batch-normalization↔layer-normalization cycle + wire domain-rl, domain-ml, self-attention
- **Type:** Data fix (JSON)
- **Why it matters:** `batch-normalization → layer-normalization (precedes)` AND `layer-normalization → batch-normalization (extends)` creates a semantic inconsistency. Three nodes are isolated (self-attention, domain-rl, domain-ml).
- **Effort:** 1–2 hours
- **Changes:**
  1. Remove `layer-normalization → batch-normalization (extends)` OR change to `inspired_by`
  2. Add `self-attention → scaled-dot-product (extends)`, `self-attention → multi-head-attention (part_of)`
  3. Add `domain-rl → rlhf (part_of)`
  4. Add `domain-ml → gradient-descent (part_of)`, `domain-ml → transfer-learning (part_of)`, `domain-ml → backpropagation (part_of)`
- **Files:** `KNOWLEDGE_GRAPH.json`
- **Verified:** `KNOWLEDGE_GRAPH_AUDIT.md` Sections 2, 3, 5

---

### P2-007: 3 New System Design Cases
- **Name:** Add vector database, LLM serving, and search ranking system design cases
- **Type:** Content authoring (system design MDX, 6–8 sections each)
- **Why it matters:** Current 12 cases cover RAG, RecSys, and Agents well but are missing foundational infrastructure cases that appear in FAANG AI interviews. Vector DB and LLM serving are the most commonly asked system design topics at AI-focused companies.
- **Effort:** 3–5 hours per case × 3 = 9–15 hours
- **Files to create:**
  - `src/content/system-design/vector-database/content.mdx`
  - `src/content/system-design/llm-serving/content.mdx`
  - `src/content/system-design/search-ranking/content.mdx`
- **Dependencies:** None
- **Verified:** `CONTENT_GAP_REPORT.md` Section 4 (missing system design cases)

---

## P3 — LOWER PRIORITY (Completeness, Nice-to-Have)

### P3-001: GPT-4, Mamba, Llama-3, DeepSeek-R1 Papers
- **Name:** Add 4 recent/referenced papers to paper library
- **Type:** Content authoring (paper MDX, 4 papers)
- **Why it matters:** These papers are referenced by dead links or are natural extensions of existing content (GPT-3 → GPT-4, LLaMA → Llama-3). The platform is missing any coverage of Mamba (state space models as transformer alternative).
- **Effort:** 3–6 hours each = 12–24 hours total
- **Files to create:**
  - `src/content/papers/gpt-4/content.mdx`
  - `src/content/papers/llama-3/content.mdx`
  - `src/content/papers/mamba/content.mdx`
  - `src/content/papers/deepseek-r1/content.mdx`
- **Dependencies:** None
- **Verified:** `PAPER_COVERAGE_MATRIX.md` missing papers table, `CONTENT_GAP_REPORT.md` Section 3

---

### P3-002: U-Net, Swin, DINO Architecture MDX
- **Name:** Fill scaffold bodies for U-Net, Swin, DINO
- **Type:** Content authoring (MDX)
- **Why it matters:** These are the 3 architectures scoring 0/10 in the Architecture Coverage Matrix. U-Net is especially important — it is the denoising backbone of Stable Diffusion, and the absence of content creates a gap in the diffusion models learning path.
- **Effort:** 4–6 hours each × 3 = 12–18 hours
- **Files:** `src/content/architectures/unet/content.mdx`, `swin/content.mdx`, `dino/content.mdx`
- **Dependencies:** P1-004 (fill other scaffolds first to establish pattern)
- **Verified:** `ARCHITECTURE_COVERAGE_MATRIX.md` Missing tier

---

### P3-003: Interview and Quiz Content for Non-Attention Topics
- **Name:** Add interview Q&A content for top 5 topics
- **Type:** Content authoring
- **Why it matters:** Interview and Quiz dimensions score 0/10 for 16 of 18 topics. The only interview prep is for `attention` and `gradient-descent`. Adding 5–10 Q&As per topic would dramatically raise the platform's interview prep value.
- **Effort:** 2–4 hours per topic × 5 topics
- **Priority topics:** transformers, backpropagation, CNNs, kv-cache, mixture-of-experts
- **Files to create or expand:** `src/content/interview/[slug]/content.mdx` for each
- **Dependencies:** Corresponding topic pages (P1-003)
- **Verified:** `CONTENT_COVERAGE_MATRIX.md` — Quiz/Interview column

---

### P3-004: Expand Coding Problems Coverage
- **Name:** Add 20 new coding problems targeting under-covered topics
- **Type:** Content authoring (problem data in `src/data/problems.ts`)
- **Why it matters:** 6 of 22 categories have 1–2 problems each. The LLM Engineering category has only 2 problems. Batch normalization, RLHF, and transfer learning have zero dedicated problems.
- **Effort:** 1–2 hours per problem × 20 = 20–40 hours
- **Target categories:** batch-normalization, rlhf, transfer-learning, diffusion-model math, segment-anything
- **Verified:** `CONTENT_COVERAGE_MATRIX.md` Exercises column, `CONTENT_GAP_REPORT.md` Section 5

---

### P3-005: Volume 8 and Volume 12 Content (from PAPER2CODE_MASTER_BOOK)
- **Name:** Author Volume 8 (Efficient Deep Learning) and Volume 12 (Research Frontiers) content
- **Type:** Large-scale content authoring
- **Why it matters:** Per PAPER2CODE_MASTER_BOOK.md, Volume 8 (flashattention, quantization, pruning, distillation) and Volume 12 (reasoning models, multimodal LLMs, diffusion policy, foundation models) have 0 content currently authored. These represent significant portions of the planned platform scope.
- **Effort:** 40–80 hours per volume
- **Dependencies:** All P0, P1, P2 items should be complete before investing this much in new content
- **Verified:** `PAPER2CODE_MASTER_BOOK.md` production priority order

---

## BACKLOG SUMMARY TABLE

| ID | Priority | Name | Type | Effort | Unblocked |
|----|---------|------|------|--------|----------|
| P0-001 | P0 | Fix LLaMA catalog status | 1-line fix | 30 min | ✅ Now |
| P0-002 | P0 | Fix VGG slug mismatch | Data fix | 2–4 hr | ✅ Now |
| P0-003 | P0 | Author 9 domain data files | TypeScript | 18–36 hr | ✅ Now |
| P0-004 | P0 | Fill 5 problem MDX stubs | MDX | 10–20 hr | ✅ Now |
| P0-005 | P0 | Fix KG metadata counts | JSON | 30 min | ✅ Now |
| P1-001 | P1 | InstructGPT/RLHF paper | MDX | 3–6 hr | ✅ Now |
| P1-002 | P1 | Flash Attention paper | MDX | 3–6 hr | ✅ Now |
| P1-003 | P1 | Top 10 topic pages | TypeScript | 40–80 hr | ✅ Now |
| P1-004 | P1 | Fill 10 architecture scaffolds | MDX | 30–60 hr | ✅ Now |
| P1-005 | P1 | DPO and LoRA papers | MDX | 6–12 hr | ✅ Now |
| P2-001 | P2 | CLIP architecture content | MDX | 4–8 hr | After P1-004 |
| P2-002 | P2 | LLaMA and ViT impl MDX | MDX | 12–20 hr | ✅ Now |
| P2-003 | P2 | Explorer diagrams for 16 archs | Catalog data | 32–64 hr | ✅ Now |
| P2-004 | P2 | KG missing nodes/edges | JSON | 2–4 hr | ✅ Now |
| P2-005 | P2 | 8 remaining topic pages | TypeScript | 32–64 hr | After P1-003 |
| P2-006 | P2 | Fix KG cycles + isolated nodes | JSON | 1–2 hr | ✅ Now |
| P2-007 | P2 | 3 new system design cases | MDX | 9–15 hr | ✅ Now |
| P3-001 | P3 | GPT-4/Mamba/Llama-3/DeepSeek papers | MDX | 12–24 hr | After P1 |
| P3-002 | P3 | U-Net/Swin/DINO architecture MDX | MDX | 12–18 hr | After P1-004 |
| P3-003 | P3 | Interview Q&A for 5 topics | MDX | 10–20 hr | After P1-003 |
| P3-004 | P3 | 20 new coding problems | Problem data | 20–40 hr | After P1-003 |
| P3-005 | P3 | Vol 8 and Vol 12 content | Large-scale | 80–160 hr | After all P2 |

### Total Estimated Effort by Priority
| Priority | Items | Estimated Hours |
|---------|-------|----------------|
| P0 (fix now) | 5 items | ~30–60 hours |
| P1 (high value) | 5 items | ~82–164 hours |
| P2 (medium) | 7 items | ~92–177 hours |
| P3 (nice-to-have) | 5 items | ~134–262 hours |
| **Total** | **22 items** | **~340–660 hours** |
