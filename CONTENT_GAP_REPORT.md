# Paper2Code Content Gap Report
**Generated:** 2026-06-20 | **Type:** Quality audit — gaps only, no content generated

This report audits every piece of content in Paper2Code for completeness and identifies gaps.
Status definitions: **Complete** = all required sections present | **Mostly Complete** = ≥80% done | **Partial** = 40–79% done | **Missing** = no topic page exists

---

## CRITICAL GAPS (Platform-breaking issues)

### GAP-001: Only 1 of ~100 topic pages authored
**Severity:** Critical
**Impact:** `/learn/[domain]/[topic]` returns "Topic Not Found" for all slugs except `attention`
**Affected:** All 12 domains × ~8 topics each = ~96 missing topic pages
**Evidence:** `src/data/topics/index.ts` only exports `attention`
**Action needed:** Author `src/data/topics/[slug].ts` for each topic, register in `index.ts`

### GAP-002: 9 of 12 domain pages show fallback placeholder content
**Severity:** Critical
**Impact:** `/learn/computer-vision`, `/learn/nlp`, `/learn/rl`, and 6 others show generic scaffold
**Affected domains:** computer-vision, nlp, reinforcement-learning, statistics, mathematics, mlops, ai-systems, research-methods, robotics
**Evidence:** `src/data/domains/index.ts` — only `deep-learning`, `machine-learning`, `llms` have full `DomainData`
**Action needed:** Author `src/data/domains/[slug].ts` for each domain, register in index

### GAP-003: Architecture catalog status mismatch for LLaMA
**Severity:** Medium
**Impact:** Architecture Explorer shows LLaMA as "Coming Soon" even though MDX exists
**Affected:** `src/data/architecture-catalog.ts` entry for `llama`
**Evidence:** `src/content/architectures/llama/content.mdx` exists; catalog entry has `status: "coming-soon"`
**Action needed:** Update catalog entry status to `"complete"` and add diagram/keyFacts

### GAP-004: VGG architecture MDX vs catalog slug mismatch
**Severity:** Medium
**Impact:** Architecture Explorer `vgg` stub links to nothing; real MDX is at `vgg16`/`vgg19`
**Affected:** Users clicking VGG in Explorer get coming-soon placeholder
**Action needed:** Either rename MDX dirs to `vgg` or update catalog stub to point to `vgg16`

---

## SECTION 1 — TOPIC PAGES AUDIT

| Topic Slug | Domain | Status | What Exists | What's Missing |
|-----------|--------|--------|------------|----------------|
| attention | deep-learning | ✅ Complete | Full 13-section topic data | Nothing |
| transformers | deep-learning | 🔴 Missing | Architecture MDX, paper MDX | Entire topic page |
| backpropagation | deep-learning | 🔴 Missing | Problem data (backpropagation slug) | Entire topic page |
| convolutional-networks | deep-learning | 🔴 Missing | Problem data, arch MDX | Entire topic page |
| residual-networks | deep-learning | 🔴 Missing | Architecture MDX + paper | Entire topic page |
| batch-normalization | deep-learning | 🔴 Missing | Paper MDX | Entire topic page |
| gradient-descent | machine-learning | 🟡 Partial | Interview MDX, problem data | Topic page in src/data/topics/ |
| loss-functions | machine-learning | 🔴 Missing | Problem data only | Entire topic page |
| transfer-learning | machine-learning | 🔴 Missing | Concepts mentioned in docs | Entire topic page |
| tokenization | llms | 🔴 Missing | Nothing | Entire topic page |
| kv-cache | llms | 🟡 Partial | Problem MDX (gpt-kv-cache-scaling.mdx) | Topic page, theory section |
| rope-embeddings | llms | 🟡 Partial | Problem MDX (llama-rope.mdx) | Topic page, theory section |
| mixture-of-experts | llms | 🟡 Partial | Architecture MDX, paper MDX | Topic page |
| rlhf | llms | 🔴 Missing | References in docs only | Entire topic page |
| rag | llms | 🟡 Partial | 3 system design cases, no topic page | Topic page |
| vision-transformers | computer-vision | 🟡 Partial | Architecture MDX, paper MDX, impl MDX | Topic page |
| diffusion-models | computer-vision | 🟡 Partial | Architecture MDX, paper MDX, impl MDX | Topic page |
| contrastive-learning | computer-vision | 🟡 Partial | Architecture MDX (CLIP) | Topic page |

**Total: 1 complete, 5 partial (have supporting content), 12 entirely missing**

---

## SECTION 2 — ARCHITECTURE PAGES AUDIT

| Slug | Status | Has MDX | Has Diagram | Has Math | Has Code | Has Papers |
|------|--------|---------|------------|---------|---------|-----------|
| transformer | ✅ Complete | ✅ | ✅ | ✅ | ✅ | ✅ |
| resnet | ✅ Complete | ✅ | ✅ | ✅ | ✅ | ✅ |
| bert | ✅ Complete | ✅ | ✅ | ✅ | ✅ | ✅ |
| gpt | ✅ Complete | ✅ | ✅ | ✅ | ✅ | ✅ |
| vit | ✅ Complete | ✅ | ✅ | ✅ | ✅ | ✅ |
| alexnet | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| vgg16 | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| vgg19 | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| googlenet | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| inceptionv3 | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| densenet | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| efficientnet | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| unet | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| swin | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| clip | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| dino | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| deeplabv3plus | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| fcn | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| lenet | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| rnn | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| gru | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| lstm | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| seq2seq | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| ae | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| vae | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| gan | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| diffusion | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| stable-diffusion | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| t5 | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| llama | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ⚠️ Catalog says coming-soon |
| moe | 🟡 Mostly Complete | ✅ | ❌ | ❌ | ❌ | ❌ |
| yolo | 🔴 Stub | ❌ | ❌ | ❌ | ❌ | ❌ |
| roberta | 🔴 Stub | ❌ | ❌ | ❌ | ❌ | ❌ |
| gpt-2-arch | 🔴 Stub | ❌ | ❌ | ❌ | ❌ | ❌ |
| mamba | 🔴 Stub | ❌ | ❌ | ❌ | ❌ | ❌ |

**Summary: 5 complete, 26 have MDX (need diagrams/math/code), 4 stubs**

**The 26 "mostly complete" architectures all have MDX content pages but lack the interactive elements (animated diagrams, math snippets, code snippets) that the Architecture Explorer shows for the 5 complete ones.**

---

## SECTION 3 — PAPER LIBRARY AUDIT

| Paper Slug | Status | Year | Completeness Note |
|-----------|--------|------|------------------|
| alexnet | ✅ Complete | 2012 | — |
| vgg | ✅ Complete | 2014 | — |
| gan | ✅ Complete | 2014 | — |
| deep-residual-learning | ✅ Complete | 2015 | — |
| batch-normalization | ✅ Complete | 2015 | — |
| attention-is-all-you-need | ✅ Complete | 2017 | — |
| bert | ✅ Complete | 2018 | — |
| gpt | ✅ Complete | 2018 | — |
| gpt-2 | ✅ Complete | 2019 | — |
| gpt-3 | ✅ Complete | 2020 | — |
| vision-transformer | ✅ Complete | 2020 | — |
| switch-transformer | ✅ Complete | 2021 | — |
| chinchilla | ✅ Complete | 2022 | — |
| palm | ✅ Complete | 2022 | — |
| clip | ✅ Complete | 2021 | — |
| latent-diffusion-models | ✅ Complete | 2022 | — |
| stable-diffusion | ✅ Complete | 2022 | — |
| segment-anything | ✅ Complete | 2023 | — |
| llama | ✅ Complete | 2023 | — |

**19/19 papers complete. No gaps in paper library.**

**Papers missing from collection (compared to broader ML literature):**
- Flash Attention (2022) — referenced by dead link, no MDX
- DeepSeek-R1 (2025) — referenced by dead link, no MDX
- Llama-3 (2024) — no MDX
- Mamba (2023) — no MDX
- GPT-4 Technical Report (2023) — no MDX
- InstructGPT / RLHF paper (2022) — no MDX
- DPO (2023) — no MDX
- LoRA (2021) — no MDX
- Constitutional AI (2022) — no MDX

---

## SECTION 4 — SYSTEM DESIGN CASES AUDIT

| Case Slug | Status | Category | Completeness |
|-----------|--------|----------|-------------|
| chatgpt-system-design | ✅ Complete | LLM Serving | — |
| recommendation-engine | ✅ Complete | RecSys | — |
| github-copilot | ✅ Complete | LLM Serving | — |
| perplexity | ✅ Complete | RAG | — |
| basic-rag | ✅ Complete | RAG | — |
| netflix-recommendation | ✅ Complete | RecSys | — |
| tiktok-recommendation | ✅ Complete | RecSys | — |
| single-agent | ✅ Complete | Agents | — |
| multi-agent | ✅ Complete | Agents | — |
| youtube-recommendation | ✅ Complete | RecSys | — |
| advanced-rag | ✅ Complete | RAG | — |
| agentic-rag | ✅ Complete | RAG | — |

**12/12 complete. No gaps.**

**Missing system design cases (for a truly comprehensive AI Systems volume):**
- Vector Database design (Pinecone, Weaviate)
- LLM Serving infrastructure (vLLM, TGI)
- DeepSeek training architecture
- Uber Surge Pricing ML system
- Anomaly detection system
- Search ranking (Google/Bing)

---

## SECTION 5 — CODING PROBLEMS AUDIT

### Next.js Dojo (22 problems)
All 22 problems in `src/data/problems.ts` are complete (have description, test cases, hints, solution, explanation).

**Coverage gaps by category:**
| Category | Problems | Missing Topics |
|---------|---------|---------------|
| linear-algebra | 5 | Eigenvalues, SVD, QR decomposition |
| deep-learning | 5 | Attention visualization, FLOP calculation |
| cnn | 3 | Stride/dilation shapes, depthwise conv |
| transformer | 5 | Cross-attention, encoder-decoder |
| llm-engineering | 2 | Beam search, speculative decoding, RLHF reward model |

**Problems with MDX deep-dives (8):** all exist in `src/content/problems/`

### Static Dojo (110 problems)
110 Python execution problems in `static/index.html`. All complete (have description, starter code, test cases, hints, solution).

**Note:** These 110 problems are separate from the 22 Next.js Dojo problems and cover a broader range of DS/ML topics with actual Python execution.

---

## SECTION 6 — LEARNING LOOP AUDIT

The Research → Learn → Practice → Research loop was completed in Phase 16DEF:

| Loop Segment | Status | Implementation |
|-------------|--------|---------------|
| Research → Learn | ✅ Complete | KG node panel → "Learn This Concept" CTA |
| Learn → Practice | ✅ Complete | Topic page → Recommended Problems section |
| Practice → Research | ✅ Complete | Dojo history tab → Related Research panel |
| Learn → Research (papers) | ✅ Complete | Topic pages show related papers |

**38 concept-to-URL mappings exist** in `PaperKnowledgeGraph.tsx`. Gap: only ~10 have a `learnUrl` that resolves to an actual authored topic page (because only `attention` topic exists).

---

## SECTION 7 — INTERACTIVE FEATURES AUDIT

| Feature | Status | Gap |
|---------|--------|-----|
| Architecture Explorer | 🟡 Partial | 26/35 architectures lack diagrams/math/code |
| Code Dojo (Next.js) | ✅ Complete | Missing coverage for speculative decoding, beam search |
| Code Dojo (Static) | ✅ Complete | 110 problems, all functional |
| AI Labs | ✅ Complete | 4 labs operational |
| Block Visualizer | ✅ Complete | — |
| Paper Upload → KG | ✅ Complete | — |
| Research Hub | ✅ Complete | — |
| Learn Domain Pages | 🔴 Critical | 9/12 domains show placeholder content |
| Learn Topic Pages | 🔴 Critical | 1/~100 topics authored |

---

## PRIORITY ACTION LIST (Ranked by impact)

| Priority | Action | Files to Create | Impact |
|---------|--------|----------------|--------|
| 1 | Author 9 missing domain data files | `src/data/domains/computer-vision.ts`, `nlp.ts`, etc. | Fixes 9 broken domain pages |
| 2 | Author 10 highest-priority topic data files | See COURSE_GENERATION_MAP.md | Enables full Learn → Practice loop |
| 3 | Fix LLaMA catalog status | `src/data/architecture-catalog.ts` line for llama | 1 line fix |
| 4 | Add Explorer diagrams to 26 architecture MDX files | `src/data/architecture-catalog.ts` | Completes Explorer feature |
| 5 | Add 8 missing papers | `src/content/papers/flash-attention/`, etc. | Completes paper library for modern ML |
| 6 | Author Volume 8 (Efficient DL) topics | flashattention, quantization, etc. | 0 content exists here |
| 7 | Author Volume 12 (Research Frontiers) | reasoning-models, mamba, etc. | 0 content exists here |
| 8 | Expand system design to 18 cases | LLM serving, vector DB, search | Completes AI Systems volume |
