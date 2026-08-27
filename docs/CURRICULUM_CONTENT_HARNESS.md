# Curriculum Content Harness — All Batches

## STATUS (checked 2026-08-26)
`src/data/content/curriculum.ts` defines 12 domains / 82 topics with rich metadata
(level, prerequisites, studyTime, why, unlocks) but **zero topics have lesson
content**. This gap was filled on 2026-08-27: all 82 canonical topics now have
lesson bodies and `/learn/[domain]/[topic]/page.tsx` renders their MDX.

Run `node scripts/audit-curriculum-content.mjs` after edits. It verifies
canonical coverage, section structure, question count, minimum lesson length,
control characters, and full MDX/KaTeX compilation.

## HOW TO USE
Copy **MASTER PREAMBLE** + one **BATCH** block into Antigravity per session.
Each batch creates the listed `content.mdx` files. Antigravity has repo
access — instruct it to open `src/data/content/curriculum.ts` and read the
named domain's topic objects directly rather than re-deriving facts; do not
let it invent different prerequisites/unlocks than what's already there.

---

## MASTER PREAMBLE (paste at the top of every prompt)

You are writing technical MDX lesson content for paper2code, an educational
ML platform, for its **Curriculum** track (distinct from the Architectures
and Papers tracks — this is textbook-style teaching content, not a
paper/model deep-dive).

Before writing each file: open `src/data/content/curriculum.ts`, find the
domain by `slug`, and read that topic's `title`, `level`, `prerequisites`,
`studyTime`, `why`, and `unlocks`. Treat these as ground truth — the lesson
must be consistent with them, not contradict or replace them.

Each file goes at `src/content/curriculum/{domain-slug}/{topic-slug}/content.mdx`.
Use GitHub-flavored Markdown with KaTeX math (`$inline$`, `$$block$$`).

**Required section structure (copy exactly):**

```
# [Topic Title]

## 1. Overview
## 2. Prerequisites in Context
## 3. Core Concept
## 4. Theoretical Foundations
## 5. Worked Example
## 6. Code / Implementation
## 7. Real-World Applications
## 8. Common Pitfalls & Misconceptions
## 9. How This Connects
## 10. Check Your Understanding
## 11. Further Reading
```

**Content rules:**
- Section 1 must open with a 2-3 sentence plain-English definition, then expand on the topic's `why` field from curriculum.ts (don't just repeat it verbatim — explain it).
- Section 2 must explain *why* each listed prerequisite is actually needed here, in prose — not just restate the array.
- Section 4 adapts to topic type:
  - **Theory/math topics** (most of domains 1-4, 8): real equations in KaTeX, every symbol defined, at least one derivation step shown.
  - **Engineering/tooling topics** (most of domains 5, 10, and all of domain 12's Beginner/Intermediate topics — e.g. Git, W&B, environment setup): replace equations with "Key Commands & Workflow" — concrete CLI/config examples instead of math.
  - **Systems/architecture topics** (domain 11, parts of 6/7/9): a labeled ASCII diagram of the system plus the core design equation/formula if one exists (e.g. throughput, latency, memory formulas).
- Section 6 must include a runnable snippet (Python/PyTorch for ML topics, bash/config for tooling topics) of 20-60 lines. Trivial tooling topics (e.g. "Git for Research Projects") may use a shell-command walkthrough instead of a Python class.
- Section 9 must explicitly reference the topic's own `prerequisites` and `unlocks` arrays from curriculum.ts and narrate the learning path in prose (e.g. "Having grounded X, this topic becomes the foundation for Y and Z because...").
- Section 10 must contain exactly 5 Q&A pairs with full multi-sentence answers.
- Every section must be substantive — minimum 120 words each (curriculum lessons are shorter than architecture deep-dives; don't pad).
- Use `:::note`, `:::tip`, `:::warning` admonitions where helpful.

---

## BATCH CURR-1 — Mathematics (Domain 1 of 12)

**Create these files:**
- `src/content/curriculum/mathematics/statistical-learning-theory/content.mdx`
- `src/content/curriculum/mathematics/optimal-transport/content.mdx`
- `src/content/curriculum/mathematics/differential-geometry-for-ml/content.mdx`
- `src/content/curriculum/mathematics/variational-inference-and-free-energy/content.mdx`
- `src/content/curriculum/mathematics/spectral-methods-and-random-matrix-theory/content.mdx`

**Context:** All Advanced-level, all pure theory. Section 4 must carry real
derivations (VC dimension / Rademacher complexity for the first; Wasserstein
distance and Kantorovich duality for the second; Riemannian metrics and
geodesics for the third; ELBO derivation for the fourth; Marchenko-Pastur law
for the fifth). Section 7 must connect each to a specific downstream
architecture or technique already on the platform (e.g. optimal transport →
WGAN in `src/content/architectures/wgan`, variational inference → VAE/DDPM).

---

## BATCH CURR-2 — Machine Learning (Domain 2 of 12)

**Create these files:**
- `src/content/curriculum/machine-learning/meta-learning/content.mdx`
- `src/content/curriculum/machine-learning/federated-learning/content.mdx`
- `src/content/curriculum/machine-learning/neural-architecture-search-nas/content.mdx`
- `src/content/curriculum/machine-learning/learning-theory/content.mdx`
- `src/content/curriculum/machine-learning/multi-task-and-transfer-learning-theory/content.mdx`

**Context:** NAS topic must cross-reference `src/content/architectures/nasnet`
and `src/content/architectures/darts` in Section 7/9 — this platform already
has deep architecture content on both; link the lesson to them instead of
re-explaining DARTS math from scratch.

---

## BATCH CURR-3 — Deep Learning (Domain 3 of 12)

**Create these files:**
- `src/content/curriculum/deep-learning/scaling-laws/content.mdx`
- `src/content/curriculum/deep-learning/mixture-of-experts-moe/content.mdx`
- `src/content/curriculum/deep-learning/mechanistic-interpretability/content.mdx`
- `src/content/curriculum/deep-learning/neural-odes-and-continuous-depth-networks/content.mdx`
- `src/content/curriculum/deep-learning/emergent-abilities-and-phase-transitions/content.mdx`

**Context:** Scaling Laws must reference `src/content/architectures/chinchilla`.
MoE must reference `src/content/architectures/mixtral-8x7b`. Don't duplicate
those architecture pages' content — this lesson teaches the general
principle; the architecture page is the concrete case study. Link both ways.

---

## BATCH CURR-4 — Natural Language Processing (Domain 4 of 12)

**Create these files:**
- `src/content/curriculum/natural-language-processing/reasoning-in-language-models/content.mdx`
- `src/content/curriculum/natural-language-processing/factuality-grounding-and-hallucination/content.mdx`
- `src/content/curriculum/natural-language-processing/alignment/content.mdx`
- `src/content/curriculum/natural-language-processing/neural-machine-translation-at-scale/content.mdx`
- `src/content/curriculum/natural-language-processing/language-model-interpretability/content.mdx`

**Context:** Alignment must reference `src/content/architectures/instructgpt`
(RLHF pipeline already documented there) rather than re-deriving PPO/RLHF
math — summarize and link. Factuality/hallucination topic should reference
the RAG Systems domain (Domain 6) topics in Section 9 "unlocks" narration.

---

## BATCH CURR-5 — LLM Engineering (Domain 5 of 12)

**Create these files:**
- `src/content/curriculum/llm-engineering/continued-pre-training-and-domain-adaptation-at-scale/content.mdx`
- `src/content/curriculum/llm-engineering/flash-attention-and-memory-efficient-attention/content.mdx`
- `src/content/curriculum/llm-engineering/tensor-parallelism-pipeline-parallelism-and-fsdp/content.mdx`
- `src/content/curriculum/llm-engineering/continual-learning-and-catastrophic-forgetting/content.mdx`
- `src/content/curriculum/llm-engineering/model-merging/content.mdx`

**Context:** Flash Attention and Tensor/Pipeline Parallelism are systems
topics — Section 4 should show IO-complexity or memory-footprint formulas,
not abstract theory. These also overlap with Domain 11 (AI System Design);
don't fully duplicate — this lesson is the concept, Domain 11's topics are
the production system built from it.

---

## BATCH CURR-6 — RAG Systems (Domain 6 of 12)

**Create these files:**
- `src/content/curriculum/rag-systems/raft/content.mdx`
- `src/content/curriculum/rag-systems/multi-modal-rag/content.mdx`
- `src/content/curriculum/rag-systems/production-rag-architecture-at-scale/content.mdx`
- `src/content/curriculum/rag-systems/continual-indexing-and-knowledge-freshness/content.mdx`
- `src/content/curriculum/rag-systems/rag-for-code/content.mdx`

**Context:** `production-rag-architecture-at-scale` and `system-design-for-retrieval-at-scale`
(Domain 11) are near-duplicates by name — this lesson should stay at the
"what and why" level; leave deep capacity-planning / sharding detail to the
System Design article for `rag-systems` (see the separate System Design
harness) and link to it in Section 7.

---

## BATCH CURR-7 — AI Agents (Domain 7 of 12)

**Create these files:**
- `src/content/curriculum/ai-agents/emergent-cooperation-in-multi-agent-systems/content.mdx`
- `src/content/curriculum/ai-agents/continual-learning-agents/content.mdx`
- `src/content/curriculum/ai-agents/simulation-environments-for-agent-training/content.mdx`
- `src/content/curriculum/ai-agents/world-models-for-agents/content.mdx`
- `src/content/curriculum/ai-agents/formal-specification-and-verification-of-agent-behavior/content.mdx`

**Context:** `world-models-for-agents` overlaps with Domain 8's `world-models`
topic — this one must stay agent-planning-focused (how an agent uses a
learned model to plan), Domain 8's is the RL/Dreamer mechanics. Cross-link,
don't merge.

---

## BATCH CURR-8 — Reinforcement Learning (Domain 8 of 12)

**Create these files:**
- `src/content/curriculum/reinforcement-learning/world-models/content.mdx`
- `src/content/curriculum/reinforcement-learning/hierarchical-rl/content.mdx`
- `src/content/curriculum/reinforcement-learning/safe-rl-and-constrained-mdp/content.mdx`
- `src/content/curriculum/reinforcement-learning/alphazero-and-mcts-with-deep-rl/content.mdx`
- `src/content/curriculum/reinforcement-learning/foundation-models-as-world-models/content.mdx`

**Context:** `world-models` and `alphazero-and-mcts-with-deep-rl` must
reference `src/content/architectures/dreamer` and
`src/content/architectures/alphazero` respectively — those architecture
pages already have the full math; this lesson teaches the surrounding
concept and links out for depth.

---

## BATCH CURR-9 — Computer Vision (Domain 9 of 12)

**Create these files:**
- `src/content/curriculum/computer-vision/vision-language-models/content.mdx`
- `src/content/curriculum/computer-vision/foundation-models-for-vision/content.mdx`
- `src/content/curriculum/computer-vision/video-generation/content.mdx`
- `src/content/curriculum/computer-vision/generative-video-editing-and-controlnet/content.mdx`
- `src/content/curriculum/computer-vision/embodied-vision/content.mdx`

**Context:** `foundation-models-for-vision` should reference `src/content/architectures/sam`
and `clip`. `generative-video-editing-and-controlnet` should reference
`src/content/architectures/controlnet`. Same rule as other batches: this
lesson is the conceptual overview, the architecture page is the deep-dive.

---

## BATCH CURR-10 — MLOps (Domain 10 of 12)

**Create these files:**
- `src/content/curriculum/mlops/ml-platform-engineering/content.mdx`
- `src/content/curriculum/mlops/large-scale-training-infrastructure/content.mdx`
- `src/content/curriculum/mlops/real-time-inference-at-scale/content.mdx`
- `src/content/curriculum/mlops/cost-engineering-and-carbon-accounting-for-ml/content.mdx`
- `src/content/curriculum/mlops/ml-governance-and-compliance/content.mdx`

**Context:** These are production-engineering topics — Section 4 ("Key
Commands & Workflow" variant) should show real tool config (Kubernetes YAML
snippet, cost-per-GPU-hour formula, etc.), not abstract ML theory. Section 7
should name real internal platforms already referenced in curriculum.ts's
`why` field (Uber Michelangelo, Airbnb Bighead, Meta FBLearner) and expand on
them.

---

## BATCH CURR-11A — AI System Design, part 1 (Domain 11 of 12)

**Create these files:**
- `src/content/curriculum/ai-system-design/inference-optimization/content.mdx`
- `src/content/curriculum/ai-system-design/speculative-decoding/content.mdx`
- `src/content/curriculum/ai-system-design/distributed-inference/content.mdx`
- `src/content/curriculum/ai-system-design/system-design-for-retrieval-at-scale/content.mdx`
- `src/content/curriculum/ai-system-design/llm-gateway-design/content.mdx`
- `src/content/curriculum/ai-system-design/evaluation-infrastructure/content.mdx`

**Context:** These are the most systems-heavy topics in the whole curriculum
— Section 4 must be a real ASCII architecture diagram plus a throughput or
latency formula, not prose. `system-design-for-retrieval-at-scale` should
link to the `rag-systems` System Design article (separate harness) as the
canonical deep dive and stay at concept level here.

---

## BATCH CURR-11B — AI System Design, part 2 (Domain 11 of 12)

**Create these files:**
- `src/content/curriculum/ai-system-design/hardware-aware-architecture-search/content.mdx`
- `src/content/curriculum/ai-system-design/custom-cuda-kernels-for-ml/content.mdx`
- `src/content/curriculum/ai-system-design/multi-cluster-training-orchestration/content.mdx`
- `src/content/curriculum/ai-system-design/production-safety-and-alignment-systems/content.mdx`
- `src/content/curriculum/ai-system-design/compound-ai-systems/content.mdx`

**Context:** All Expert-level. `custom-cuda-kernels-for-ml` Section 6 should
show a minimal Triton or CUDA-style kernel sketch, not just PyTorch.
`compound-ai-systems` Section 9 should tie together nearly every other
domain (RAG, Agents, LLM Engineering) as the capstone integration topic.

---

## BATCH CURR-12A — Research Engineering, Beginner tools (Domain 12 of 12)

**Create these files:**
- `src/content/curriculum/research-engineering/reading-ml-papers-effectively/content.mdx`
- `src/content/curriculum/research-engineering/setting-up-a-research-environment/content.mdx`
- `src/content/curriculum/research-engineering/git-for-research-projects/content.mdx`
- `src/content/curriculum/research-engineering/experiment-tracking-basics-wandb-mlflow/content.mdx`
- `src/content/curriculum/research-engineering/writing-research-code/content.mdx`
- `src/content/curriculum/research-engineering/reproducing-sota-results/content.mdx`

**Context:** These are practical-skill topics, not theory. Section 4 becomes
"Key Commands & Workflow" for all six — real `conda`/`git`/`wandb` CLI
snippets in Section 6, not neural net code. Keep it concrete and short; this
is the onboarding tier of the whole curriculum.

---

## BATCH CURR-12B — Research Engineering, Intermediate practice (Domain 12 of 12)

**Create these files:**
- `src/content/curriculum/research-engineering/ablation-study-design/content.mdx`
- `src/content/curriculum/research-engineering/hyperparameter-search-at-scale/content.mdx`
- `src/content/curriculum/research-engineering/dataset-curation-and-preprocessing-pipelines/content.mdx`
- `src/content/curriculum/research-engineering/writing-technical-research-reports/content.mdx`
- `src/content/curriculum/research-engineering/open-source-library-contribution/content.mdx`

**Context:** Section 6 for `hyperparameter-search-at-scale` should show an
Optuna/W&B Sweeps config snippet. `ablation-study-design` Section 4 should
include the statistical-significance formula for comparing two runs
(t-test or bootstrap CI), tying back to `statistical-learning-theory` in
Domain 1.

---

## BATCH CURR-12C — Research Engineering, Advanced research (Domain 12 of 12)

**Create these files:**
- `src/content/curriculum/research-engineering/large-scale-pretraining-experiments/content.mdx`
- `src/content/curriculum/research-engineering/scaling-law-modeling/content.mdx`
- `src/content/curriculum/research-engineering/efficient-fine-tuning-research/content.mdx`
- `src/content/curriculum/research-engineering/evaluation-research/content.mdx`
- `src/content/curriculum/research-engineering/implementing-novel-architectures/content.mdx`

**Context:** `scaling-law-modeling` must reference `chinchilla` and
`scaling-laws` (Domain 3) directly and show the power-law curve-fitting
procedure in Section 6 (Python: fit $L(N,D)$ to a few (N, D, loss) points).
`efficient-fine-tuning-research` should reference `src/content/architectures`
LoRA-adjacent context if present, otherwise define LoRA/QLoRA inline.

---

## BATCH CURR-12D — Research Engineering, Expert frontier (Domain 12 of 12)

**Create these files:**
- `src/content/curriculum/research-engineering/first-principles-research-taste/content.mdx`
- `src/content/curriculum/research-engineering/mechanistic-interpretability-2/content.mdx`
- `src/content/curriculum/research-engineering/alignment-research-methods/content.mdx`
- `src/content/curriculum/research-engineering/research-paper-writing-for-top-venues/content.mdx`
- `src/content/curriculum/research-engineering/building-research-infrastructure-at-scale/content.mdx`

**Context:** `mechanistic-interpretability-2` is a duplicate-titled sibling
of Domain 3's `mechanistic-interpretability` — this Expert-tier version must
go deeper (SAEs, superposition, circuits) and explicitly say in Section 2
how it builds on the Domain 3 topic rather than repeating it.
`first-principles-research-taste` has no code section to speak of; Section 6
may instead be a short case-study walkthrough of a real research decision.

---

## QUALITY CHECKLIST (Antigravity must verify before finishing each file)

- [ ] Section 1 expands the `why` field from curriculum.ts, doesn't just quote it
- [ ] Section 2 explains *why* each prerequisite matters, not just a restated list
- [ ] Section 4 matches topic type (equations for theory, commands for tooling, diagram+formula for systems)
- [ ] Section 6 has a runnable/concrete snippet, 20-60 lines or an equivalent command walkthrough
- [ ] Section 9 explicitly narrates this topic's own `prerequisites` and `unlocks` arrays from curriculum.ts
- [ ] Section 10 has exactly 5 Q&A pairs with full multi-sentence answers
- [ ] No section is a stub — every section has substantive content (120+ words)
- [ ] Cross-references to existing architecture/paper/system-design pages use real slugs that exist in the repo (check `src/content/architectures/`, don't invent slugs)
