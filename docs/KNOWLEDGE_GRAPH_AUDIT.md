# Paper2Code Knowledge Graph Audit
**Generated:** 2026-06-21 | **Source:** KNOWLEDGE_GRAPH.json (computed analysis)

This audit computes structural properties of the KNOWLEDGE_GRAPH.json to validate coverage and find gaps.

---

## SECTION 1 — BASIC STATISTICS

| Metric | Claimed (metadata) | Actual (computed) | Discrepancy |
|--------|-------------------|-------------------|------------|
| Total Nodes | 95 | **94** | -1 (metadata overcounts by 1) |
| Total Edges | 187 | **164** | -23 (metadata overcounts by 23) |
| Relation Types | 10 | 10 | ✓ |

### Node Type Breakdown (Actual)

| Type | Count | IDs |
|------|-------|-----|
| topic | 1 | attention |
| architecture | 15 | transformer, resnet, bert, gpt, vit, llama, moe-arch, stable-diffusion, gan, vae, clip, alexnet, vgg, lstm, gru |
| concept | 46 | multi-head-attention, self-attention, scaled-dot-product, positional-encoding, layer-normalization, feed-forward-network, residual-connection, softmax, gradient-descent, backpropagation, cross-entropy-loss, sigmoid, relu, batch-normalization, dropout, convolution, max-pooling, skip-connection, embedding, tokenization, pre-training, fine-tuning, transfer-learning, rlhf, dpo, kv-cache, flash-attention, grouped-query-attention, rope, rmsnorm, swiglu, mixture-of-experts, sparse-routing, patch-embedding, contrastive-learning, score-matching, diffusion-process, latent-space, variational-inference, adversarial-training, rag, vector-search, embedding-model, agent, tool-use, chain-of-thought |
| math | 3 | linear-algebra, matrix-multiplication, dot-product |
| paper | 11 | paper-attention, paper-resnet, paper-bert, paper-gpt3, paper-vit, paper-chinchilla, paper-llama, paper-clip, paper-diffusion, paper-gan, paper-switch |
| domain | 6 | domain-deep-learning, domain-llms, domain-cv, domain-rl, domain-nlp, domain-ml |
| system-design | 6 | chatgpt-system, netflix-system, rag-system, advanced-rag-system, agent-system, multi-agent-system |
| problem | 6 | prob-matrix-mult, prob-attention, prob-kv-cache, prob-rope, prob-moe, prob-backprop |
| **TOTAL** | **94** | |

### Edge Relation Type Breakdown

| Relation | Count (estimated) |
|---------|-----------------|
| uses | ~55 |
| derived_from | ~12 |
| implements | ~15 |
| extends | ~18 |
| part_of | ~22 |
| requires | ~12 |
| enables | ~8 |
| references | ~8 |
| explains | ~4 |
| precedes | ~1 |

---

## SECTION 2 — ISOLATED NODES

Isolated nodes have ZERO incoming or outgoing edges. They exist in the graph but have no connections.

| Node ID | Type | Title | Impact |
|---------|------|-------|--------|
| `self-attention` | concept | Self-Attention | Referenced in architecture content but not wired to any concept node. Should connect to `multi-head-attention` and `scaled-dot-product` |
| `domain-rl` | domain | Reinforcement Learning | Domain exists but connects to nothing — `rlhf` (which uses RL) has no domain link |
| `domain-ml` | domain | Machine Learning | Domain exists but connects to nothing — `gradient-descent`, `transfer-learning` should be `part_of` this domain |

**Total isolated nodes: 3 of 94 (3.2%)**

---

## SECTION 3 — SOURCE-ONLY NODES (No Incoming Edges)

These nodes have outgoing edges but nothing points to them. They are graph roots.

| Node ID | Type | Outgoing Edges | Note |
|---------|------|---------------|------|
| `dpo` | concept | `dpo → rlhf` | No node points to DPO; correct as a modern technique |
| `flash-attention` | concept | `flash-attention → scaled-dot-product` | Should be targeted by `kv-cache → flash-attention (enables)` |
| `paper-chinchilla` | paper | `paper-chinchilla → pre-training` | No architecture `derived_from` chinchilla; correct as a scaling laws paper |

---

## SECTION 4 — SINK NODES (No Outgoing Edges)

These nodes have incoming edges but connect to nothing downstream.

| Node ID | Type | Incoming Edges | Note |
|---------|------|---------------|------|
| `dropout` | concept | `alexnet → dropout` | Should connect to `backpropagation (uses)` or `gradient-descent` |
| `paper-gpt3` | paper | `gpt → paper-gpt3` | Paper has no outgoing edges; should have `paper-gpt3 → gpt (implements)` |
| `max-pooling` | concept | `alexnet → max-pooling`, `vgg → max-pooling` | Terminal concept; acceptable as primitive |
| `latent-space` | concept | Multiple incoming | Terminal concept; should connect to `vae` or `diffusion-process` as source |

---

## SECTION 5 — CIRCULAR DEPENDENCIES

A circular dependency exists when A → B → ... → A.

| Cycle | Path | Relation Types | Severity |
|-------|------|---------------|---------|
| **batch-normalization ↔ layer-normalization** | `batch-normalization → layer-normalization (precedes)` AND `layer-normalization → batch-normalization (extends)` | precedes + extends | **Medium** — semantically inconsistent: if BatchNorm *precedes* LayerNorm (chronologically correct: 2015 vs 2016), then LayerNorm cannot *extend* BatchNorm in the `extends` sense without creating a loop. One edge should be removed or changed to `inspired_by`. |

**Total circular dependencies: 1**

---

## SECTION 6 — BROKEN REFERENCES

A broken reference is an edge where source or target ID does not exist in the nodes list.

**Result: 0 broken references found.**

All 164 edge source/target IDs resolve to valid node entries. The knowledge graph has clean referential integrity.

---

## SECTION 7 — MISSING RELATIONSHIPS

High-priority relationships that should exist but are absent.

### 7A — Missing Node Coverage (Architectures in codebase but not in KG)
These architecture MDX files have no corresponding node in KNOWLEDGE_GRAPH.json:

| Missing Node | Type | Should Connect To |
|-------------|------|-----------------|
| `densenet` | architecture | extends: `resnet`, uses: `residual-connection`, uses: `convolution` |
| `efficientnet` | architecture | extends: `resnet`, part_of: `domain-cv` |
| `googlenet` | architecture | extends: `alexnet`, uses: `convolution`, precedes: `resnet` |
| `lenet` | architecture | part_of: `domain-cv`, uses: `convolution`, precedes: `alexnet` |
| `swin` | architecture | extends: `vit`, part_of: `domain-cv` |
| `dino` | architecture | extends: `vit`, uses: `contrastive-learning` |
| `unet` | architecture | uses: `skip-connection`, uses: `convolution` |
| `t5` | architecture | extends: `transformer`, derived_from: `paper-attention` |
| `vgg16` | architecture | extends: `alexnet`, uses: `convolution` |
| `vgg19` | architecture | extends: `vgg16` |
| `inceptionv3` | architecture | extends: `googlenet` |

### 7B — Missing Paper Nodes (Papers in library but not in KG)
These paper MDX files have no corresponding paper node in KNOWLEDGE_GRAPH.json:

| Missing Paper Node | Slug in /papers/ | Should Connect To |
|-------------------|-----------------|-----------------|
| `paper-gpt` | `gpt` | `gpt (implements)` |
| `paper-gpt2` | `gpt-2` | `gpt (extends)`, `tokenization (explains)` |
| `paper-vgg` | `vgg` | `vgg (implements)` |
| `paper-alexnet` | `alexnet` | `alexnet (implements)` |
| `paper-batch-norm` | `batch-normalization` | `batch-normalization (implements)` |
| `paper-segment-anything` | `segment-anything` | `vit (uses)`, `clip (uses)` |
| `paper-palm` | `palm` | `transformer (derived_from)` |
| `paper-chinchilla` | already in KG | — |

### 7C — Missing Semantic Edges
Edges that should exist given current nodes:

| Source | Target | Relation | Reason |
|--------|--------|----------|--------|
| `domain-rl` | `rlhf` | part_of | Reinforcement Learning domain should contain RLHF |
| `domain-ml` | `gradient-descent` | part_of | Core ML concept |
| `domain-ml` | `transfer-learning` | part_of | Core ML concept |
| `domain-ml` | `backpropagation` | part_of | Core ML concept |
| `self-attention` | `scaled-dot-product` | extends | Self-attention IS the scaled dot-product mechanism |
| `self-attention` | `multi-head-attention` | part_of | Multi-head attention uses self-attention heads |
| `paper-gpt3` | `gpt` | implements | Bidirectional link is standard for all other paper→arch pairs |
| `flash-attention` | `kv-cache` | enables | FlashAttention makes large KV caches feasible |
| `stable-diffusion` | `unet` | uses | U-Net is the denoising backbone of Stable Diffusion |
| `vae` | `stable-diffusion` | enables | VAE enables the latent space SD operates in |
| `dpo` | `llama` | enables | DPO is a primary alignment method for LLaMA-family models |
| `dpo` | `fine-tuning` | extends | DPO is a fine-tuning technique |

---

## SECTION 8 — COVERAGE GAPS VS PLATFORM CONTENT

The KG covers 15 of the 31 architecture MDX files (48%) and 11 of the 19 paper MDX files (58%).

| Content Type | In Platform | In KG | Coverage |
|-------------|------------|-------|---------|
| Architectures | 31 MDX | 15 nodes | 48% |
| Papers | 19 MDX | 11 nodes | 58% |
| Topics | 1 authored | 1 node | 100% |
| Domains | 12 domain pages | 6 nodes | 50% |
| System Designs | 12 cases | 6 nodes | 50% |
| Problems | 22+110 | 6 nodes | ~2% |

**The knowledge graph covers the core spine of the platform (attention, transformers, LLMs, diffusion) but is missing coverage for:**
- All CNN-era architectures (DenseNet, EfficientNet, GoogLeNet, LeNet, VGG variants)
- Segmentation architectures (U-Net, FCN, DeepLabV3+)
- Self-supervised architectures (Swin, DINO)
- 8 of 19 paper nodes
- 6 of 12 domains
- 129 of 132 coding problems

---

## AUDIT SUMMARY

| Check | Result |
|-------|--------|
| Node count accuracy | Metadata overcounts by 1 (94 actual vs 95 claimed) |
| Edge count accuracy | Metadata overcounts by 23 (164 actual vs 187 claimed) |
| Isolated nodes | 3 (self-attention, domain-rl, domain-ml) |
| Circular dependencies | 1 (batch-normalization ↔ layer-normalization) |
| Broken references | 0 |
| Missing architecture nodes | 11 |
| Missing paper nodes | 7 |
| Missing semantic edges | 12 high-priority |
| Overall graph completeness | ~48–58% of platform content represented |
