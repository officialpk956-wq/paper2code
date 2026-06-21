# Paper2Code Content Duplicates & Alias Resolution
**Extracted:** 2026-06-20 | **Source:** Codebase audit across src/content/, src/data/, static/

This file identifies duplicate content, aliased slugs, naming collisions, and canonical merge targets.

---

## DUPLICATE GROUP 1: Attention / Transformer Mechanism

**Aliases found:**
- `attention` (topic slug, `src/data/topics/attention.ts`)
- `attention-mechanism` (referenced in problem tags)
- `multi-head-attention` (problem slug, `data/problems.ts`)
- `scaled-dot-product-attention` (problem slug)
- `self-attention` (tag in topic data)
- `explain-attention` (interview content slug)
- `attention-is-all-you-need` (paper slug)
- `attention-calculation` (problem MDX slug)

**Canonical slug:** `attention`
**Canonical title:** "Attention Mechanism"
**Merge action:** All references that mean "the attention concept" should resolve to `/learn/deep-learning/attention`. Sub-topics (multi-head, masked, cross-attention) should be sub-sections of this page, not separate slugs.

---

## DUPLICATE GROUP 2: Transformer Architecture vs Topic

**Aliases found:**
- `transformer` (architecture slug, `architecture-catalog.ts`)
- `transformer` (topic tag)
- `transformers` (domain section header)
- `Transformers & LLMs` (architecture category label)
- `transformer-architecture` (referenced in engineering docs)

**Canonical slug:** `transformer` for architecture (`/architectures/transformer`), `transformers` for topic area
**Merge action:** Architecture page = `/architectures/transformer`. Topic page = `/learn/deep-learning/transformers`. Do not create a topic page that duplicates the architecture page content.

---

## DUPLICATE GROUP 3: GPT Family

**Aliases found:**
- `gpt` (architecture slug + implementation slug + paper slug)
- `gpt-2` (paper slug)
- `gpt-2-arch` (architecture-catalog stub slug — COLLISION with paper `gpt-2`)
- `gpt-3` (paper slug)
- `gpt` (multiple usages for GPT-1 and GPT in general)

**Canonical resolution:**
- Paper `gpt` → GPT-1 (2018 OpenAI paper)
- Paper `gpt-2` → GPT-2 (2019)
- Paper `gpt-3` → GPT-3 (2020)
- Architecture `gpt` → general GPT decoder architecture
- Architecture stub `gpt-2-arch` → RENAME to `gpt-2` if MDX is ever created (conflicts with paper slug — needs namespace resolution)

**Action required:** The architecture catalog uses `gpt-2-arch` to avoid collision with paper slug `gpt-2`. This is a known intentional workaround. Document and keep.

---

## DUPLICATE GROUP 4: ResNet / Deep Residual Learning

**Aliases found:**
- `resnet` (architecture slug + implementation slug)
- `deep-residual-learning` (paper slug)
- `residual-connection` (concept in CONCEPT_META, PaperKnowledgeGraph.tsx)
- `ResNet` (architecture title)
- `dl-1` through `dl-5` (problem IDs that link to `deep-residual-learning` paper)

**Canonical slugs:**
- Architecture: `resnet` → `/architectures/resnet`
- Paper: `deep-residual-learning` → `/papers/deep-residual-learning`
- Concept: `residual-connection` maps to `/architectures/resnet` in CONCEPT_META

---

## DUPLICATE GROUP 5: VGG Variants

**Aliases found:**
- `vgg` (paper slug)
- `vgg16` (architecture MDX slug, `src/content/architectures/vgg16/`)
- `vgg19` (architecture MDX slug, `src/content/architectures/vgg19/`)
- `vgg` (architecture catalog stub slug)

**Problem:** Architecture catalog uses slug `vgg` as a stub, but actual MDX content lives at `vgg16` and `vgg19` — no unified `vgg` architecture MDX.

**Resolution:**
- Paper `vgg` = the VGGNet paper (covers both variants) — keep
- Architecture: create a `vgg` parent page or rename MDX dirs to match catalog slugs
- `vgg16` and `vgg19` can be sub-entries or variants within a single `vgg` architecture page

---

## DUPLICATE GROUP 6: Diffusion Models

**Aliases found:**
- `diffusion` (architecture MDX slug — `src/content/architectures/diffusion/`)
- `ddpm` (architecture catalog stub slug)
- `stable-diffusion` (architecture MDX slug + paper slug + implementation slug)
- `latent-diffusion-models` (paper slug)
- `diffusion-models` (trending topic ID in `src/data/learn/topics.ts`)

**Resolution:**
- `ddpm` catalog entry → points to `diffusion` MDX (architecture page covers DDPM)
- `stable-diffusion` = separate architecture page, distinct from base DDPM
- `latent-diffusion-models` = the research paper; `stable-diffusion` = the derived product
- `diffusion-models` as trending topic = refers to the domain, links to a future `/learn/computer-vision/diffusion-models` topic

---

## DUPLICATE GROUP 7: GAN / Generative Adversarial Networks

**Aliases found:**
- `gan` (architecture MDX + paper slug + implementation slug — ALL use same slug)
- `adversarial-training` (referenced in tags)
- `generative-adversarial-networks` (full title used in docs)

**Status:** No actual collision — `gan` is used consistently in all three namespaces (paper/architecture/implementation). This is safe. Document as intentional.

---

## DUPLICATE GROUP 8: CLIP

**Aliases found:**
- `clip` (architecture MDX slug)
- `clip` (architecture catalog stub slug)
- `clip` (implementation slug)
- `clip` (paper slug)
- `paper-to-code/clip` (navigation href in left rail)

**Status:** Same slug `clip` used across paper/architecture/implementation. Safe because routing namespace separates them (`/papers/clip` vs `/architectures/clip` vs `/paper-to-code/clip`). However, `paper-to-code/clip` and `/implementations/clip` may be the same content under different routes — needs audit.

---

## DUPLICATE GROUP 9: LLaMA

**Aliases found:**
- `llama` (architecture MDX slug)
- `llama` (paper slug)
- `llama` (implementation slug)
- `llama` (architecture catalog stub slug — WRONG: MDX exists at `src/content/architectures/llama/`)

**Action:** Architecture catalog entry for `llama` is marked `coming-soon` but MDX already exists at `src/content/architectures/llama/content.mdx`. Update catalog entry status to `complete`.

---

## DUPLICATE GROUP 10: MoE / Mixture of Experts

**Aliases found:**
- `moe` (architecture MDX slug + catalog stub slug)
- `mixture-of-experts` (trending topic ID)
- `switch-transformer` (paper that introduces MoE for transformers)
- `sparse-moe` (was a dead link in recommendations — now removed)

**Resolution:** `moe` = architecture page. `mixture-of-experts` = trending topic in `/learn/llms/mixture-of-experts` (not yet built). `switch-transformer` = paper page. No collision.

---

## NAMING COLLISION TABLE (Needs Action)

| Slug | Conflict Type | Namespaces | Recommended Action |
|------|--------------|------------|-------------------|
| `gpt-2` | slug collision | paper vs architecture-catalog | Keep `gpt-2-arch` in catalog as workaround |
| `vgg` | missing MDX | catalog has stub, no matching MDX | Create `vgg` MDX or point catalog to `vgg16` |
| `llama` | status mismatch | catalog says coming-soon, MDX exists | Update catalog status to `complete` |
| `clip` | multi-namespace | paper + architecture + implementation | Already handled by route namespace, document |
| `gan` | multi-namespace | paper + architecture + implementation | Already handled by route namespace, document |

---

## SAFE ALIASES (No Action Needed)

| Context | Aliases | Canonical | Notes |
|---------|---------|-----------|-------|
| Attention | self-attention, multi-head-attention, cross-attention | `attention` | Sub-topics of same concept |
| ResNet | residual network, residual connection | `resnet` | Alt names for same thing |
| ViT | vision transformer | `vit` | Official abbreviation |
| BERT | bidirectional encoder | `bert` | Official abbreviation |
| DDPM | diffusion model | `diffusion` | MDX covers DDPM |
