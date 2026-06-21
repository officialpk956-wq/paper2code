# Paper2Code Content Coverage Matrix
**Generated:** 2026-06-21 | **Phase:** Content Validation Audit

Coverage dimensions measured per topic. Scores are based on content that **currently exists** in the codebase — not what should exist.

**Dimensions:**
- **Theory** — Conceptual explanation of the topic with motivation and intuition
- **Math** — Key equations, formulas, or derivations
- **Visual** — Diagrams, animations, or visual explanations
- **Interactive** — Interactive component (lab, diagram, demo)
- **Code** — Step-by-step code walkthrough
- **PyTorch** — Working PyTorch implementation
- **Exercises** — Practice problems linked to this topic
- **Quiz** — Multiple-choice or short-answer questions
- **References** — Research paper citations/links
- **Interview** — Interview Q&A for this topic

**Score:** Y = full coverage | P = partial coverage | N = absent

---

## DOMAIN: deep-learning

### attention (✅ COMPLETE TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | **10/10 — 100%** |

**Sources:**
- Topic page: `src/data/topics/attention.ts` (200+ lines, all 13 sections complete)
- Formulas: Scaled dot-product, multi-head, softmax (with derivation steps, variable definitions)
- Interactive: archNodes/archEdges diagram data, renders as interactive flow
- PyTorch: `ScaledDotProductAttention`, `MultiHeadAttention`, `CachedMHA`, `GQA` classes
- Exercises: 4 linked problems (scaled-dot-product-attention, positional-encoding, multi-head-attention, masked-attention)
- Quiz: 10 interview Q&As in topic data (function as quiz questions)
- References: `attention-is-all-you-need` paper MDX linked
- Interview: `src/content/interview/explain-attention/content.mdx` (43 lines)

---

### transformers (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | P | N | Y | Y | Y | N | Y | N | **7/10 — 70%** |

**Sources:**
- Theory + Math: `src/content/architectures/transformer/content.mdx` (175 lines, 15 sections)
- Visual: MDX has diagram section but no interactive rendering — partial
- Code + PyTorch: `src/content/implementations/attention-is-all-you-need/content.mdx` (510 lines, full implementation)
- Exercises: `mini-transformer-block`, `masked-attention`, `positional-encoding` in problems.ts
- References: `src/content/papers/attention-is-all-you-need/content.mdx`
- **Missing:** Interactive component, quiz, interview Q&A

---

### backpropagation (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| P | P | N | N | Y | N | Y | N | N | N | **4/10 — 40%** |

**Sources:**
- Theory + Math: Partial coverage from architecture MDX context; `src/content/interview/gradient-descent/content.mdx` covers adjacent concepts
- Code: Problem solution code (backpropagation problem in PROBLEMS array)
- Exercises: `backpropagation` problem in problems.ts; `gradient-descent` problem
- **Missing:** Dedicated theory explainer, full math derivation, visual diagram, PyTorch autograd walkthrough, quiz, papers, interview prep

---

### convolutional-networks (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | P | N | Y | Y | Y | N | Y | N | **7/10 — 70%** |

**Sources:**
- Theory + Math: `src/content/architectures/alexnet/content.mdx`, `vgg16/content.mdx`, `resnet/content.mdx` cover convolution theory across multiple files
- Visual: Architecture MDX has visual description sections but no interactive rendering
- Code + PyTorch: `src/content/implementations/resnet/content.mdx` (279 lines, full PyTorch)
- Exercises: `convolution-2d`, `max-pooling`, `output-shape-calculation` in problems.ts
- References: `src/content/papers/alexnet/content.mdx`, `deep-residual-learning/content.mdx`
- **Missing:** Single unified theory page, interactive conv visualizer, quiz, interview prep

---

### residual-networks (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | P | N | Y | Y | N | N | Y | N | **6/10 — 60%** |

**Sources:**
- Theory + Math: `src/content/architectures/resnet/content.mdx` (177 lines)
- Visual: Architecture MDX has visual sections; Architecture Explorer has interactive diagram (ResNet is one of 5 complete)
- Code + PyTorch: `src/content/implementations/resnet/content.mdx` (279 lines)
- References: `src/content/papers/deep-residual-learning/content.mdx`, `batch-normalization/content.mdx`
- **Missing:** Dedicated topic page, exercises for skip connections, quiz, interview prep

---

### batch-normalization (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | N | N | N | N | N | N | Y | N | **3/10 — 30%** |

**Sources:**
- Theory + Math: `src/content/papers/batch-normalization/content.mdx` covers the paper's equations and methodology
- References: `src/content/papers/batch-normalization/content.mdx`
- **Missing:** Standalone theory explainer, interactive normalization demo, code walkthrough, PyTorch implementation guide, exercises, quiz, interview prep

---

## DOMAIN: machine-learning

### gradient-descent (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | N | N | Y | N | Y | N | N | Y | **5/10 — 50%** |

**Sources:**
- Theory + Math: `src/content/interview/gradient-descent/content.mdx` (55 lines)
- Code: Problem solution code available
- Exercises: `gradient-descent` problem in problems.ts
- Interview: `src/content/interview/gradient-descent/content.mdx`
- **Missing:** Visual learning rate animation, interactive optimizer comparison, PyTorch optimizer walkthrough, quiz, paper references (Adam, SGD with Momentum)

---

### loss-functions (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| N | N | N | N | Y | N | Y | N | N | N | **2/10 — 20%** |

**Sources:**
- Code: Problem solution code for cross-entropy
- Exercises: `cross-entropy-loss` problem in problems.ts
- **Missing:** Everything else — no dedicated content for MSE, BCE, focal loss, hinge loss, KL divergence, etc.

---

### transfer-learning (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| P | N | N | N | N | N | N | N | Y | N | **2/10 — 20%** |

**Sources:**
- Theory: Brief mentions in BERT/GPT architecture MDX; covered conceptually in Knowledge Graph
- References: `src/content/papers/bert/content.mdx`, `gpt/content.mdx`
- **Missing:** Dedicated explainer, fine-tuning walkthrough, PyTorch code, exercises, quiz

---

## DOMAIN: llms

### tokenization (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| N | N | N | N | N | N | N | N | P | N | **1/10 — 10%** |

**Sources:**
- References: Mentioned in `src/content/papers/gpt-2/content.mdx` and `bert/content.mdx` (partial)
- **Missing:** BPE algorithm explanation, SentencePiece walkthrough, vocabulary size analysis, code, exercises, interactive tokenizer demo

---

### kv-cache (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | N | N | N | Y | Y | Y | N | Y | P | **6/10 — 60%** |

**Sources:**
- Theory: `src/data/topics/attention.ts` production section covers KV cache deeply
- Code + PyTorch: `src/content/problems/gpt-kv-cache-scaling/content.mdx` (40 lines) + problem solution
- Exercises: `gpt-kv-cache-scaling` problem MDX; `prob-kv-cache` in problems.ts
- References: `src/content/papers/gpt-3/content.mdx` references KV cache
- Interview: Covered in attention topic's production section Q&As (partial)
- **Missing:** Standalone math (memory calculation formulas), visual memory layout diagram, quiz

---

### rope-embeddings (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | N | N | Y | Y | Y | N | Y | N | **6/10 — 60%** |

**Sources:**
- Theory + Math: `src/content/architectures/llama/content.mdx` (167 lines) covers RoPE in detail including rotation math
- Code + PyTorch: `src/content/problems/llama-rope/content.mdx` (7-line stub — header only) + problem solution in problems.ts
- Exercises: `llama-rope` problem; `prob-rope` in KG
- References: `src/content/papers/llama/content.mdx`
- **Missing:** Visual rotation animation, quiz, interview prep; note the problem MDX deep-dive is a 7-line stub

---

### mixture-of-experts (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | P | N | Y | Y | Y | N | Y | N | **7/10 — 70%** |

**Sources:**
- Theory + Math: `src/content/architectures/moe/content.mdx` (full 15 sections, router math)
- Visual: Architecture MDX has visual sections; partial
- Code + PyTorch: `src/content/problems/moe-routing/content.mdx` (7-line stub) + problem solution
- Exercises: `moe-routing` problem; `prob-moe` in KG
- References: `src/content/papers/switch-transformer/content.mdx`
- **Missing:** Interactive routing visualization, quiz, interview prep; problem MDX stub

---

### rlhf (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| P | N | N | N | N | N | N | N | N | N | **1/10 — 10%** |

**Sources:**
- Theory: Brief coverage in `src/content/system-design/chatgpt-system-design/content.mdx`; KG has `rlhf` concept node
- **Missing:** Reward model explanation, PPO walkthrough, preference dataset format, code, papers (no InstructGPT or PPO papers in library), exercises, quiz, interview prep

---

### rag (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | N | N | N | N | N | N | N | N | N | **1/10 — 10%** |

**Sources:**
- Theory: `src/content/system-design/basic-rag/content.mdx` (79L), `advanced-rag/content.mdx` (85L), `agentic-rag/content.mdx` (79L), `perplexity/content.mdx` (85L) — 4 system design cases covering RAG in depth from an architecture perspective
- **Missing:** Conceptual theory explainer (separate from system design), vector similarity math, embedding model explanation, code walkthrough (RAG pipeline in PyTorch/LangChain), exercises, references (no RAG paper in library), quiz

---

## DOMAIN: computer-vision

### vision-transformers (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | P | Y | P | P | Y | N | Y | N | **7/10 — 70%** |

**Sources:**
- Theory + Math: `src/content/architectures/vit/content.mdx` (173 lines, patch embedding math, CLS token)
- Visual: Architecture MDX visual sections; partial rendering
- Interactive: AI Lab for ViT (`/labs`) with real PyTorch shape inference
- Code + PyTorch: `src/content/implementations/vision-transformer/content.mdx` (122 lines, partial — no structured H2 sections)
- Exercises: `vit-patch-size` problem (7-line stub)
- References: `src/content/papers/vision-transformer/content.mdx`
- **Missing:** Complete implementation walkthrough, full interactive diagram, quiz, interview prep; ViT implementation MDX is minimal

---

### diffusion-models (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| Y | Y | P | Y | Y | Y | Y | N | Y | N | **8/10 — 80%** |

**Sources:**
- Theory + Math: `src/content/architectures/diffusion/content.mdx` + `stable-diffusion/content.mdx` (both full 15 sections); score matching math, forward/reverse process
- Visual: Architecture MDX visual sections; partial
- Interactive: AI Lab for Stable Diffusion (`/labs`) — real PyTorch shape inference across UNet/VAE/CLIP
- Code + PyTorch: `src/content/implementations/stable-diffusion/content.mdx` (433 lines, full PyTorch)
- Exercises: `stable-diffusion-cfg` problem (7-line stub)
- References: `src/content/papers/latent-diffusion-models/content.mdx`, `stable-diffusion/content.mdx`
- **Missing:** Quiz, interview prep, CFG problem MDX is a stub

---

### contrastive-learning (🔴 NO TOPIC PAGE)

| Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Score** |
|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|-----------|
| N | N | N | N | Y | Y | Y | N | Y | N | **4/10 — 40%** |

**Sources:**
- Code + PyTorch: `src/content/implementations/clip/content.mdx` (87 lines) — contains CLIP dual encoder, InfoNCE loss, training loop in `<Milestone>` components
- Exercises: `clip-batch-size` problem (7-line stub)
- References: `src/content/papers/clip/content.mdx`
- **Missing:** Theory explainer (CLIP architecture MDX is 35-line scaffold — all section bodies empty), InfoNCE math, visual embedding space diagram, interactive similarity demo, quiz, interview prep

---

## SUMMARY TABLE

| Topic | Domain | Theory | Math | Visual | Interactive | Code | PyTorch | Exercises | Quiz | References | Interview | **Coverage %** |
|-------|--------|--------|------|--------|------------|------|---------|----------|------|-----------|-----------|--------------|
| attention | deep-learning | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | **100%** |
| transformers | deep-learning | Y | Y | P | N | Y | Y | Y | N | Y | N | **70%** |
| backpropagation | deep-learning | P | P | N | N | Y | N | Y | N | N | N | **40%** |
| convolutional-networks | deep-learning | Y | Y | P | N | Y | Y | Y | N | Y | N | **70%** |
| residual-networks | deep-learning | Y | Y | P | N | Y | Y | N | N | Y | N | **60%** |
| batch-normalization | deep-learning | Y | Y | N | N | N | N | N | N | Y | N | **30%** |
| gradient-descent | machine-learning | Y | Y | N | N | Y | N | Y | N | N | Y | **50%** |
| loss-functions | machine-learning | N | N | N | N | Y | N | Y | N | N | N | **20%** |
| transfer-learning | machine-learning | P | N | N | N | N | N | N | N | Y | N | **20%** |
| tokenization | llms | N | N | N | N | N | N | N | N | P | N | **10%** |
| kv-cache | llms | Y | N | N | N | Y | Y | Y | N | Y | P | **60%** |
| rope-embeddings | llms | Y | Y | N | N | Y | Y | Y | N | Y | N | **60%** |
| mixture-of-experts | llms | Y | Y | P | N | Y | Y | Y | N | Y | N | **70%** |
| rlhf | llms | P | N | N | N | N | N | N | N | N | N | **10%** |
| rag | llms | Y | N | N | N | N | N | N | N | N | N | **10%** |
| vision-transformers | computer-vision | Y | Y | P | Y | P | P | Y | N | Y | N | **70%** |
| diffusion-models | computer-vision | Y | Y | P | Y | Y | Y | Y | N | Y | N | **80%** |
| contrastive-learning | computer-vision | N | N | N | N | Y | Y | Y | N | Y | N | **40%** |

### Platform Average Coverage: **47%**

| Fully scored (≥80%) | Mostly covered (50–79%) | Partial (30–49%) | Poor (≤20%) |
|--------------------|------------------------|-----------------|------------|
| attention (100%), diffusion-models (80%) | transformers (70%), CNNs (70%), MoE (70%), ViT (70%), ResNets (60%), GD (50%), KV-Cache (60%), RoPE (60%) | backpropagation (40%), contrastive-learning (40%), batch-norm (30%) | loss-functions (20%), transfer-learning (20%), tokenization (10%), rlhf (10%), rag (10%) |

**The biggest gaps are Exercises + Quiz + Interview dimensions, which are consistently missing outside the `attention` topic page.**
