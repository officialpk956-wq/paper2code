# Paper2Code Master Content Inventory
**Extracted:** 2026-06-20 | **Source:** All Claude Code sessions + codebase audit

This file catalogs every piece of verified content in the Paper2Code platform.
Content status is derived from actual files in `src/content/`, `src/data/`, and `static/`.

---

## SECTION 1 — ARCHITECTURE CONTENT
> Location: `src/content/architectures/<slug>/content.mdx`

| # | Title | Slug | Category | Year | Status | Completeness |
|---|-------|------|----------|------|--------|--------------|
| 1 | Transformer | transformer | Transformers & LLMs | 2017 | Complete | 100% |
| 2 | ResNet | resnet | Convolutional Networks | 2015 | Complete | 100% |
| 3 | BERT | bert | Transformers & LLMs | 2018 | Complete | 100% |
| 4 | GPT | gpt | Transformers & LLMs | 2020 | Complete | 100% |
| 5 | ViT | vit | Vision | 2020 | Complete | 100% |
| 6 | AlexNet | alexnet | Convolutional Networks | 2012 | MDX only | 70% |
| 7 | VGG-16 | vgg16 | Convolutional Networks | 2014 | MDX only | 70% |
| 8 | VGG-19 | vgg19 | Convolutional Networks | 2014 | MDX only | 70% |
| 9 | GoogLeNet | googlenet | Convolutional Networks | 2014 | MDX only | 70% |
| 10 | Inception-v3 | inceptionv3 | Convolutional Networks | 2015 | MDX only | 70% |
| 11 | DenseNet | densenet | Convolutional Networks | 2016 | MDX only | 70% |
| 12 | EfficientNet | efficientnet | Efficient & Emerging | 2019 | MDX only | 70% |
| 13 | U-Net | unet | Vision | 2015 | MDX only | 70% |
| 14 | Swin Transformer | swin | Vision | 2021 | MDX only | 70% |
| 15 | CLIP | clip | Vision | 2021 | MDX only | 70% |
| 16 | DINO | dino | Vision | 2021 | MDX only | 70% |
| 17 | DeepLabV3+ | deeplabv3plus | Vision | 2018 | MDX only | 70% |
| 18 | FCN | fcn | Vision | 2014 | MDX only | 70% |
| 19 | LeLet | lenet | Convolutional Networks | 1998 | MDX only | 70% |
| 20 | RNN | rnn | Sequence Models | 1986 | MDX only | 70% |
| 21 | GRU | gru | Sequence Models | 2014 | MDX only | 70% |
| 22 | LSTM | lstm | Sequence Models | 1997 | MDX only | 70% |
| 23 | Seq2Seq | seq2seq | Sequence Models | 2014 | MDX only | 70% |
| 24 | Autoencoder | ae | Generative Models | 2006 | MDX only | 70% |
| 25 | VAE | vae | Generative Models | 2013 | MDX only | 70% |
| 26 | GAN | gan | Generative Models | 2014 | MDX only | 70% |
| 27 | Diffusion (DDPM) | diffusion | Generative Models | 2020 | MDX only | 70% |
| 28 | Stable Diffusion | stable-diffusion | Generative Models | 2022 | MDX only | 70% |
| 29 | T5 | t5 | Transformers & LLMs | 2019 | MDX only | 70% |
| 30 | LLaMA | llama | Transformers & LLMs | 2023 | MDX only | 70% |
| 31 | Mixture of Experts | moe | Efficient & Emerging | 2022 | MDX only | 70% |

**Catalog-only entries (no MDX, stub in architecture-catalog.ts):**

| # | Title | Slug | Category | Year | Status |
|---|-------|------|----------|------|--------|
| 32 | YOLO | yolo | Vision | 2016 | Stub |
| 33 | RoBERTa | roberta | Transformers & LLMs | 2019 | Stub |
| 34 | GPT-2 | gpt-2-arch | Transformers & LLMs | 2019 | Stub |
| 35 | Mamba | mamba | Efficient & Emerging | 2023 | Stub |

**Architecture Explorer UI:** 5 complete (with diagrams + math snippets + code snippets) + 26 with MDX + 4 stubs = **35 total**

---

## SECTION 2 — PAPER LIBRARY
> Location: `src/content/papers/<slug>/content.mdx`

| # | Title | Slug | Year | Domain | Status | Completeness |
|---|-------|------|------|--------|--------|--------------|
| 1 | ImageNet Classification with Deep CNNs (AlexNet) | alexnet | 2012 | Vision | Complete | 100% |
| 2 | Very Deep Convolutional Networks (VGG) | vgg | 2014 | Vision | Complete | 100% |
| 3 | Generative Adversarial Networks | gan | 2014 | Generative | Complete | 100% |
| 4 | Deep Residual Learning for Image Recognition | deep-residual-learning | 2015 | Vision | Complete | 100% |
| 5 | Batch Normalization | batch-normalization | 2015 | Foundations | Complete | 100% |
| 6 | Attention Is All You Need | attention-is-all-you-need | 2017 | NLP/Transformers | Complete | 100% |
| 7 | BERT | bert | 2018 | NLP | Complete | 100% |
| 8 | Improving Language Understanding (GPT) | gpt | 2018 | LLMs | Complete | 100% |
| 9 | Language Models are Unsupervised Multitask Learners (GPT-2) | gpt-2 | 2019 | LLMs | Complete | 100% |
| 10 | Language Models are Few-Shot Learners (GPT-3) | gpt-3 | 2020 | LLMs | Complete | 100% |
| 11 | An Image is Worth 16×16 Words (ViT) | vision-transformer | 2020 | Vision | Complete | 100% |
| 12 | Switch Transformers | switch-transformer | 2021 | LLMs/MoE | Complete | 100% |
| 13 | Training Compute-Optimal LLMs (Chinchilla) | chinchilla | 2022 | LLMs | Complete | 100% |
| 14 | PaLM | palm | 2022 | LLMs | Complete | 100% |
| 15 | CLIP | clip | 2021 | Multimodal | Complete | 100% |
| 16 | High-Resolution Image Synthesis (Latent Diffusion) | latent-diffusion-models | 2022 | Generative | Complete | 100% |
| 17 | Stable Diffusion | stable-diffusion | 2022 | Generative | Complete | 100% |
| 18 | Segment Anything | segment-anything | 2023 | Vision | Complete | 100% |
| 19 | LLaMA | llama | 2023 | LLMs | Complete | 100% |

**Total papers with MDX: 19**

---

## SECTION 3 — IMPLEMENTATIONS (Paper-to-Code)
> Location: `src/content/implementations/<slug>/content.mdx`

| # | Title | Slug | Source Paper | Language | Status | Completeness |
|---|-------|------|-------------|----------|--------|--------------|
| 1 | Attention Is All You Need — Implementation | attention-is-all-you-need | attention-is-all-you-need | PyTorch | Complete | 100% |
| 2 | ResNet — Implementation | resnet | deep-residual-learning | PyTorch | Complete | 100% |
| 3 | BERT — Implementation | bert | bert | PyTorch | Complete | 100% |
| 4 | GPT — Implementation | gpt | gpt | PyTorch | Complete | 100% |
| 5 | GAN — Implementation | gan | gan | PyTorch | Complete | 100% |
| 6 | Stable Diffusion — Implementation | stable-diffusion | stable-diffusion | PyTorch | Complete | 100% |
| 7 | Vision Transformer — Implementation | vision-transformer | vision-transformer | PyTorch | Complete | 100% |
| 8 | LLaMA — Implementation | llama | llama | PyTorch | Complete | 100% |
| 9 | CLIP — Implementation | clip | clip | PyTorch | Complete | 100% |

**Total implementations: 9**

---

## SECTION 4 — SYSTEM DESIGN CASES
> Location: `src/content/system-design/<slug>/content.mdx`

| # | Title | Slug | Category | Difficulty | Status | Completeness |
|---|-------|------|----------|-----------|--------|--------------|
| 1 | ChatGPT System Design | chatgpt-system-design | LLM Serving | Advanced | Complete | 100% |
| 2 | Recommendation Engine | recommendation-engine | RecSys | Intermediate | Complete | 100% |
| 3 | GitHub Copilot | github-copilot | LLM Serving | Advanced | Complete | 100% |
| 4 | Perplexity AI | perplexity | RAG Systems | Advanced | Complete | 100% |
| 5 | Basic RAG | basic-rag | RAG Systems | Beginner | Complete | 100% |
| 6 | Netflix Recommendation | netflix-recommendation | RecSys | Intermediate | Complete | 100% |
| 7 | TikTok Recommendation | tiktok-recommendation | RecSys | Intermediate | Complete | 100% |
| 8 | Single Agent System | single-agent | Agent Systems | Intermediate | Complete | 100% |
| 9 | Multi-Agent System | multi-agent | Agent Systems | Advanced | Complete | 100% |
| 10 | YouTube Recommendation | youtube-recommendation | RecSys | Intermediate | Complete | 100% |
| 11 | Advanced RAG | advanced-rag | RAG Systems | Advanced | Complete | 100% |
| 12 | Agentic RAG | agentic-rag | RAG Systems | Advanced | Complete | 100% |

**Total system design cases: 12**

---

## SECTION 5 — CODING PROBLEMS
### 5A. Next.js Dojo Problems (src/data/problems.ts + src/content/problems/)

| # | Title | Slug | Category | Difficulty | Has MDX | Status |
|---|-------|------|----------|-----------|---------|--------|
| 1 | Implement Matrix Multiplication | matrix-multiplication | linear-algebra | Beginner | Yes | Complete |
| 2 | Softmax Implementation | softmax | linear-algebra | Beginner | No | Complete |
| 3 | Cosine Similarity | cosine-similarity | linear-algebra | Beginner | No | Complete |
| 4 | Dot Product Basics | dot-product-basic | linear-algebra | Beginner | No | Complete |
| 5 | Batch Matrix Multiply | batch-matrix-multiply | linear-algebra | Beginner | No | Complete |
| 6 | Sigmoid Activation | sigmoid-activation | deep-learning | Beginner | No | Complete |
| 7 | ReLU Activation | relu-activation | deep-learning | Beginner | No | Complete |
| 8 | Cross-Entropy Loss | cross-entropy-loss | deep-learning | Intermediate | No | Complete |
| 9 | Gradient Descent | gradient-descent | deep-learning | Intermediate | No | Complete |
| 10 | Backpropagation | backpropagation | deep-learning | Intermediate | No | Complete |
| 11 | 2D Convolution | convolution-2d | cnn | Intermediate | No | Complete |
| 12 | Max Pooling | max-pooling | cnn | Beginner | No | Complete |
| 13 | Output Shape Calculation | output-shape-calculation | cnn | Beginner | No | Complete |
| 14 | Scaled Dot-Product Attention | scaled-dot-product-attention | transformer | Intermediate | Yes | Complete |
| 15 | Positional Encoding | positional-encoding | transformer | Intermediate | No | Complete |
| 16 | Multi-Head Attention | multi-head-attention | transformer | Intermediate | No | Complete |
| 17 | Masked Attention | masked-attention | transformer | Intermediate | No | Complete |
| 18 | Layer Normalization | layer-normalization | transformer | Beginner | No | Complete |
| 19 | Top-K Sampling | top-k-sampling | llm-engineering | Intermediate | No | Complete |
| 20 | KV Cache | kv-cache | llm-engineering | Intermediate | No | Complete |
| 21 | Dot Product | dot-product | linear-algebra | Beginner | No | Complete |
| 22 | Mini Transformer Block | mini-transformer-block | transformer | Advanced | No | Complete |

**Additional problem MDX files (deeper write-ups):**
- attention-calculation, gpt-kv-cache-scaling, llama-rope, moe-routing, clip-batch-size, vit-patch-size, stable-diffusion-cfg

### 5B. Static Dojo Problems (static/index.html — Python execution)
110 DS/ML coding problems across 5 categories. Executed via Python backend API. Stored as JS objects in static/index.html.

---

## SECTION 6 — TOPIC PAGES
> Location: `src/data/topics/<slug>.ts` + `src/app/learn/[domain]/[topic]/page.tsx`

| # | Title | Slug | Domain | Difficulty | Status | Completeness |
|---|-------|------|--------|-----------|--------|--------------|
| 1 | Attention Mechanism | attention | deep-learning | Intermediate | Complete | 100% |

**Note:** All other topic slugs return "Topic Not Found" from registry. The `attention` topic has full 13-section data (motivation, intuition, formula, derivation, code, variants, production, tradeoffs, summary, problems, interview).

---

## SECTION 7 — LEARNING DOMAINS
> Location: `src/data/domains/<slug>.ts`

| # | Domain | Slug | Stages | Clusters | Lessons | Papers | Status | Completeness |
|---|--------|------|--------|---------|---------|--------|--------|--------------|
| 1 | Deep Learning | deep-learning | 8 | 4 | 6 | 5 | Complete | 100% |
| 2 | Machine Learning | machine-learning | ~8 | ~4 | ~6 | ~5 | Complete | 90% |
| 3 | Large Language Models | llms | ~8 | ~4 | ~6 | ~5 | Complete | 90% |
| 4 | Computer Vision | computer-vision | — | — | — | — | Fallback only | 15% |
| 5 | NLP | nlp | — | — | — | — | Fallback only | 15% |
| 6 | Reinforcement Learning | reinforcement-learning | — | — | — | — | Fallback only | 15% |
| 7 | Statistics | statistics | — | — | — | — | Fallback only | 15% |
| 8 | Mathematics | mathematics | — | — | — | — | Fallback only | 15% |
| 9 | MLOps | mlops | — | — | — | — | Fallback only | 15% |
| 10 | AI Systems | ai-systems | — | — | — | — | Fallback only | 15% |
| 11 | Research Methods | research-methods | — | — | — | — | Fallback only | 15% |
| 12 | Robotics | robotics | — | — | — | — | Fallback only | 15% |

---

## SECTION 8 — MATH CONTENT
> Location: `src/content/math/<slug>/content.mdx`

| # | Title | Slug | Difficulty | Status | Completeness |
|---|-------|------|-----------|--------|--------------|
| 1 | Linear Algebra | linear-algebra | Beginner | Complete | 100% |
| 2 | Softmax | softmax | Beginner | Complete | 100% |

---

## SECTION 9 — INTERVIEW PREP
> Location: `src/content/interview/<slug>/content.mdx`

| # | Title | Slug | Topic | Status |
|---|-------|------|-------|--------|
| 1 | Explain Attention | explain-attention | Attention Mechanism | Complete |
| 2 | Gradient Descent | gradient-descent | Optimization | Complete |

---

## SECTION 10 — ROADMAPS
> Location: `src/content/roadmaps/<slug>/content.mdx`

| # | Title | Slug | Status |
|---|-------|------|--------|
| 1 | AI Engineer Path | ai-engineer | Complete |

---

## SECTION 11 — INTERACTIVE FEATURES (Non-MDX)

| Feature | Route | Status | Description |
|---------|-------|--------|-------------|
| Architecture Explorer | /architectures | Complete | 35 arch entries, 5 with full diagrams/code/math |
| Code Dojo (Next.js) | /dojo | Complete | 22 problems, Monaco editor, submissions history |
| Code Dojo (Static) | /dojo (static) | Complete | 110 Python problems, Monaco, real execution |
| AI Labs | /labs | Complete | 4 labs: Transformer/CNN/ViT/Diffusion |
| Block Visualizer | /block-viz | Complete | PyTorch hook-based 3-level block hierarchy |
| Paper Upload | /papers/upload | Complete | PDF → Knowledge Graph pipeline |
| Research Hub | /papers | Complete | 4-tab research hub with papers/implementations/collections |
| Knowledge Graph Viewer | /papers/upload/[id] | Complete | Interactive SVG with 38 concept mappings |
| Learn Domain Pages | /learn/[domain] | Partial | 3 of 12 domains authored |
| Learn Topic Pages | /learn/[domain]/[topic] | Partial | 1 of ~100 planned topics authored |

---

## SECTION 12 — ENGINEERING DOCUMENTATION

| Document | Location | Status | Description |
|----------|----------|--------|-------------|
| Learning Curriculum | docs/PAPER2CODE_LEARNING_CURRICULUM.md | Complete | 9-phase build roadmap, 19 knowledge prerequisites |
| Engineering Handbook | docs/PAPER2CODE_ENGINEERING_HANDBOOK.md | Complete | 18 sections, 200 Q&As, architecture deep dive |
| Phase Reports | PHASE_1 through PHASE_16F | Complete | 20+ implementation reports |
| Telemetry Docs | .telemetry/current-implementation.md | Complete | Analytics/tracking architecture |

---

## SUMMARY COUNTS

| Content Type | Total | Complete | Partial | Stub/Missing |
|-------------|-------|---------|---------|-------------|
| Architectures (with MDX) | 31 | 5 | 26 | 4 |
| Papers | 19 | 19 | 0 | 0 |
| Implementations | 9 | 9 | 0 | 0 |
| System Design Cases | 12 | 12 | 0 | 0 |
| Coding Problems (Next.js) | 22 | 22 | 0 | 0 |
| Coding Problems (Static) | 110 | 110 | 0 | 0 |
| Topic Pages | 1 | 1 | 0 | ~99 missing |
| Learning Domains | 12 | 3 | 0 | 9 |
| Math Content | 2 | 2 | 0 | 0 |
| Interview Prep | 2 | 2 | 0 | 0 |
| Roadmaps | 1 | 1 | 0 | 0 |
| **TOTAL** | **231** | **186** | **26** | **19** |
