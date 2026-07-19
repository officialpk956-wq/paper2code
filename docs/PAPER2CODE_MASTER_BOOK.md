# Paper2Code Master Book — 13-Volume Blueprint
**Generated:** 2026-06-20 | **Type:** Architecture blueprint only — NOT the book content itself

This document maps all existing and planned Paper2Code content into a 13-volume book structure.
Each entry shows: chapter title, source slug (if content exists), difficulty progression, estimated pages, and status.

---

## VOLUME STRUCTURE OVERVIEW

| Vol | Title | Chapters | Estimated Pages | Audience |
|-----|-------|----------|-----------------|---------|
| 1 | Foundations of Deep Learning | 8 | 280 | Beginner |
| 2 | Neural Network Architectures | 10 | 320 | Beginner–Intermediate |
| 3 | Attention and Transformers | 7 | 290 | Intermediate |
| 4 | Large Language Models | 9 | 350 | Intermediate–Advanced |
| 5 | Computer Vision | 8 | 280 | Intermediate |
| 6 | Generative Models | 7 | 300 | Advanced |
| 7 | Training and Optimization | 6 | 220 | Intermediate |
| 8 | Efficient Deep Learning | 6 | 240 | Advanced |
| 9 | AI Systems Engineering | 12 | 380 | Advanced |
| 10 | From Papers to Code | 9 | 340 | All levels |
| 11 | Mathematics for AI | 6 | 200 | Beginner–Intermediate |
| 12 | Research Frontiers | 8 | 310 | Advanced–Research |
| 13 | Interview Preparation | 5 | 180 | All levels |
| **TOTAL** | | **101** | **~3,690 pages** | |

---

## VOLUME 1 — Foundations of Deep Learning
*Prerequisites: None. Audience: Complete beginners.*

| Ch | Title | Topic Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 1.1 | What Is Machine Learning? | gradient-descent | Beginner | 25 | 🔴 Missing |
| 1.2 | Linear Algebra for Deep Learning | linear-algebra | Beginner | 40 | ✅ Content exists (math/linear-algebra) |
| 1.3 | Probability and Statistics | — | Beginner | 35 | 🔴 Missing |
| 1.4 | Calculus and Chain Rule | — | Beginner | 30 | 🔴 Missing |
| 1.5 | Gradient Descent and Optimization | gradient-descent | Beginner | 35 | 🔴 Missing |
| 1.6 | Loss Functions | loss-functions | Beginner | 25 | 🔴 Missing |
| 1.7 | Softmax and Probability Distributions | softmax | Beginner | 25 | ✅ Content exists (math/softmax) |
| 1.8 | Activation Functions | relu-activation, sigmoid-activation | Beginner | 30 | 🟡 Problem data exists, no topic page |

**Volume status: 2/8 chapters have content**

---

## VOLUME 2 — Neural Network Architectures
*Prerequisites: Volume 1. Audience: Beginners learning to build.*

| Ch | Title | Topic Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 2.1 | The Multi-Layer Perceptron | — | Beginner | 28 | 🔴 Missing |
| 2.2 | Backpropagation | backpropagation | Beginner | 35 | 🟡 Problem data only |
| 2.3 | Convolutional Neural Networks | convolutional-networks | Beginner | 38 | 🟡 Problem data only |
| 2.4 | AlexNet — The CNN That Started Everything | alexnet | Beginner | 30 | ✅ Architecture MDX exists |
| 2.5 | VGG — Depth With Uniform Kernels | vgg16 | Beginner | 28 | ✅ Architecture MDX exists |
| 2.6 | GoogLeNet / Inception — Multi-Scale Convolutions | googlenet | Intermediate | 32 | ✅ Architecture MDX exists |
| 2.7 | ResNet — Residual Connections | resnet | Intermediate | 40 | ✅ Architecture MDX + paper + implementation |
| 2.8 | DenseNet — Maximum Feature Reuse | densenet | Intermediate | 28 | ✅ Architecture MDX exists |
| 2.9 | EfficientNet — Neural Architecture Search | efficientnet | Intermediate | 28 | ✅ Architecture MDX exists |
| 2.10 | Sequence Models: RNN, LSTM, GRU | lstm, gru | Intermediate | 38 | ✅ Architecture MDX exists (all 3) |

**Volume status: 7/10 chapters have architecture content; topic pages missing**

---

## VOLUME 3 — Attention and Transformers
*Prerequisites: Volume 2. Audience: Intermediate learners.*

| Ch | Title | Topic Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 3.1 | The Attention Mechanism | attention | Intermediate | 55 | ✅ COMPLETE — full topic data exists |
| 3.2 | The Transformer Architecture | transformers | Intermediate | 55 | 🟡 Architecture MDX + paper; no topic page |
| 3.3 | BERT — Bidirectional Pre-Training | bert | Intermediate | 42 | ✅ Architecture + paper + implementation |
| 3.4 | GPT — Autoregressive Language Modeling | gpt | Intermediate | 42 | ✅ Architecture + paper + implementation |
| 3.5 | T5 — Text-to-Text Transfer Transformers | t5 | Intermediate | 35 | ✅ Architecture MDX exists |
| 3.6 | Positional Encoding and RoPE | positional-encoding, rope-embeddings | Advanced | 38 | 🟡 Problem data; no topic page |
| 3.7 | Layer Normalization and RMSNorm | layer-normalization | Intermediate | 25 | 🟡 Problem data only |

**Volume status: 4/7 have architecture/paper content; only `attention` has full topic page**

---

## VOLUME 4 — Large Language Models
*Prerequisites: Volume 3. Audience: Intermediate to Advanced.*

| Ch | Title | Topic Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 4.1 | Scaling Laws and Chinchilla | — | Advanced | 35 | ✅ Paper exists (chinchilla) |
| 4.2 | GPT-3 and Few-Shot Learning | — | Advanced | 40 | ✅ Paper exists (gpt-3) |
| 4.3 | LLaMA — Open-Weight LLMs | llama | Advanced | 45 | ✅ Architecture + paper + implementation |
| 4.4 | KV Cache and Inference Efficiency | kv-cache | Advanced | 38 | 🟡 Problem data; no topic page |
| 4.5 | Tokenization | tokenization | Intermediate | 28 | 🔴 Missing |
| 4.6 | Mixture of Experts | mixture-of-experts | Advanced | 42 | 🟡 Architecture MDX + paper; no topic page |
| 4.7 | RLHF — Alignment via Human Feedback | rlhf | Advanced | 45 | 🔴 Missing topic page |
| 4.8 | RAG — Retrieval-Augmented Generation | rag | Advanced | 45 | 🔴 Missing topic page |
| 4.9 | Reasoning Models and Chain-of-Thought | — | Research | 32 | 🔴 Missing |

**Volume status: 3/9 have partial content; 3 entirely missing**

---

## VOLUME 5 — Computer Vision
*Prerequisites: Volume 2. Audience: Intermediate.*

| Ch | Title | Topic Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 5.1 | Vision Transformers (ViT) | vision-transformers | Intermediate | 42 | ✅ Architecture + paper + implementation |
| 5.2 | Swin Transformer — Hierarchical ViT | swin | Advanced | 32 | ✅ Architecture MDX exists |
| 5.3 | DINO — Self-Supervised Vision | dino | Advanced | 30 | ✅ Architecture MDX exists |
| 5.4 | CLIP — Contrastive Image-Text | contrastive-learning | Advanced | 38 | ✅ Architecture + paper + implementation |
| 5.5 | Object Detection — YOLO | — | Intermediate | 32 | 🔴 Missing (catalog stub only) |
| 5.6 | Segmentation — U-Net and FCN | unet, fcn | Intermediate | 35 | ✅ Architecture MDX exists |
| 5.7 | Segment Anything Model (SAM) | — | Advanced | 30 | ✅ Paper exists (segment-anything) |
| 5.8 | DeepLab — Semantic Segmentation | deeplabv3plus | Advanced | 28 | ✅ Architecture MDX exists |

**Volume status: 6/8 have architecture/paper content**

---

## VOLUME 6 — Generative Models
*Prerequisites: Volume 2. Audience: Intermediate to Advanced.*

| Ch | Title | Topic Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 6.1 | Autoencoder | ae | Beginner | 25 | ✅ Architecture MDX exists |
| 6.2 | Variational Autoencoder (VAE) | vae | Intermediate | 38 | ✅ Architecture MDX exists |
| 6.3 | Generative Adversarial Networks (GAN) | gan | Intermediate | 45 | ✅ Architecture + paper + implementation |
| 6.4 | Diffusion Models — DDPM | diffusion-models | Advanced | 50 | 🟡 Architecture MDX; no topic page |
| 6.5 | Stable Diffusion — Latent Diffusion | — | Advanced | 45 | ✅ Architecture + paper + implementation |
| 6.6 | Score Matching and Flow Matching | — | Research | 35 | 🔴 Missing |
| 6.7 | PaLM and Multimodal Generation | — | Research | 30 | ✅ Paper exists (palm) |

**Volume status: 5/7 have architecture/paper content**

---

## VOLUME 7 — Training and Optimization
*Prerequisites: Volume 1–2. Audience: Intermediate.*

| Ch | Title | Topic Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 7.1 | Stochastic Gradient Descent | gradient-descent | Beginner | 30 | 🟡 Problem + interview content; no topic page |
| 7.2 | Adaptive Optimizers — Adam, AdamW | — | Intermediate | 28 | 🔴 Missing |
| 7.3 | Learning Rate Schedules | — | Intermediate | 25 | 🔴 Missing |
| 7.4 | Batch Normalization and Layer Norm | batch-normalization | Intermediate | 30 | ✅ Paper exists (batch-normalization) |
| 7.5 | Regularization — Dropout, Weight Decay | — | Intermediate | 25 | 🔴 Missing |
| 7.6 | Mixed Precision Training | — | Advanced | 28 | 🔴 Missing |

**Volume status: 2/6 have any content**

---

## VOLUME 8 — Efficient Deep Learning
*Prerequisites: Volumes 3–4. Audience: Advanced.*

| Ch | Title | Topic Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 8.1 | FlashAttention — IO-Aware Exact Attention | flash-attention | Advanced | 38 | 🔴 Missing topic page |
| 8.2 | KV Cache — Memory-Efficient Inference | kv-cache | Advanced | 35 | 🟡 Problem data only |
| 8.3 | Quantization — INT8 and INT4 | — | Advanced | 32 | 🔴 Missing |
| 8.4 | Grouped Query Attention (GQA, MQA) | — | Advanced | 30 | 🔴 Missing |
| 8.5 | Speculative Decoding | — | Research | 28 | 🔴 Missing |
| 8.6 | Model Pruning and Distillation | — | Advanced | 32 | 🔴 Missing |

**Volume status: 0/6 have complete content**

---

## VOLUME 9 — AI Systems Engineering
*Prerequisites: Volumes 3–4. Audience: Advanced engineers.*

| Ch | Title | System Design Slug | Difficulty | Est. Pages | Status |
|----|-------|------------------|-----------|-----------|--------|
| 9.1 | Designing ChatGPT | chatgpt-system-design | Advanced | 45 | ✅ System design MDX exists |
| 9.2 | LLM Serving — vLLM and Continuous Batching | — | Advanced | 38 | 🔴 Missing |
| 9.3 | Recommendation Systems — Netflix | netflix-recommendation | Intermediate | 38 | ✅ System design MDX exists |
| 9.4 | Recommendation Systems — TikTok | tiktok-recommendation | Intermediate | 35 | ✅ System design MDX exists |
| 9.5 | Recommendation Systems — YouTube | youtube-recommendation | Intermediate | 35 | ✅ System design MDX exists |
| 9.6 | Basic RAG System Design | basic-rag | Beginner | 32 | ✅ System design MDX exists |
| 9.7 | Advanced RAG | advanced-rag | Advanced | 38 | ✅ System design MDX exists |
| 9.8 | Agentic RAG | agentic-rag | Advanced | 35 | ✅ System design MDX exists |
| 9.9 | Single Agent Systems | single-agent | Intermediate | 32 | ✅ System design MDX exists |
| 9.10 | Multi-Agent Systems | multi-agent | Advanced | 35 | ✅ System design MDX exists |
| 9.11 | GitHub Copilot Architecture | github-copilot | Advanced | 35 | ✅ System design MDX exists |
| 9.12 | Perplexity AI Architecture | perplexity | Advanced | 32 | ✅ System design MDX exists |

**Volume status: 11/12 chapters have content — strongest volume**

---

## VOLUME 10 — From Papers to Code
*Prerequisites: Volumes 2–4. Audience: All levels.*

| Ch | Title | Impl. Slug | Difficulty | Est. Pages | Status |
|----|-------|-----------|-----------|-----------|--------|
| 10.1 | Implementing Attention Is All You Need | attention-is-all-you-need | Intermediate | 50 | ✅ Full implementation MDX exists |
| 10.2 | Implementing ResNet | resnet | Intermediate | 42 | ✅ Full implementation MDX exists |
| 10.3 | Implementing BERT | bert | Intermediate | 42 | ✅ Full implementation MDX exists |
| 10.4 | Implementing GPT | gpt | Intermediate | 45 | ✅ Full implementation MDX exists |
| 10.5 | Implementing Vision Transformer | vision-transformer | Intermediate | 40 | ✅ Full implementation MDX exists |
| 10.6 | Implementing LLaMA | llama | Advanced | 48 | ✅ Full implementation MDX exists |
| 10.7 | Implementing Stable Diffusion | stable-diffusion | Advanced | 48 | ✅ Full implementation MDX exists |
| 10.8 | Implementing GAN | gan | Intermediate | 38 | ✅ Full implementation MDX exists |
| 10.9 | Implementing CLIP | clip | Advanced | 42 | ✅ Full implementation MDX exists |

**Volume status: 9/9 chapters have content — second strongest volume**

---

## VOLUME 11 — Mathematics for AI
*Prerequisites: None. Audience: Beginners and intermediate learners needing mathematical foundation.*

| Ch | Title | Math Slug | Difficulty | Est. Pages | Status |
|----|-------|----------|-----------|-----------|--------|
| 11.1 | Linear Algebra Foundations | linear-algebra | Beginner | 45 | ✅ Content exists (math/linear-algebra) |
| 11.2 | Softmax and Probability Theory | softmax | Beginner | 32 | ✅ Content exists (math/softmax) |
| 11.3 | Calculus and Automatic Differentiation | — | Beginner | 35 | 🔴 Missing |
| 11.4 | Statistics for Deep Learning | — | Intermediate | 35 | 🔴 Missing |
| 11.5 | Information Theory — Entropy and KL Divergence | — | Intermediate | 30 | 🔴 Missing |
| 11.6 | Graph Theory for Knowledge Graphs | — | Intermediate | 30 | 🔴 Missing |

**Volume status: 2/6 chapters have content**

---

## VOLUME 12 — Research Frontiers
*Prerequisites: All previous volumes. Audience: Advanced and research-track learners.*

| Ch | Title | Topic | Difficulty | Est. Pages | Status |
|----|-------|-------|-----------|-----------|--------|
| 12.1 | Reasoning Models and o1/o3 | reasoning-models | Research | 35 | 🔴 Missing (topic in trending data) |
| 12.2 | Agentic AI and Multi-Agent Systems | agentic-ai | Research | 40 | 🔴 Missing |
| 12.3 | Mamba and State Space Models | mamba | Research | 38 | 🔴 Missing (catalog stub only) |
| 12.4 | RAG Evaluation with RAGAS | rag-evaluation | Research | 32 | 🔴 Missing |
| 12.5 | Constitutional AI and Safety | — | Research | 30 | 🔴 Missing |
| 12.6 | World Models and Embodied AI | — | Research | 35 | 🔴 Missing |
| 12.7 | Multimodal Foundation Models | — | Research | 38 | 🔴 Missing |
| 12.8 | Efficient Training — FSDP, ZeRO, DeepSpeed | — | Research | 40 | 🔴 Missing |

**Volume status: 0/8 chapters have content**

---

## VOLUME 13 — Interview Preparation
*Prerequisites: Volumes 1–10. Audience: Job seekers, all levels.*

| Ch | Title | Interview Slug | Difficulty | Est. Pages | Status |
|----|-------|--------------|-----------|-----------|--------|
| 13.1 | Explaining Attention in Interviews | explain-attention | Intermediate | 32 | ✅ Interview MDX exists |
| 13.2 | Gradient Descent and Optimization Questions | gradient-descent | Beginner | 30 | ✅ Interview MDX exists |
| 13.3 | System Design Interview Patterns | — | Advanced | 45 | 🔴 Missing |
| 13.4 | ML Coding Interview Problems | — | All | 42 | 🟡 Problem data exists, no interview-format page |
| 13.5 | Research Paper Walkthrough — Attention Is All You Need | — | Advanced | 35 | 🔴 Missing |

**Volume status: 2/5 chapters have content**

---

## PRODUCTION PRIORITY ORDER

Given current content, these volumes are closest to completion:

| Priority | Volume | Current Completeness | Gap |
|---------|--------|---------------------|-----|
| 1 | Vol. 10: Paper-to-Code | 9/9 chapters | 0 chapters missing |
| 2 | Vol. 9: AI Systems Engineering | 11/12 chapters | 1 chapter |
| 3 | Vol. 3: Attention & Transformers | 4/7 chapters have content | Need 3 topic pages |
| 4 | Vol. 2: Neural Network Architectures | 7/10 have MDX | Need topic pages |
| 5 | Vol. 5: Computer Vision | 6/8 have MDX | Need 2 more |
| 6 | Vol. 6: Generative Models | 5/7 have MDX | Need 2 more |
| 7 | Vol. 4: Large Language Models | 3/9 complete | Need 6 topic pages |
| 8 | Vol. 11: Mathematics | 2/6 complete | Need 4 chapters |
| 9 | Vol. 1: Foundations | 2/8 complete | Need 6 chapters |
| 10 | Vol. 13: Interview Prep | 2/5 complete | Need 3 chapters |
| 11 | Vol. 7: Training & Optimization | 2/6 complete | Need 4 chapters |
| 12 | Vol. 8: Efficient Deep Learning | 0/6 complete | All 6 missing |
| 13 | Vol. 12: Research Frontiers | 0/8 complete | All 8 missing |
