# Paper2Code Course Generation Map
**Extracted:** 2026-06-20 | **Source:** Codebase audit + curriculum design

This file specifies the `topic.yaml` schema for every topic in the Paper2Code learning platform.
Topics marked `MISSING` have no authored data in `src/data/topics/` yet.

---

## YAML SCHEMA REFERENCE

```yaml
# topic.yaml specification
slug: string               # URL slug: /learn/[domain]/[slug]
title: string              # Display title
subtitle: string           # One-sentence description
difficulty: beginner | intermediate | advanced
duration_minutes: number
domain_slug: string        # Parent domain slug
prerequisites: string[]    # List of topic titles or concept names
tags: string[]
status: complete | partial | missing

# Sections (all 13 required for complete status)
sections:
  motivation:              # Why this topic exists
  intuition:               # Concrete real-world example
  formula:                 # Core mathematical expression
  derivation:              # Step-by-step math proof
  code:                    # PyTorch implementation
  variants:                # Related techniques / extensions
  production:              # Real-world engineering considerations
  tradeoffs:               # When to use / when not to
  common_mistakes:         # Errors beginners make
  interview_questions:     # 5+ Q&As
  practice_problems:       # Links to /dojo/[slug]
  related_papers:          # Links to /papers/[slug]
  summary:                 # Key takeaways

# Content connections
practice_problem_slugs: string[]    # From PROBLEMS array
related_paper_slugs: string[]       # From src/content/papers/
related_architecture_slugs: string[]
```

---

## DOMAIN: deep-learning

### topic: attention ✅ COMPLETE
```yaml
slug: attention
title: Attention Mechanism
subtitle: The core building block that enabled modern Transformers and LLMs.
difficulty: intermediate
duration_minutes: 45
domain_slug: deep-learning
prerequisites:
  - Linear Algebra
  - Softmax
  - Matrix Multiplication
  - Neural Networks
tags:
  - Transformers
  - NLP
  - Self-Attention
  - Multi-Head Attention
status: complete
sections:
  motivation: complete      # Sequential bottleneck, vanishing context, direct connections
  intuition: complete       # "The animal didn't cross..." sentence example
  formula: complete         # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
  derivation: complete      # sqrt(d_k) variance proof, multi-head parameter count
  code: complete            # scaled_dot_product_attention, MultiHeadAttention, CachedMHA, GQA
  variants: complete        # Self/Cross/Causal, MHA, MQA, GQA, FlashAttention
  production: complete      # KV cache, continuous batching, PagedAttention, vLLM
  tradeoffs: complete       # O(n²) complexity, FlashAttention IO, memory vs compute
  common_mistakes: complete # 8 documented mistakes with fixes
  interview_questions: complete  # 10 Q&As
  practice_problems: complete    # scaled-dot-product-attention, positional-encoding, multi-head-attention, layer-normalization
  related_papers: complete       # attention-is-all-you-need
  summary: complete
practice_problem_slugs:
  - scaled-dot-product-attention
  - positional-encoding
  - multi-head-attention
  - masked-attention
  - layer-normalization
related_paper_slugs:
  - attention-is-all-you-need
related_architecture_slugs:
  - transformer
  - bert
  - gpt
  - vit
  - llama
```

### topic: transformers 🔴 MISSING
```yaml
slug: transformers
title: The Transformer Architecture
subtitle: Encoder-decoder attention that replaced recurrence and changed everything.
difficulty: intermediate
duration_minutes: 60
domain_slug: deep-learning
prerequisites:
  - Attention Mechanism
  - Positional Encoding
  - Layer Normalization
  - Feed-Forward Networks
tags:
  - Architecture
  - Encoder-Decoder
  - NLP
status: missing
practice_problem_slugs:
  - mini-transformer-block
  - masked-attention
  - positional-encoding
related_paper_slugs:
  - attention-is-all-you-need
related_architecture_slugs:
  - transformer
  - bert
  - gpt
```

### topic: backpropagation 🔴 MISSING
```yaml
slug: backpropagation
title: Backpropagation
subtitle: How neural networks learn by flowing gradients backward through the computation graph.
difficulty: intermediate
duration_minutes: 50
domain_slug: deep-learning
prerequisites:
  - Calculus (Chain Rule)
  - Linear Algebra
  - Neural Networks
  - Loss Functions
tags:
  - Optimization
  - Gradients
  - Training
status: missing
practice_problem_slugs:
  - backpropagation
  - gradient-descent
  - cross-entropy-loss
related_paper_slugs: []
related_architecture_slugs:
  - resnet
```

### topic: convolutional-networks 🔴 MISSING
```yaml
slug: convolutional-networks
title: Convolutional Neural Networks
subtitle: Spatial pattern recognition via weight sharing and translation equivariance.
difficulty: beginner
duration_minutes: 40
domain_slug: deep-learning
prerequisites:
  - Linear Algebra
  - Backpropagation
  - Activation Functions
tags:
  - CNN
  - Vision
  - Convolution
status: missing
practice_problem_slugs:
  - convolution-2d
  - max-pooling
  - output-shape-calculation
related_paper_slugs:
  - alexnet
  - deep-residual-learning
related_architecture_slugs:
  - alexnet
  - vgg
  - resnet
  - efficientnet
```

### topic: residual-networks 🔴 MISSING
```yaml
slug: residual-networks
title: Residual Networks
subtitle: Skip connections that enable training of very deep networks without vanishing gradients.
difficulty: intermediate
duration_minutes: 35
domain_slug: deep-learning
prerequisites:
  - Convolutional Networks
  - Backpropagation
  - Batch Normalization
tags:
  - ResNet
  - Skip Connections
  - Deep Learning
status: missing
practice_problem_slugs: []
related_paper_slugs:
  - deep-residual-learning
  - batch-normalization
related_architecture_slugs:
  - resnet
```

### topic: batch-normalization 🔴 MISSING
```yaml
slug: batch-normalization
title: Batch Normalization
subtitle: Normalizing layer inputs to stabilize and accelerate training.
difficulty: intermediate
duration_minutes: 25
domain_slug: deep-learning
prerequisites:
  - Neural Networks
  - Backpropagation
tags:
  - Normalization
  - Training
status: missing
practice_problem_slugs: []
related_paper_slugs:
  - batch-normalization
related_architecture_slugs:
  - resnet
```

---

## DOMAIN: machine-learning

### topic: gradient-descent 🔴 MISSING
```yaml
slug: gradient-descent
title: Gradient Descent
subtitle: Iterative first-order optimization that underpins all of deep learning.
difficulty: beginner
duration_minutes: 30
domain_slug: machine-learning
prerequisites:
  - Calculus
  - Linear Algebra
tags:
  - Optimization
  - SGD
  - Adam
status: missing
practice_problem_slugs:
  - gradient-descent
related_paper_slugs: []
related_architecture_slugs: []
```

### topic: loss-functions 🔴 MISSING
```yaml
slug: loss-functions
title: Loss Functions
subtitle: Quantifying how wrong the model is — the signal that drives learning.
difficulty: beginner
duration_minutes: 25
domain_slug: machine-learning
prerequisites:
  - Probability
  - Calculus
tags:
  - Cross-Entropy
  - MSE
  - Objectives
status: missing
practice_problem_slugs:
  - cross-entropy-loss
related_paper_slugs: []
related_architecture_slugs: []
```

### topic: transfer-learning 🔴 MISSING
```yaml
slug: transfer-learning
title: Transfer Learning
subtitle: Reusing knowledge from large pre-trained models to solve new tasks efficiently.
difficulty: intermediate
duration_minutes: 35
domain_slug: machine-learning
prerequisites:
  - Neural Networks
  - Fine-Tuning
tags:
  - Pre-training
  - Fine-tuning
  - Domain Adaptation
status: missing
practice_problem_slugs: []
related_paper_slugs:
  - bert
  - gpt
related_architecture_slugs:
  - bert
  - gpt
  - vit
```

---

## DOMAIN: llms

### topic: tokenization 🔴 MISSING
```yaml
slug: tokenization
title: Tokenization
subtitle: Converting raw text into model-readable integer IDs — the first step in every LLM pipeline.
difficulty: beginner
duration_minutes: 20
domain_slug: llms
prerequisites:
  - Python
  - Basic NLP concepts
tags:
  - BPE
  - SentencePiece
  - Vocabulary
status: missing
practice_problem_slugs: []
related_paper_slugs:
  - gpt-2
  - bert
related_architecture_slugs:
  - bert
  - gpt
```

### topic: kv-cache 🔴 MISSING
```yaml
slug: kv-cache
title: KV Cache
subtitle: Storing past keys and values to eliminate redundant computation during autoregressive inference.
difficulty: intermediate
duration_minutes: 30
domain_slug: llms
prerequisites:
  - Attention Mechanism
  - Transformers
  - Autoregressive Decoding
tags:
  - Inference
  - Memory
  - Efficiency
status: missing
practice_problem_slugs:
  - kv-cache
related_paper_slugs:
  - gpt-3
related_architecture_slugs:
  - gpt
  - llama
```

### topic: rope-embeddings 🔴 MISSING
```yaml
slug: rope-embeddings
title: Rotary Position Embeddings (RoPE)
subtitle: Position-dependent rotation of query/key vectors that encodes relative position in dot products.
difficulty: advanced
duration_minutes: 35
domain_slug: llms
prerequisites:
  - Attention Mechanism
  - Complex Numbers
  - Linear Algebra
tags:
  - Positional Encoding
  - RoPE
  - LLaMA
status: missing
practice_problem_slugs:
  - llama-rope
related_paper_slugs:
  - llama
related_architecture_slugs:
  - llama
```

### topic: mixture-of-experts 🔴 MISSING
```yaml
slug: mixture-of-experts
title: Mixture of Experts
subtitle: Sparse activation of expert FFNs — more parameters without proportionally more FLOPs.
difficulty: advanced
duration_minutes: 40
domain_slug: llms
prerequisites:
  - Feed-Forward Networks
  - Transformer Architecture
  - Routing Algorithms
tags:
  - MoE
  - Sparse
  - Scaling
  - Efficiency
status: missing
practice_problem_slugs:
  - moe-routing
related_paper_slugs:
  - switch-transformer
related_architecture_slugs:
  - moe
```

### topic: rlhf 🔴 MISSING
```yaml
slug: rlhf
title: RLHF — Reinforcement Learning from Human Feedback
subtitle: The training recipe behind ChatGPT — aligning LLMs to human preferences using reward models and PPO.
difficulty: advanced
duration_minutes: 45
domain_slug: llms
prerequisites:
  - Transformer Architecture
  - Fine-Tuning
  - Reinforcement Learning Basics
tags:
  - Alignment
  - PPO
  - Reward Model
  - ChatGPT
status: missing
practice_problem_slugs: []
related_paper_slugs:
  - gpt-3
related_architecture_slugs:
  - gpt
  - llama
```

### topic: rag 🔴 MISSING
```yaml
slug: rag
title: Retrieval-Augmented Generation
subtitle: Grounding LLM outputs in retrieved documents to reduce hallucination and add up-to-date knowledge.
difficulty: intermediate
duration_minutes: 40
domain_slug: llms
prerequisites:
  - Transformer Architecture
  - Vector Search
  - Embeddings
tags:
  - RAG
  - Vector DB
  - Grounding
status: missing
practice_problem_slugs: []
related_paper_slugs: []
related_architecture_slugs:
  - bert
  - gpt
```

---

## DOMAIN: computer-vision

### topic: vision-transformers 🔴 MISSING
```yaml
slug: vision-transformers
title: Vision Transformers (ViT)
subtitle: Treating image patches as tokens and applying transformer attention to visual data.
difficulty: intermediate
duration_minutes: 40
domain_slug: computer-vision
prerequisites:
  - Attention Mechanism
  - Transformers
  - CNNs
tags:
  - ViT
  - Patches
  - Self-Supervised Learning
status: missing
practice_problem_slugs:
  - vit-patch-size
related_paper_slugs:
  - vision-transformer
related_architecture_slugs:
  - vit
  - swin
  - dino
```

### topic: diffusion-models 🔴 MISSING
```yaml
slug: diffusion-models
title: Diffusion Models
subtitle: Score matching and iterative denoising — the engine behind modern image generation.
difficulty: advanced
duration_minutes: 50
domain_slug: computer-vision
prerequisites:
  - Probability Theory
  - Score Matching
  - U-Net Architecture
tags:
  - DDPM
  - Score Matching
  - Stable Diffusion
status: missing
practice_problem_slugs:
  - stable-diffusion-cfg
related_paper_slugs:
  - latent-diffusion-models
  - stable-diffusion
related_architecture_slugs:
  - diffusion
  - stable-diffusion
  - unet
  - vae
```

### topic: contrastive-learning 🔴 MISSING
```yaml
slug: contrastive-learning
title: Contrastive Learning
subtitle: Learning representations by pulling similar pairs together and pushing dissimilar pairs apart.
difficulty: intermediate
duration_minutes: 35
domain_slug: computer-vision
prerequisites:
  - Embeddings
  - Loss Functions
  - Neural Networks
tags:
  - CLIP
  - SimCLR
  - Representations
status: missing
practice_problem_slugs:
  - clip-batch-size
related_paper_slugs:
  - clip
related_architecture_slugs:
  - clip
  - dino
```

---

## SUMMARY TABLE

| Domain | Topic Slug | Status | Priority |
|--------|-----------|--------|---------|
| deep-learning | attention | ✅ Complete | — |
| deep-learning | transformers | 🔴 Missing | High |
| deep-learning | backpropagation | 🔴 Missing | High |
| deep-learning | convolutional-networks | 🔴 Missing | High |
| deep-learning | residual-networks | 🔴 Missing | Medium |
| deep-learning | batch-normalization | 🔴 Missing | Medium |
| machine-learning | gradient-descent | 🔴 Missing | High |
| machine-learning | loss-functions | 🔴 Missing | High |
| machine-learning | transfer-learning | 🔴 Missing | Medium |
| llms | tokenization | 🔴 Missing | High |
| llms | kv-cache | 🔴 Missing | High |
| llms | rope-embeddings | 🔴 Missing | Medium |
| llms | mixture-of-experts | 🔴 Missing | Medium |
| llms | rlhf | 🔴 Missing | High |
| llms | rag | 🔴 Missing | High |
| computer-vision | vision-transformers | 🔴 Missing | High |
| computer-vision | diffusion-models | 🔴 Missing | Medium |
| computer-vision | contrastive-learning | 🔴 Missing | Medium |

**1 complete, 17 specified-but-missing, ~80+ unspecified**
