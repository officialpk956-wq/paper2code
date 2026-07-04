// Total topics: 82
export type CurriculumTopic = {
  slug: string; title: string;
  level: 'Beginner'|'Intermediate'|'Advanced'|'Expert';
  prerequisites: string[];
  studyTime: string;
  why: string;
  unlocks: string[];
};

export type CurriculumDomain = {
  slug: string; number: number; name: string;
  topics: CurriculumTopic[];
};

export const CURRICULUM: CurriculumDomain[] = [
  {
    "slug": "mathematics",
    "number": 1,
    "name": "MATHEMATICS",
    "topics": [
      {
        "slug": "statistical-learning-theory",
        "title": "Statistical Learning Theory",
        "level": "Advanced",
        "prerequisites": [
          "Measure Theory",
          "Statistical Inference",
          "Convex Opt."
        ],
        "studyTime": "24h",
        "why": "Why neural networks generalize despite overparameterization.",
        "unlocks": [
          "Scaling Law Theory",
          "Double Descent",
          "Implicit Regularization"
        ]
      },
      {
        "slug": "optimal-transport",
        "title": "Optimal Transport",
        "level": "Advanced",
        "prerequisites": [
          "Measure Theory",
          "Convex Optimization"
        ],
        "studyTime": "20h",
        "why": "Wasserstein distances power GAN training stability,",
        "unlocks": [
          "Wasserstein GANs",
          "Flow Matching",
          "Diffusion Theory"
        ]
      },
      {
        "slug": "differential-geometry-for-ml",
        "title": "Differential Geometry for ML",
        "level": "Advanced",
        "prerequisites": [
          "Calculus II",
          "Linear Algebra II",
          "Measure Theory"
        ],
        "studyTime": "24h",
        "why": "Riemannian optimization, manifold hypothesis, and",
        "unlocks": [
          "Riemannian Optimizers",
          "Geometric Deep Learning"
        ]
      },
      {
        "slug": "variational-inference-and-free-energy",
        "title": "Variational Inference & Free Energy",
        "level": "Advanced",
        "prerequisites": [
          "Information Theory",
          "Probability Theory",
          "Convex Opt."
        ],
        "studyTime": "20h",
        "why": "ELBO, VAE training, DDPM derivation, and Bayesian",
        "unlocks": [
          "VAE",
          "Diffusion Models (Theory)",
          "Bayesian NNs"
        ]
      },
      {
        "slug": "spectral-methods-and-random-matrix-theory",
        "title": "Spectral Methods & Random Matrix Theory",
        "level": "Advanced",
        "prerequisites": [
          "Linear Algebra II",
          "Measure Theory"
        ],
        "studyTime": "20h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "machine-learning",
    "number": 2,
    "name": "MACHINE LEARNING",
    "topics": [
      {
        "slug": "meta-learning",
        "title": "Meta-Learning",
        "level": "Advanced",
        "prerequisites": [
          "Bayesian ML",
          "Gradient-Based Optimization"
        ],
        "studyTime": "20h",
        "why": "Learning to learn. Foundation for few-shot prompting theory",
        "unlocks": [
          "In-Context Learning Theory",
          "Foundation Model Adaptation"
        ]
      },
      {
        "slug": "federated-learning",
        "title": "Federated Learning",
        "level": "Advanced",
        "prerequisites": [
          "Distributed Optimization",
          "Regularization",
          "Privacy"
        ],
        "studyTime": "18h",
        "why": "Training on private distributed data without centralization.",
        "unlocks": [
          "Privacy-Preserving ML",
          "On-Device Training"
        ]
      },
      {
        "slug": "neural-architecture-search-nas",
        "title": "Neural Architecture Search (NAS)",
        "level": "Advanced",
        "prerequisites": [
          "Deep Learning",
          "Meta-Learning",
          "Optimization"
        ],
        "studyTime": "24h",
        "why": "EfficientNet, MobileNet, and DARTS were found by NAS.",
        "unlocks": [
          "Efficient Architecture Design",
          "AutoML Systems"
        ]
      },
      {
        "slug": "learning-theory",
        "title": "Learning Theory",
        "level": "Advanced",
        "prerequisites": [
          "Statistical Learning Theory",
          "Kernel Methods"
        ],
        "studyTime": "20h",
        "why": "Explains why overparameterized neural networks generalize.",
        "unlocks": [
          "Scaling Law Research",
          "Implicit Bias Research"
        ]
      },
      {
        "slug": "multi-task-and-transfer-learning-theory",
        "title": "Multi-Task & Transfer Learning Theory",
        "level": "Advanced",
        "prerequisites": [
          "Statistical Learning Theory",
          "Meta-Learning"
        ],
        "studyTime": "18h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "deep-learning",
    "number": 3,
    "name": "DEEP LEARNING",
    "topics": [
      {
        "slug": "scaling-laws",
        "title": "Scaling Laws",
        "level": "Advanced",
        "prerequisites": [
          "Transformer",
          "Training Loop",
          "Statistical Learning Theory"
        ],
        "studyTime": "20h",
        "why": "The empirical laws governing compute-optimal training.",
        "unlocks": [
          "Training Budget Decisions",
          "Pre-training Research"
        ]
      },
      {
        "slug": "mixture-of-experts-moe",
        "title": "Mixture of Experts (MoE)",
        "level": "Advanced",
        "prerequisites": [
          "Transformer",
          "Load Balancing",
          "Distributed Training"
        ],
        "studyTime": "20h",
        "why": "Mixtral, GPT-4, Gemini Ultra are all MoE. Sparse activation",
        "unlocks": [
          "Routing Algorithms",
          "Expert Specialization",
          "Sparse Training"
        ]
      },
      {
        "slug": "mechanistic-interpretability",
        "title": "Mechanistic Interpretability",
        "level": "Advanced",
        "prerequisites": [
          "Transformer",
          "Attention Patterns",
          "Probing Methods"
        ],
        "studyTime": "30h",
        "why": "Understanding induction heads, circuits, and superposition",
        "unlocks": [
          "AI Safety Research",
          "Circuit Analysis",
          "Sparse Autoencoders"
        ]
      },
      {
        "slug": "neural-odes-and-continuous-depth-networks",
        "title": "Neural ODEs & Continuous-Depth Networks",
        "level": "Advanced",
        "prerequisites": [
          "Differential Equations",
          "Residual Networks",
          "ODE Solvers"
        ],
        "studyTime": "20h",
        "why": "Theoretical underpinning of normalizing flows and",
        "unlocks": [
          "Flow Matching",
          "Normalizing Flows",
          "Score-Based Models"
        ]
      },
      {
        "slug": "emergent-abilities-and-phase-transitions",
        "title": "Emergent Abilities & Phase Transitions",
        "level": "Advanced",
        "prerequisites": [
          "Scaling Laws",
          "Statistical Learning Theory"
        ],
        "studyTime": "20h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "natural-language-processing",
    "number": 4,
    "name": "NATURAL LANGUAGE PROCESSING",
    "topics": [
      {
        "slug": "reasoning-in-language-models",
        "title": "Reasoning in Language Models",
        "level": "Advanced",
        "prerequisites": [
          "Chain-of-Thought",
          "RLHF",
          "Process Reward Models"
        ],
        "studyTime": "24h",
        "why": "o1, DeepSeek-R1 — extended thinking and process supervision",
        "unlocks": [
          "Process Reward Models",
          "MCTS over Thoughts",
          "GRPO"
        ]
      },
      {
        "slug": "factuality-grounding-and-hallucination",
        "title": "Factuality, Grounding & Hallucination",
        "level": "Advanced",
        "prerequisites": [
          "RLHF",
          "RAG",
          "Evaluation Methodology"
        ],
        "studyTime": "20h",
        "why": "Hallucination is the primary obstacle to deploying LLMs",
        "unlocks": [
          "Faithful RAG",
          "Citation Generation",
          "Verification Pipelines"
        ]
      },
      {
        "slug": "alignment",
        "title": "Alignment",
        "level": "Advanced",
        "prerequisites": [
          "RL Policy Gradient",
          "Reward Modeling",
          "Instruction Tuning"
        ],
        "studyTime": "28h",
        "why": "Making models helpful, harmless, and honest. This is",
        "unlocks": [
          "Red-Teaming",
          "Safety Evals",
          "Constitutional AI",
          "RLAIF"
        ]
      },
      {
        "slug": "neural-machine-translation-at-scale",
        "title": "Neural Machine Translation at Scale",
        "level": "Advanced",
        "prerequisites": [
          "Seq2Seq",
          "Multilingual",
          "Scaling Laws"
        ],
        "studyTime": "20h",
        "why": "Google Translate and Meta's NLLB serve billions.",
        "unlocks": [
          "Universal Translation Models",
          "Multilingual Generation"
        ]
      },
      {
        "slug": "language-model-interpretability",
        "title": "Language Model Interpretability",
        "level": "Advanced",
        "prerequisites": [
          "Mechanistic Interpretability",
          "Probing",
          "Attention Analysis"
        ],
        "studyTime": "24h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "llm-engineering",
    "number": 5,
    "name": "LLM ENGINEERING",
    "topics": [
      {
        "slug": "continued-pre-training-and-domain-adaptation-at-scale",
        "title": "Continued Pre-training & Domain Adaptation at Scale",
        "level": "Advanced",
        "prerequisites": [
          "Scaling Laws",
          "Fine-Tuning",
          "Distributed Training"
        ],
        "studyTime": "28h",
        "why": "How specialized models (CodeLlama, BioMedLM, StarCoder)",
        "unlocks": [
          "Domain-Specific Foundation Models",
          "Curriculum Training"
        ]
      },
      {
        "slug": "flash-attention-and-memory-efficient-attention",
        "title": "Flash Attention & Memory-Efficient Attention",
        "level": "Advanced",
        "prerequisites": [
          "GPU Architecture",
          "CUDA",
          "Attention (full)",
          "IO Complexity"
        ],
        "studyTime": "24h",
        "why": "Flash Attention 2 is in every frontier training run.",
        "unlocks": [
          "Custom CUDA Kernels",
          "Training Efficiency",
          "Long Context"
        ]
      },
      {
        "slug": "tensor-parallelism-pipeline-parallelism-and-fsdp",
        "title": "Tensor Parallelism, Pipeline Parallelism & FSDP",
        "level": "Advanced",
        "prerequisites": [
          "Distributed Training",
          "GPU Programming",
          "Transformer"
        ],
        "studyTime": "30h",
        "why": "Training models that don't fit on one GPU. Megatron-LM,",
        "unlocks": [
          "100B+ Scale Training",
          "Cluster Engineering"
        ]
      },
      {
        "slug": "continual-learning-and-catastrophic-forgetting",
        "title": "Continual Learning & Catastrophic Forgetting",
        "level": "Advanced",
        "prerequisites": [
          "Fine-Tuning",
          "Regularization Theory",
          "EWC"
        ],
        "studyTime": "20h",
        "why": "Production LLMs need to learn from new data without",
        "unlocks": [
          "Lifelong Learning Systems",
          "Model Merging",
          "Distillation"
        ]
      },
      {
        "slug": "model-merging",
        "title": "Model Merging",
        "level": "Advanced",
        "prerequisites": [
          "Fine-Tuning",
          "LoRA",
          "Model Weight Analysis"
        ],
        "studyTime": "16h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "rag-systems",
    "number": 6,
    "name": "RAG SYSTEMS",
    "topics": [
      {
        "slug": "raft",
        "title": "RAFT",
        "level": "Advanced",
        "prerequisites": [
          "Fine-Tuning",
          "RAG",
          "Dataset Construction"
        ],
        "studyTime": "20h",
        "why": "Fine-tuning models to use retrieved documents reliably.",
        "unlocks": [
          "Domain-Tuned RAG Systems",
          "Distillation from RAG Pipelines"
        ]
      },
      {
        "slug": "multi-modal-rag",
        "title": "Multi-Modal RAG",
        "level": "Advanced",
        "prerequisites": [
          "VLMs",
          "PDF Parsing",
          "OCR",
          "Multi-Modal Embeddings"
        ],
        "studyTime": "24h",
        "why": "Real enterprise documents contain charts, tables, and",
        "unlocks": [
          "Document AI",
          "Colpali",
          "Vision-Language RAG"
        ]
      },
      {
        "slug": "production-rag-architecture-at-scale",
        "title": "Production RAG Architecture at Scale",
        "level": "Advanced",
        "prerequisites": [
          "All RAG Components",
          "Distributed Systems",
          "Monitoring"
        ],
        "studyTime": "28h",
        "why": "Building RAG for 10 users vs 1M users are completely",
        "unlocks": [
          "ML Platform Design",
          "RAG Infrastructure",
          "SRE for AI"
        ]
      },
      {
        "slug": "continual-indexing-and-knowledge-freshness",
        "title": "Continual Indexing & Knowledge Freshness",
        "level": "Advanced",
        "prerequisites": [
          "Production RAG",
          "Streaming Pipelines",
          "Change Detection"
        ],
        "studyTime": "20h",
        "why": "Real corpora change. Keeping indexes fresh without",
        "unlocks": [
          "Real-Time RAG",
          "Streaming Knowledge Updates"
        ]
      },
      {
        "slug": "rag-for-code",
        "title": "RAG for Code",
        "level": "Advanced",
        "prerequisites": [
          "Code Embeddings",
          "AST Parsing",
          "GraphRAG",
          "Fine-Tuning"
        ],
        "studyTime": "20h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "ai-agents",
    "number": 7,
    "name": "AI AGENTS",
    "topics": [
      {
        "slug": "emergent-cooperation-in-multi-agent-systems",
        "title": "Emergent Cooperation in Multi-Agent Systems",
        "level": "Advanced",
        "prerequisites": [
          "Multi-Agent",
          "Game Theory",
          "MARL"
        ],
        "studyTime": "24h",
        "why": "Do agents develop communication protocols, division of labor,",
        "unlocks": [
          "Agent Societies",
          "AI-Human Teams",
          "Emergent Language Research"
        ]
      },
      {
        "slug": "continual-learning-agents",
        "title": "Continual Learning Agents",
        "level": "Advanced",
        "prerequisites": [
          "Continual Learning",
          "Memory Systems",
          "Self-Reflection"
        ],
        "studyTime": "24h",
        "why": "Agents that improve with experience without forgetting.",
        "unlocks": [
          "Life-Long Agents",
          "Personalized AI",
          "AIXI-Adjacent Research"
        ]
      },
      {
        "slug": "simulation-environments-for-agent-training",
        "title": "Simulation Environments for Agent Training",
        "level": "Advanced",
        "prerequisites": [
          "RL Environments",
          "Multi-Agent",
          "LLM Fine-Tuning"
        ],
        "studyTime": "24h",
        "why": "AgentBench, WebArena, OSWorld — simulations scale",
        "unlocks": [
          "Synthetic Agent Training",
          "RL from Interaction"
        ]
      },
      {
        "slug": "world-models-for-agents",
        "title": "World Models for Agents",
        "level": "Advanced",
        "prerequisites": [
          "Model-Based RL",
          "Neural ODEs",
          "Transformer"
        ],
        "studyTime": "28h",
        "why": "Agents that can imagine the future plan better.",
        "unlocks": [
          "Embodied AI",
          "Video Generation",
          "Physical Reasoning"
        ]
      },
      {
        "slug": "formal-specification-and-verification-of-agent-behavior",
        "title": "Formal Specification & Verification of Agent Behavior",
        "level": "Advanced",
        "prerequisites": [
          "Formal Methods",
          "Agent Architecture",
          "Safety Research"
        ],
        "studyTime": "24h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "reinforcement-learning",
    "number": 8,
    "name": "REINFORCEMENT LEARNING",
    "topics": [
      {
        "slug": "world-models",
        "title": "World Models",
        "level": "Advanced",
        "prerequisites": [
          "Model-Based RL",
          "VAEs",
          "RSSM",
          "Latent Dynamics"
        ],
        "studyTime": "28h",
        "why": "Dreamer achieves human-level on 150+ Atari games from pixels",
        "unlocks": [
          "Video Game AI",
          "Embodied Robotics",
          "Physical Simulation"
        ]
      },
      {
        "slug": "hierarchical-rl",
        "title": "Hierarchical RL",
        "level": "Advanced",
        "prerequisites": [
          "PPO/SAC",
          "Planning",
          "Temporal Abstraction"
        ],
        "studyTime": "24h",
        "why": "Long-horizon tasks require temporal abstraction. Hierarchical",
        "unlocks": [
          "Instruction-Following Robots",
          "Task Planning",
          "HRL Research"
        ]
      },
      {
        "slug": "safe-rl-and-constrained-mdp",
        "title": "Safe RL & Constrained MDP",
        "level": "Advanced",
        "prerequisites": [
          "MDPs",
          "Constrained Optimization",
          "PPO"
        ],
        "studyTime": "22h",
        "why": "Autonomous vehicles and surgical robots cannot fail.",
        "unlocks": [
          "AI Safety",
          "Constrained Policy Search",
          "Shielding Methods"
        ]
      },
      {
        "slug": "alphazero-and-mcts-with-deep-rl",
        "title": "AlphaZero & MCTS with Deep RL",
        "level": "Advanced",
        "prerequisites": [
          "MCTS",
          "Model-Based RL",
          "Self-Play",
          "Value Functions"
        ],
        "studyTime": "24h",
        "why": "AlphaZero mastered Chess, Go, and Shogi with zero human",
        "unlocks": [
          "Self-Play Alignment",
          "Reasoning with MCTS",
          "AlphaCode"
        ]
      },
      {
        "slug": "foundation-models-as-world-models",
        "title": "Foundation Models as World Models",
        "level": "Advanced",
        "prerequisites": [
          "World Models",
          "Scaling Laws",
          "Video Generation"
        ],
        "studyTime": "28h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "computer-vision",
    "number": 9,
    "name": "COMPUTER VISION",
    "topics": [
      {
        "slug": "vision-language-models",
        "title": "Vision-Language Models",
        "level": "Advanced",
        "prerequisites": [
          "LLMs",
          "CLIP",
          "Visual Instruction Tuning"
        ],
        "studyTime": "24h",
        "why": "VLMs power GPT-4V, Claude Vision, and Gemini.",
        "unlocks": [
          "Visual Agents",
          "Multi-Modal RAG",
          "Document Understanding"
        ]
      },
      {
        "slug": "foundation-models-for-vision",
        "title": "Foundation Models for Vision",
        "level": "Advanced",
        "prerequisites": [
          "ViT",
          "Masked Modeling",
          "CLIP",
          "Scaling Laws"
        ],
        "studyTime": "24h",
        "why": "Segment Anything Model prompts visual segmentation like",
        "unlocks": [
          "Open-Vocabulary Vision",
          "Zero-Shot Detection",
          "Robotics Vision"
        ]
      },
      {
        "slug": "video-generation",
        "title": "Video Generation",
        "level": "Advanced",
        "prerequisites": [
          "Diffusion",
          "Video Understanding",
          "DiT",
          "3D Attention"
        ],
        "studyTime": "28h",
        "why": "Sora generates photorealistic video from text. DiT-based",
        "unlocks": [
          "World Models",
          "Video Editing",
          "Generative Film Production"
        ]
      },
      {
        "slug": "generative-video-editing-and-controlnet",
        "title": "Generative Video Editing & ControlNet",
        "level": "Advanced",
        "prerequisites": [
          "Latent Diffusion",
          "LoRA Fine-Tuning",
          "Conditioning Mechanisms"
        ],
        "studyTime": "20h",
        "why": "ControlNet, IP-Adapter, and custom diffusion models enable",
        "unlocks": [
          "Commercial Image Generation",
          "Virtual Try-On",
          "Style Transfer"
        ]
      },
      {
        "slug": "embodied-vision",
        "title": "Embodied Vision",
        "level": "Advanced",
        "prerequisites": [
          "3D Vision",
          "RL",
          "Depth Estimation",
          "Foundation Models"
        ],
        "studyTime": "30h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "mlops",
    "number": 10,
    "name": "MLOPS",
    "topics": [
      {
        "slug": "ml-platform-engineering",
        "title": "ML Platform Engineering",
        "level": "Advanced",
        "prerequisites": [
          "Kubernetes",
          "Feature Stores",
          "CI/CD",
          "Model Registry"
        ],
        "studyTime": "30h",
        "why": "Uber Michelangelo, Airbnb Bighead, Meta FBLearner — internal",
        "unlocks": [
          "Platform Architecture",
          "Multi-Team ML Infrastructure"
        ]
      },
      {
        "slug": "large-scale-training-infrastructure",
        "title": "Large-Scale Training Infrastructure",
        "level": "Advanced",
        "prerequisites": [
          "Distributed Training",
          "GPU Networking (InfiniBand/NVLink)"
        ],
        "studyTime": "30h",
        "why": "Training frontier models at 10K+ GPU scale requires",
        "unlocks": [
          "Frontier Model Training",
          "Training Infrastructure Design"
        ]
      },
      {
        "slug": "real-time-inference-at-scale",
        "title": "Real-Time Inference at Scale",
        "level": "Advanced",
        "prerequisites": [
          "LLMOps",
          "PagedAttention",
          "Continuous Batching",
          "KV Cache"
        ],
        "studyTime": "24h",
        "why": "OpenAI serves billions of requests. Continuous batching,",
        "unlocks": [
          "Inference Infrastructure Design",
          "Cost Optimization at Scale"
        ]
      },
      {
        "slug": "cost-engineering-and-carbon-accounting-for-ml",
        "title": "Cost Engineering & Carbon Accounting for ML",
        "level": "Advanced",
        "prerequisites": [
          "Inference Infrastructure",
          "Distributed Training",
          "Cloud Pricing"
        ],
        "studyTime": "16h",
        "why": "Training a GPT-4-scale model costs $100M+. Responsible",
        "unlocks": [
          "Efficiency Research",
          "Sustainable AI",
          "Infrastructure ROI"
        ]
      },
      {
        "slug": "ml-governance-and-compliance",
        "title": "ML Governance & Compliance",
        "level": "Advanced",
        "prerequisites": [
          "Fairness in ML",
          "Model Cards",
          "Legal Basics",
          "Monitoring"
        ],
        "studyTime": "20h",
        "why": "",
        "unlocks": []
      }
    ]
  },
  {
    "slug": "ai-system-design",
    "number": 11,
    "name": "AI SYSTEM DESIGN",
    "topics": [
      {
        "slug": "inference-optimization",
        "title": "Inference Optimization",
        "level": "Advanced",
        "prerequisites": [
          "Model Serving",
          "Quantization Basics",
          "CUDA Fundamentals"
        ],
        "studyTime": "12 hours",
        "why": "Inference is 90% of production compute cost. Optimization directly translates to economics.",
        "unlocks": [
          "Custom CUDA kernels",
          "speculative decoding",
          "hardware-model co-design"
        ]
      },
      {
        "slug": "speculative-decoding",
        "title": "Speculative Decoding",
        "level": "Advanced",
        "prerequisites": [
          "Autoregressive LLM inference",
          "KV-cache",
          "Draft models"
        ],
        "studyTime": "8 hours",
        "why": "2–4× throughput on LLMs at same quality. Industry standard for real-time generation.",
        "unlocks": [
          "Medusa heads",
          "lookahead decoding",
          "tree attention"
        ]
      },
      {
        "slug": "distributed-inference",
        "title": "Distributed Inference",
        "level": "Advanced",
        "prerequisites": [
          "Model parallelism",
          "Tensor parallelism",
          "Ray/vLLM"
        ],
        "studyTime": "14 hours",
        "why": "LLMs at 70B+ don't fit on one GPU. Distributed inference is the only option.",
        "unlocks": [
          "Pipeline parallelism for inference",
          "disaggregated prefill/decode"
        ]
      },
      {
        "slug": "system-design-for-retrieval-at-scale",
        "title": "System Design for Retrieval at Scale",
        "level": "Advanced",
        "prerequisites": [
          "FAISS",
          "ANN algorithms",
          "Embedding models",
          "HNSW"
        ],
        "studyTime": "10 hours",
        "why": "Billion-scale vector search requires careful index sharding, caching, and approximate search tuning.",
        "unlocks": [
          "Hybrid BM25+dense retrieval",
          "multi-vector ColBERT",
          "re-ranking pipelines"
        ]
      },
      {
        "slug": "llm-gateway-design",
        "title": "LLM Gateway Design",
        "level": "Advanced",
        "prerequisites": [
          "Load balancing",
          "Rate limiting",
          "API design",
          "Cost accounting"
        ],
        "studyTime": "8 hours",
        "why": "Enterprise LLM access needs routing, fallback, auth, and cost controls at the gateway layer.",
        "unlocks": [
          "Semantic caching",
          "prompt injection defenses",
          "multi-model routing"
        ]
      },
      {
        "slug": "evaluation-infrastructure",
        "title": "Evaluation Infrastructure",
        "level": "Advanced",
        "prerequisites": [
          "LLM-as-judge",
          "RAGAS",
          "Human eval pipelines",
          "Statistical testing"
        ],
        "studyTime": "10 hours",
        "why": "Without rigorous evals, you're flying blind in production. Evals are the test suite for AI.",
        "unlocks": [
          "Automated regression detection",
          "red-teaming automation",
          "eval-driven fine-tuning"
        ]
      },
      {
        "slug": "hardware-aware-architecture-search",
        "title": "Hardware-Aware Architecture Search",
        "level": "Expert",
        "prerequisites": [
          "Neural architecture search",
          "Hardware profiling",
          "Pareto optimization"
        ],
        "studyTime": "20 hours",
        "why": "EfficientNet, MobileNet-style models beat hand-designed ones because they're optimized for actual hardware constraints.",
        "unlocks": [
          "TinyML",
          "edge deployment",
          "hardware/software co-design"
        ]
      },
      {
        "slug": "custom-cuda-kernels-for-ml",
        "title": "Custom CUDA Kernels for ML",
        "level": "Expert",
        "prerequisites": [
          "CUDA programming",
          "GPU memory hierarchy",
          "Triton",
          "Flash Attention internals"
        ],
        "studyTime": "30 hours",
        "why": "The fastest models (Flash Attention, xFormers, cuDNN) use hand-written kernels that outperform compiler-generated code.",
        "unlocks": [
          "FlashAttention variants",
          "custom attention patterns",
          "memory-efficient backprop"
        ]
      },
      {
        "slug": "multi-cluster-training-orchestration",
        "title": "Multi-Cluster Training Orchestration",
        "level": "Expert",
        "prerequisites": [
          "FSDP",
          "Megatron-LM",
          "DeepSpeed",
          "Network topology awareness"
        ],
        "studyTime": "25 hours",
        "why": "GPT-4 scale training requires multi-cluster coordination across thousands of GPUs. This is the frontier of distributed systems.",
        "unlocks": [
          "Inter-datacenter training",
          "fault-tolerant distributed optimization"
        ]
      },
      {
        "slug": "production-safety-and-alignment-systems",
        "title": "Production Safety & Alignment Systems",
        "level": "Expert",
        "prerequisites": [
          "RLHF",
          "Constitutional AI",
          "Red-teaming",
          "Content moderation"
        ],
        "studyTime": "20 hours",
        "why": "Deploying powerful models without safety systems creates unacceptable risk. This is mandatory for any public-facing LLM product.",
        "unlocks": [
          "Scalable oversight",
          "automated red-teaming",
          "preference learning at scale"
        ]
      },
      {
        "slug": "compound-ai-systems",
        "title": "Compound AI Systems",
        "level": "Expert",
        "prerequisites": [
          "LLM Agents",
          "RAG",
          "Tool use",
          "Memory systems",
          "Orchestration frameworks"
        ],
        "studyTime": "20 hours",
        "why": "The frontier is no longer single-model systems — it's orchestrated pipelines of models, retrievers, tools, and verifiers.",
        "unlocks": [
          "Research engineering",
          "novel AI product architectures"
        ]
      }
    ]
  },
  {
    "slug": "research-engineering",
    "number": 12,
    "name": "Research Engineering",
    "topics": [
      {
        "slug": "reading-ml-papers-effectively",
        "title": "Reading ML Papers Effectively",
        "level": "Beginner",
        "prerequisites": [
          "Basic ML knowledge",
          "Linear algebra"
        ],
        "studyTime": "4 hours",
        "why": "95% of frontier AI knowledge lives in papers. Reading speed and comprehension directly cap your learning rate.",
        "unlocks": [
          "Paper implementation",
          "reproducing results",
          "literature reviews"
        ]
      },
      {
        "slug": "setting-up-a-research-environment",
        "title": "Setting Up a Research Environment",
        "level": "Beginner",
        "prerequisites": [
          "Python",
          "Git",
          "Conda/pip"
        ],
        "studyTime": "3 hours",
        "why": "Reproducible environments prevent the #1 source of wasted research time: \"it worked on my machine.\"",
        "unlocks": [
          "Remote GPU training",
          "containerized experiments",
          "team reproducibility"
        ]
      },
      {
        "slug": "git-for-research-projects",
        "title": "Git for Research Projects",
        "level": "Beginner",
        "prerequisites": [
          "Basic terminal",
          "Python"
        ],
        "studyTime": "3 hours",
        "why": "Experiments without version control are unrecoverable. Good Git hygiene lets you bisect failures and share work.",
        "unlocks": [
          "Branching experiments",
          "code review workflows",
          "open-source contribution"
        ]
      },
      {
        "slug": "experiment-tracking-basics-wandb-mlflow",
        "title": "Experiment Tracking Basics (W&B / MLflow)",
        "level": "Beginner",
        "prerequisites": [
          "Python",
          "PyTorch basics"
        ],
        "studyTime": "4 hours",
        "why": "You cannot improve what you cannot measure. Tracking every hyperparameter and metric is table stakes for research.",
        "unlocks": [
          "Hyperparameter sweeps",
          "experiment comparison",
          "ablation automation"
        ]
      },
      {
        "slug": "writing-research-code",
        "title": "Writing Research Code",
        "level": "Beginner",
        "prerequisites": [
          "Python",
          "OOP",
          "basic software engineering"
        ],
        "studyTime": "5 hours",
        "why": "Research code that can't be modified or understood by others (including future you) slows iteration.",
        "unlocks": [
          "Config-driven experiments",
          "modular architectures",
          "open-source release"
        ]
      },
      {
        "slug": "reproducing-sota-results",
        "title": "Reproducing SOTA Results",
        "level": "Intermediate",
        "prerequisites": [
          "Reading papers",
          "PyTorch",
          "W&B",
          "GPU access"
        ],
        "studyTime": "20 hours",
        "why": "Reproduction is the baseline for research credibility. If you can't reproduce, you can't extend.",
        "unlocks": [
          "Finding paper errors",
          "ablation studies",
          "novel contributions"
        ]
      },
      {
        "slug": "ablation-study-design",
        "title": "Ablation Study Design",
        "level": "Intermediate",
        "prerequisites": [
          "Experiment tracking",
          "Statistical testing",
          "Model training"
        ],
        "studyTime": "8 hours",
        "why": "Ablations identify which components actually drive gains vs. which are incidental. Essential for honest reporting.",
        "unlocks": [
          "Causal analysis",
          "systematic evaluation",
          "writing results sections"
        ]
      },
      {
        "slug": "hyperparameter-search-at-scale",
        "title": "Hyperparameter Search at Scale",
        "level": "Intermediate",
        "prerequisites": [
          "Bayesian optimization",
          "W&B Sweeps",
          "Ray Tune"
        ],
        "studyTime": "8 hours",
        "why": "Grid search is exponential. Intelligent search (Optuna, BOHB) finds better configs 10× faster.",
        "unlocks": [
          "Neural architecture search",
          "AutoML",
          "efficient training at scale"
        ]
      },
      {
        "slug": "dataset-curation-and-preprocessing-pipelines",
        "title": "Dataset Curation and Preprocessing Pipelines",
        "level": "Intermediate",
        "prerequisites": [
          "Pandas",
          "SQL",
          "Data formats (Parquet",
          "WebDataset)"
        ],
        "studyTime": "10 hours",
        "why": "Model quality is bounded by data quality. Dataset engineering is often the highest-leverage research task.",
        "unlocks": [
          "Data flywheel design",
          "deduplication at scale",
          "quality filtering"
        ]
      },
      {
        "slug": "writing-technical-research-reports",
        "title": "Writing Technical Research Reports",
        "level": "Intermediate",
        "prerequisites": [
          "Completing experiments",
          "LaTeX basics"
        ],
        "studyTime": "6 hours",
        "why": "A result that isn't communicated clearly doesn't exist in the research community.",
        "unlocks": [
          "Paper writing",
          "blog posts",
          "internal research memos",
          "conference submissions"
        ]
      },
      {
        "slug": "open-source-library-contribution",
        "title": "Open-Source Library Contribution",
        "level": "Intermediate",
        "prerequisites": [
          "Git",
          "Python packaging",
          "Unit testing"
        ],
        "studyTime": "10 hours",
        "why": "Contributing to HuggingFace, PyTorch, or LangChain is how research ideas become infrastructure.",
        "unlocks": [
          "Maintaining your own research library",
          "building a public research portfolio"
        ]
      },
      {
        "slug": "large-scale-pretraining-experiments",
        "title": "Large-Scale Pretraining Experiments",
        "level": "Advanced",
        "prerequisites": [
          "Distributed training",
          "Megatron-LM/NanoGPT",
          "Chinchilla scaling laws"
        ],
        "studyTime": "30 hours",
        "why": "Pretraining is where the frontier is made. Understanding scaling dynamics and training stability unlocks the ability to contribute to foundational research.",
        "unlocks": [
          "Novel architecture research",
          "data mixture optimization",
          "scaling law modeling"
        ]
      },
      {
        "slug": "scaling-law-modeling",
        "title": "Scaling Law Modeling",
        "level": "Advanced",
        "prerequisites": [
          "Statistics",
          "Pretraining experiments",
          "Power-law fitting",
          "Chinchilla"
        ],
        "studyTime": "15 hours",
        "why": "Scaling laws predict compute-optimal training before running the full experiment, saving millions in GPU hours.",
        "unlocks": [
          "Compute-optimal training strategies",
          "data-constrained scaling",
          "emergent capability prediction"
        ]
      },
      {
        "slug": "efficient-fine-tuning-research",
        "title": "Efficient Fine-Tuning Research",
        "level": "Advanced",
        "prerequisites": [
          "LoRA",
          "QLoRA",
          "PEFT",
          "Instruction tuning",
          "RLHF"
        ],
        "studyTime": "15 hours",
        "why": "Parameter-efficient methods (LoRA, adapters, prefix-tuning) define how practitioners adapt frontier models. Improvements here have outsized impact.",
        "unlocks": [
          "Novel PEFT methods",
          "merging fine-tuned models",
          "continual learning"
        ]
      },
      {
        "slug": "evaluation-research",
        "title": "Evaluation Research",
        "level": "Advanced",
        "prerequisites": [
          "Benchmark design",
          "Statistical testing",
          "LLM-as-judge",
          "Human evaluation"
        ],
        "studyTime": "15 hours",
        "why": "Benchmarks shape what the field optimizes for. Designing good evals is itself a high-impact research contribution.",
        "unlocks": [
          "New benchmark creation",
          "capability elicitation research",
          "safety evaluation"
        ]
      },
      {
        "slug": "implementing-novel-architectures",
        "title": "Implementing Novel Architectures",
        "level": "Advanced",
        "prerequisites": [
          "Transformer internals",
          "PyTorch autograd",
          "CUDA basics",
          "Flash Attention"
        ],
        "studyTime": "25 hours",
        "why": "The ability to go from paper to working implementation in days separates top researchers from everyone else.",
        "unlocks": [
          "Architecture research",
          "custom operators",
          "publishing competitive systems"
        ]
      },
      {
        "slug": "first-principles-research-taste",
        "title": "First-Principles Research Taste",
        "level": "Expert",
        "prerequisites": [
          "All prior domains",
          "2+ years of research practice"
        ],
        "studyTime": "Ongoing",
        "why": "Knowing which problems are important, tractable, and neglected is the rarest and highest-value skill in AI. It cannot be taught directly — only cultivated.",
        "unlocks": [
          "Independent research agenda",
          "novel contributions to the field"
        ]
      },
      {
        "slug": "mechanistic-interpretability-2",
        "title": "Mechanistic Interpretability",
        "level": "Expert",
        "prerequisites": [
          "Transformer internals",
          "Probing classifiers",
          "Activation patching",
          "Circuits work"
        ],
        "studyTime": "30 hours",
        "why": "Understanding what computations transformers actually implement is essential for reliability, safety, and debugging at the frontier.",
        "unlocks": [
          "Superposition research",
          "feature geometry",
          "causal tracing",
          "SAEs"
        ]
      },
      {
        "slug": "alignment-research-methods",
        "title": "Alignment Research Methods",
        "level": "Expert",
        "prerequisites": [
          "RLHF",
          "Constitutional AI",
          "Preference modeling",
          "Red-teaming",
          "Scalable oversight"
        ],
        "studyTime": "40 hours",
        "why": "The most consequential unsolved problem in AI. Researchers who understand both ML engineering and alignment theory are extremely rare and needed.",
        "unlocks": [
          "Novel alignment proposals",
          "safety evaluations",
          "debate/amplification research"
        ]
      },
      {
        "slug": "research-paper-writing-for-top-venues",
        "title": "Research Paper Writing for Top Venues",
        "level": "Expert",
        "prerequisites": [
          "Completed novel experiments",
          "Evaluation research",
          "LaTeX",
          "peer review experience"
        ],
        "studyTime": "30 hours (per paper)",
        "why": "NeurIPS/ICML/ICLR acceptance requires not just good results but clear motivation, rigorous evaluation, and precise writing. The craft is learnable.",
        "unlocks": [
          "Building a research track record",
          "PhD/lab opportunities",
          "influencing the field"
        ]
      },
      {
        "slug": "building-research-infrastructure-at-scale",
        "title": "Building Research Infrastructure at Scale",
        "level": "Expert",
        "prerequisites": [
          "Distributed systems",
          "MLOps",
          "Cluster management",
          "Cost optimization"
        ],
        "studyTime": "30 hours",
        "why": "Research groups that can iterate 10× faster than others consistently produce more impactful work. Infrastructure is a research multiplier.",
        "unlocks": [
          "Running a research lab",
          "building a research platform",
          "enabling entire research teams"
        ]
      }
    ]
  }
];
