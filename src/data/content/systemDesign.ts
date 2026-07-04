export type SDModule = {
  level: 'Beginner'|'Intermediate'|'Advanced'|'Research';
  learningObjectives: string[]; prerequisites: string[];
  diagramsNeeded: string[]; caseStudies: string[];
  handsOnProjects: string[]; interviewQuestions: string[];
};

export type SDSystem = { slug: string; number: number; name: string; modules: SDModule[] };

export const SD_SYSTEMS: SDSystem[] = [
  {
    "slug": "vector-databases",
    "number": 1,
    "name": "Vector Databases",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Explain what a vector embedding is and why similarity search requires a different data model than SQL",
          "Describe the fundamental difference between exact nearest-neighbor and approximate nearest-neighbor search",
          "Name the 3 dominant indexing families (HNSW, IVF, LSH) and what problem each solves",
          "Set up a Qdrant/Pinecone/Weaviate instance and perform a basic semantic search query"
        ],
        "prerequisites": [
          "Linear algebra: vectors, dot product, cosine similarity",
          "Basic Python",
          "What a database index is (conceptually)"
        ],
        "diagramsNeeded": [
          "Embedding pipeline: text → tokenizer → transformer → pooling → vector",
          "Vector database anatomy: index layer / storage layer / query layer",
          "ANN search path: query vector → index traversal → distance computation → k-NN result",
          "HNSW graph visualization: layers, entry point, greedy traversal"
        ],
        "caseStudies": [
          "Spotify: how song embeddings power \"Recommended for You\"",
          "Stack Overflow: semantic search over 50M questions without keyword matching",
          "Notion AI: embedding-based document retrieval for in-product search"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "What is the curse of dimensionality and how does it affect nearest-neighbor search?",
          "Why can't you use a B-tree index for vector search?",
          "Explain HNSW to a non-technical interviewer in 90 seconds",
          "When would you choose L2 distance over cosine similarity?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design a vector database from requirements: choose indexing strategy given latency, recall, and memory constraints",
          "Explain HNSW construction and search in detail (M, ef_construction, ef_search parameters)",
          "Design a filtered vector search system (metadata + vector search combined)",
          "Reason about the recall-latency tradeoff and how to tune it",
          "Choose between a dedicated vector DB and pgvector for a given use case"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Understanding of B-trees and LSM trees",
          "Basic distributed systems: replication, sharding"
        ],
        "diagramsNeeded": [
          "HNSW multi-layer graph with construction algorithm (insert node at each layer)",
          "IVF index: K-means cluster centroids + inverted lists + PQ compression",
          "Filtered search: pre-filter vs post-filter vs in-filter (HNSW with payload index)",
          "Vector DB horizontal sharding: consistent hashing, replica sets, coordinator routing",
          "Hybrid search: dense vector score + BM25 score → RRF fusion → reranked result"
        ],
        "caseStudies": [
          "Airbnb: how they combine listing embeddings with metadata filters (price range, location) — why post-filter fails at scale and why they moved to HNSW-native filtering",
          "LinkedIn: Skills graph — embedding 900M member profiles, handling real-time updates to embeddings on profile edit",
          "Elastic (Elasticsearch): migration story from BM25-only to hybrid dense+sparse search"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "You have 1B vectors, 128 dimensions. Memory budget is 32GB. Walk me through your indexing strategy.",
          "A customer complains that semantic search returns irrelevant results when they add a price filter. What's happening and how do you fix it?",
          "Compare HNSW and IVF+PQ. When do you choose each?",
          "How does product quantization work? What is the recall cost?",
          "Design vector search that supports real-time inserts at 10K/sec without taking the index offline."
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design a distributed vector database serving 100M QPS with 10ms p99 latency",
          "Implement index compression strategies: PQ, SQ, binary quantization — and reason about quality/memory tradeoffs",
          "Design multi-tenancy: per-tenant index isolation vs shared indexes with namespace filtering",
          "Build a vector DB observability stack: recall monitoring, latency percentiles, drift detection",
          "Handle embedding model updates without full re-indexing (incremental migration strategies)"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Distributed systems: consensus, replication, CAP theorem",
          "Production infra: Kubernetes, observability basics"
        ],
        "diagramsNeeded": [
          "Full production vector DB cluster: write path (primary + replicas + WAL) + read path (coordinator + shards + cache)",
          "Tiered index storage: hot (HNSW in DRAM) → warm (PQ-compressed on SSD) → cold (flat index on object storage)",
          "Multi-tenancy: tenant-per-collection vs namespace-per-tenant vs shared index with payload filtering — memory and latency profile of each",
          "Embedding pipeline at scale: batch embedding service → queue → index update worker → version-stamped index",
          "Recall monitoring dashboard: ground truth sampling, approximate recall@K tracking over time"
        ],
        "caseStudies": [
          "Pinecone: architecture of their serverless tier — how they achieve sub-10ms latency without pinning HNSW in RAM (lazy loading + caching strategy)",
          "Weaviate: multi-tenancy architecture — how they isolated 10K tenants without proportional memory overhead",
          "Meta FAISS at scale: how Instagram uses FAISS for 500M photo embeddings — sharding strategy, replication, the GPU index tier"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design a vector database for a legal discovery system: 50B documents, queries must return results in under 100ms, recall must be above 95%. Walk through every layer of the architecture.",
          "How do you handle embedding model drift? The new model produces incompatible vector spaces.",
          "Your vector DB is at 90% memory utilization. What are your options? Walk through the tradeoffs of each.",
          "How do you test recall in production without ground truth labels?",
          "A tenant's index has 500M vectors and HNSW construction is taking 6 hours. What do you change?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze the academic frontier: learned indexes for ANN search, GPU-resident HNSW, hybrid sparse-dense indexes",
          "Critique the latency vs recall Pareto frontier and identify where current algorithms fall short",
          "Design a novel indexing strategy for a specific hard constraint (streaming updates, extreme dimensionality, binary vectors)",
          "Propose a research contribution at the intersection of vector search and LLM serving"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: HNSW (Malkov 2018), FAISS (Johnson 2019), DiskANN (Microsoft 2019), SPFresh (2022), CAPS (2023)"
        ],
        "diagramsNeeded": [
          "DiskANN: SSD-resident Vamana graph with beam search — I/O pattern analysis vs HNSW",
          "Hybrid sparse-dense index: SPLADE sparse encoder + dense HNSW, unified scoring",
          "Learned index: neural network as approximate KNN oracle — training pipeline + fallback strategy",
          "Streaming HNSW: concurrent insert/delete with epoch-based garbage collection"
        ],
        "caseStudies": [
          "Microsoft DiskANN: how they serve 1B-vector search from commodity SSDs at 1ms latency — the Vamana graph construction algorithm vs HNSW construction complexity",
          "Qdrant's sparse vector support: implementing SPLADE-compatible sparse index alongside dense HNSW for hybrid search",
          "Google ScaNN: learned quantization via anisotropic quantization loss — how it outperforms standard PQ on inner product spaces"
        ],
        "handsOnProjects": [
          "What are the fundamental limits of graph-based ANN search? Is there a theoretical lower bound on the index size / recall tradeoff?",
          "DiskANN beats HNSW on billion-scale benchmarks. Why doesn't everyone use it? What are its failure modes?",
          "How would you design a vector index that supports efficient deletion without index rebuild?",
          "Describe an open research problem in vector search that you find compelling and sketch a solution."
        ],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "search-engines",
    "number": 2,
    "name": "Search Engines",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Explain the three phases of web search: crawling, indexing, retrieval",
          "Describe how an inverted index works and how BM25 scoring ranks documents",
          "Understand the difference between term-based and semantic search",
          "Know what a query pipeline looks like: query parsing → retrieval → ranking → serving"
        ],
        "prerequisites": [
          "Basic probability and statistics",
          "Familiarity with key-value data structures"
        ],
        "diagramsNeeded": [
          "End-to-end search pipeline: user query → query parser → inverted index lookup → BM25 scorer → top-K → response",
          "Inverted index structure: term → postings list (docID, TF, positions)",
          "Web crawler architecture: seed URLs → frontier queue → fetcher → parser → URL extractor → indexer",
          "Search result page (SERP) anatomy: organic results, featured snippets, ads, knowledge panels"
        ],
        "caseStudies": [
          "Google's first architecture (Brin & Page 1998): PageRank + inverted index on commodity hardware — what made it better than AltaVista",
          "Elasticsearch: how it made Lucene accessible as a distributed search platform — usage at Wikipedia for article search",
          "Algolia: how they built a developer-friendly search API for e-commerce (instant search, typo tolerance)"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Explain BM25. Why is it better than TF-IDF?",
          "What is PageRank and why did it beat keyword-only search?",
          "What happens between the user pressing Enter and seeing search results?",
          "Why is inverted index lookup O(1) for a term but O(k) for a k-word query?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design a search system from requirements: estimate index size, query throughput, indexing latency",
          "Implement a two-stage ranking pipeline: retrieval (recall) → L2 ranking (precision)",
          "Design query understanding: intent classification, entity extraction, query expansion, spell correction",
          "Reason about indexing tradeoffs: forward index vs inverted index, index compression, positional posting lists"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Basic ML: logistic regression, gradient boosting",
          "Distributed systems: partitioning, replication"
        ],
        "diagramsNeeded": [
          "Two-stage ranking: L1 (BM25 → top-1000) + L2 (LTR model → top-10) + L3 (neural re-ranker → top-3)",
          "Query understanding pipeline: spell check → tokenization → NER → intent classifier → query rewriter → expansion",
          "Index partition strategy: document-partitioned (horizontal sharding) vs term-partitioned — fanout diagrams for each",
          "Freshness architecture: real-time index (recent documents) + batch index (historical) → merged result set",
          "Relevance feedback loop: click data → implicit labels → LTR training → model deployment"
        ],
        "caseStudies": [
          "Bing: their two-phase ranking with LambdaMART for L2 and a BERT-based L3 re-ranker — how they balance latency with quality",
          "DoorDash: restaurant search with freshness signals (open now, estimated delivery time) + relevance — hybrid ranking",
          "Shopify: product search with faceted filtering + full-text search + vector search unified under a single query planner"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design a search system for Twitter/X: 500M tweets/day. How do you index real-time content?",
          "Walk me through a Learning to Rank pipeline. What features would you use?",
          "A user searches for \"python\" — they might mean the language, the snake, or the movie. How do you handle this?",
          "Explain the difference between document-partitioned and term-partitioned indexing. When does each fail?",
          "How do you measure relevance? What is NDCG and why do we use it?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design a neural search stack: dense retrieval (DPR) + sparse retrieval (BM25) + re-ranking (cross-encoder) + answer generation (RAG)",
          "Implement production query serving: latency budgets, circuit breakers, graceful degradation",
          "Design a relevance experimentation platform: A/B testing, interleaving experiments, metrics pipeline",
          "Handle search at trillion-scale: index sharding across 10K+ nodes, query routing, tail latency management"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Distributed systems: consensus, consistent hashing, tail latency patterns",
          "ML: BERT-class models, knowledge distillation"
        ],
        "diagramsNeeded": [
          "Neural ranking stack: DPR retriever → bi-encoder batch → ANN index lookup → cross-encoder re-ranker → LLM answer generation",
          "Trillion-scale index: shard map → routing layer → per-shard Lucene → result merging → global re-ranker",
          "Search experimentation platform: traffic splitter → shadow rankers → interleaving experiment → click collection → metric computation → rollout decision",
          "Freshness pipeline: Kafka event stream → incremental indexing worker → near-real-time shard updates with version stamps",
          "Search quality dashboard: MRR@10, NDCG@10 per query segment, p50/p99/p999 latency, zero-result rate, query abandonment rate"
        ],
        "caseStudies": [
          "Google MUM and Search Generative Experience: transition from ten blue links to AI-generated answers — architecture of the generative layer sitting on top of the traditional stack",
          "Pinterest: visual search — how they combine text + image queries, embedding image uploads for similarity search, personalization signals mixed with relevance signals",
          "Meilisearch: building a typo-tolerant, sub-50ms full-text search engine on a single node — architecture decisions that enabled this"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design Google Search at 1 trillion documents, 100K QPS, 200ms p99 SLA. Walk me through every system.",
          "How do you maintain index freshness for a news search engine where articles are published at 10K/minute?",
          "Your re-ranker adds 80ms of latency. The product team says you have 150ms total budget. Walk through how you optimize the stack.",
          "How would you design a search system that personalizes results per user without leaking data between users?",
          "Describe how you would evaluate whether neural re-ranking is actually improving user outcomes."
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze the frontier of learned sparse retrieval: SPLADE, DeepImpact, uniCOIL — and their relationship to BM25",
          "Design a research contribution to query understanding or re-ranking",
          "Critically evaluate multi-vector retrieval models (ColBERT) vs bi-encoders vs cross-encoders",
          "Formulate a research question about search and sketch an experimental protocol"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: ColBERT (Khattab 2020), SPLADE (Formal 2021), DPR (Karpukhin 2020), MonoT5 (Nogueira 2020), MS MARCO benchmark"
        ],
        "diagramsNeeded": [
          "ColBERT MaxSim: per-token embeddings, late interaction, MaxSim scoring vs bi-encoder single-vector retrieval",
          "SPLADE: BERT → MLM head → FLOPS-regularized sparse activations → inverted index with learned weights",
          "Generative retrieval: DSI — document ID generation from query without an explicit index — training and inference diagram",
          "Multi-stage pipeline comparison: BM25 vs DPR vs ColBERT vs monoT5 — recall@100, MRR@10, latency tradeoff chart"
        ],
        "caseStudies": [
          "Google's BERT for re-ranking (2019): how they inserted BERT into the ranking stack, what the latency cost was, how they used knowledge distillation to make it practical",
          "Facebook's DPR: why bi-encoders trained with in-batch negatives work so well — the importance of negative mining",
          "Vespa.ai's approach to combining BM25 + neural retrieval: WAND algorithm for early termination in hybrid search"
        ],
        "handsOnProjects": [
          "What are the fundamental limitations of bi-encoder retrieval and how does ColBERT address them? What does ColBERT sacrifice?",
          "Generative retrieval (DSI) represents a paradigm shift. What are the open research problems before it can scale to web-scale search?",
          "Why does SPLADE outperform BM25 on some domains and underperform on others? What does this tell us about vocabulary mismatch?"
        ],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "recommendation-systems",
    "number": 3,
    "name": "Recommendation Systems",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Explain collaborative filtering, content-based filtering, and hybrid approaches",
          "Understand the cold-start problem and strategies for solving it",
          "Know what matrix factorization is and why it works for recommendation",
          "Describe a basic recommendation pipeline: candidate generation → scoring → serving"
        ],
        "prerequisites": [
          "Linear algebra: matrix operations, SVD conceptually",
          "Basic probability: conditional probability, Bayes rule"
        ],
        "diagramsNeeded": [
          "Recommendation pipeline: user history → candidate generation → scoring → filtering → ranking → UI",
          "User-item matrix: sparse binary interactions → matrix factorization → user/item embedding space",
          "Cold-start decision tree: new user → demographics-based / trending → first interaction → collaborative filtering activates",
          "Content-based filter: item feature vector → similarity computation → top-N similar items"
        ],
        "caseStudies": [
          "Netflix Prize (2006–2009): what the winning solution (ensemble of 800+ models) revealed about the limits of matrix factorization alone",
          "Amazon \"Customers who bought this also bought\": the item-to-item collaborative filtering system that drove 35% of Amazon's revenue",
          "YouTube 2016 recommendation paper: the two-stage DNN candidate generation + ranking architecture that replaced matrix factorization"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "What is the cold-start problem? Give 3 strategies to handle it.",
          "Explain matrix factorization in 2 minutes.",
          "Why does collaborative filtering fail for niche items (the long tail problem)?",
          "What's the difference between explicit and implicit feedback?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design a two-tower neural recommendation system from data collection to serving",
          "Implement feature engineering for recommendations: user features, item features, context features",
          "Reason about the explore/exploit tradeoff: epsilon-greedy, UCB, Thompson Sampling",
          "Design an offline evaluation protocol: precision@K, recall@K, NDCG@K, coverage, novelty"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Deep learning: embeddings, feedforward networks",
          "Distributed systems: key-value stores, feature stores"
        ],
        "diagramsNeeded": [
          "Two-tower model: user tower (demographics + history → embedding) + item tower (content + metadata → embedding) → dot product → ANN retrieval",
          "Feature store architecture: raw events → stream processing → feature computation → online store (low latency) + offline store (training)",
          "Explore-exploit bandit: A/B arm assignment → UCB score computation → item selection → reward observation → posterior update",
          "Recommendation feedback loop: served item → user interaction → reward signal → online learning update",
          "Multi-objective ranking: relevance score + diversity score + freshness score → weighted combination → Pareto front"
        ],
        "caseStudies": [
          "Spotify Discover Weekly: how they combined collaborative filtering (Matrix Factorization on play history) + NLP (word2vec on playlist co-occurrence) + audio features — the three-signal architecture",
          "Pinterest: Pinnability model — training a two-tower model on 1B+ interactions with 10B items, how they handle extreme long-tail",
          "Etsy: diversity in recommendations — how they penalize showing similar items from the same shop, the effect on user session length"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Walk me through a two-tower recommendation model from raw logs to serving.",
          "How do you handle the popularity bias problem — when your model always recommends popular items?",
          "Design a recommendation system for a new e-commerce site with zero historical data.",
          "How would you detect if your recommendation model is causing a filter bubble?",
          "What's the difference between online and offline evaluation of a recommendation system?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design an industrial recommendation system handling 100M users, 1B items, 1M QPS",
          "Implement sequential recommendation: session-based models (GRU4Rec, SASRec, BERT4Rec)",
          "Design a multi-task learning ranker: simultaneously optimize CTR, watch time, share rate, and not-interested rate",
          "Build a recommendation monitoring stack: drift detection, position bias correction, counterfactual evaluation"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Sequence models: LSTM, Transformers",
          "Causal inference basics: propensity scoring, inverse probability weighting"
        ],
        "diagramsNeeded": [
          "Industrial recommendation stack: log collection → feature store → retrieval (multiple candidate generators in parallel) → L2 ranking → business rules filter → experiment layer → serving",
          "SASRec / BERT4Rec: self-attention over item interaction sequence → next-item prediction → candidate generation",
          "Multi-task ranker: shared bottom tower + task-specific heads (CTR head, LTM head, share head) → Multi-gate Mixture of Experts (MMoE)",
          "Position bias correction: propensity-weighted loss function — position propensity model training → debiased ranker training",
          "Recommendation A/B test: orthogonal experiment layers (retrieval experiment / ranking experiment / UI experiment) — how Google's experimentation framework handles this"
        ],
        "caseStudies": [
          "YouTube Deep Ranking (2019): multi-task ranking with 1B+ users — the Mixture of Experts architecture, how they decompose \"satisfied\" users from \"engaged\" users (watch time ≠ satisfaction)",
          "TikTok FYP (deep dive at this level): the dual-optimization of completion rate + like rate + follow rate + not-interested signals — how they weight them per user cohort",
          "Alibaba: real-time personalization with reinforcement learning — updating user embeddings in real time during a shopping session using recurrent models"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design YouTube's recommendation system. How do you generate candidates for 2.5B users with 800M videos?",
          "You notice your recommendation CTR is going up but user satisfaction scores are going down. What's happening and how do you fix it?",
          "How do you evaluate a recommendation system change when you can't run a proper A/B test (e.g., small user base, no control group)?",
          "Walk me through how you'd implement real-time user modeling — updating user representations as they interact with the app.",
          "What is position bias and how does it corrupt your training data?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze the frontier: contrastive self-supervised recommendation, causal recommendation, LLM-augmented recommendation",
          "Design a research experiment testing a specific hypothesis about user behavior modeling",
          "Critique current evaluation protocols and propose improvements",
          "Read and synthesize 5 papers on sequential recommendation"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: SASRec (2018), BERT4Rec (2019), P5 (2022), LLMRank (2023), RecSys survey (He 2017)"
        ],
        "diagramsNeeded": [
          "P5: pretraining 5 recommendation tasks with a T5 model as a unified text-to-text framework",
          "LLM as ranker: in-context learning with user history as context → LLM → ranked item list — analysis of why LLMs struggle with popularity bias",
          "Causal recommendation: treatment (recommendation exposure) / outcome (conversion) / confounder (user intent) — do-calculus diagram for debiased recommendation",
          "Contrastive self-supervised pre-training: item sequence → augmented views → contrastive loss → pre-trained encoder → fine-tuned recommendation model"
        ],
        "caseStudies": [
          "Meta's DLRM at 1T parameters: embedding tables too large for GPU memory — the hybrid CPU/GPU pipeline, gradient compression for embedding tables",
          "Google's \"Are We Really Making Much Progress? Revisiting, Benchmarking, and Refining Heterogeneous Graph Neural Networks\" — the case that GNN-based recommendation doesn't consistently outperform simpler baselines",
          "RecSys reproducibility crisis: papers that claimed improvements on MovieLens but couldn't be reproduced — what went wrong in evaluation"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "rag-systems",
    "number": 4,
    "name": "RAG Systems",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Explain why RAG exists: LLMs hallucinate facts they don't have in weights; retrieval grounds answers in evidence",
          "Describe the basic RAG pipeline: ingest documents → chunk → embed → store → retrieve → augment prompt → generate",
          "Know what a chunk is, why chunk size matters, and what overlap does",
          "Understand why RAG + citations is more trustworthy than parametric generation alone"
        ],
        "prerequisites": [
          "Know what an LLM is and how prompting works",
          "Know what vector databases are (System 1 Beginner)"
        ],
        "diagramsNeeded": [
          "Basic RAG pipeline: document → chunker → embedding model → vector store → query → retrieve top-K → prompt template → LLM → answer",
          "Chunking strategies: fixed-size, sentence-based, paragraph-based, semantic chunking — visual comparison",
          "Retrieval quality spectrum: BM25 only / dense only / hybrid — Venn diagram of what each finds",
          "Prompt augmentation template: [System prompt] + [Retrieved context chunks] + [User question] → [LLM answer with citations]"
        ],
        "caseStudies": [
          "Notion AI Q&A: how they built a document Q&A system over user workspaces — chunk strategy, why they chose semantic chunking over fixed-size",
          "Perplexity.ai: basic architecture of their retrieval-then-generate pipeline — how web search results become LLM context",
          "GitHub Copilot Chat: how they retrieve relevant code context from the open repository to answer questions about a codebase"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "What is RAG and why does it reduce hallucination?",
          "What is chunk overlap and why do we use it?",
          "What's the difference between retrieval precision and retrieval recall?",
          "When would RAG fail even with perfect retrieval?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design a production RAG system: ingestion pipeline, retrieval pipeline, generation pipeline, evaluation pipeline",
          "Implement advanced retrieval: HyDE, multi-query retrieval, RAG-Fusion, query rewriting",
          "Build a RAG evaluation framework: RAGAS, context precision, context recall, faithfulness, answer relevancy",
          "Reason about chunking strategy choices: document-type awareness, semantic chunking, parent-child chunks"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Vector Databases Intermediate",
          "Basic understanding of transformer models"
        ],
        "diagramsNeeded": [
          "Production RAG architecture: document ingestion (async, queue-based) → chunk pipeline → embedding service → vector store + BM25 index → hybrid retrieval → re-ranker → LLM → response + citations",
          "HyDE pipeline: user query → LLM → hypothetical document → embed hypothetical doc → retrieve real docs similar to hypothetical",
          "Multi-query retrieval: query → LLM generates 5 query variations → retrieve for each → RRF merge → top-K unique",
          "Parent-child chunking: large parent chunks (512 tokens) stored for context; small child chunks (128 tokens) embedded for retrieval — retrieve child, return parent",
          "RAGAS evaluation pipeline: question + ground truth → retrieved context (context precision, context recall) + generated answer (faithfulness, relevancy) → composite RAGAS score"
        ],
        "caseStudies": [
          "Anthropic's Contextual Retrieval: how adding a contextual header to each chunk before embedding reduced retrieval failure rate by 49% — the implementation and the results",
          "LangChain's production RAG patterns: the ensemble retriever (BM25 + vector), multi-query retriever, and document compressor — when each is appropriate",
          "Financial services RAG at JPMorgan: compliance challenges (document access control, audit trails), how they implemented per-user retrieval filtering to prevent document leakage"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Walk me through how you'd design a RAG system for a 10,000-page legal document corpus.",
          "What is RAGAS? What does each score measure?",
          "Why does HyDE improve retrieval for domain-specific queries?",
          "A user asks a question and the answer spans 3 different documents. How does your RAG system handle this?",
          "What are the failure modes of RAG? Give 5 specific scenarios where RAG fails."
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design a production RAG system handling 100K documents, multi-tenancy, 10K QPS",
          "Implement agentic RAG: multi-hop retrieval, tool use within RAG, self-correction loops",
          "Build a RAG observability stack: hallucination detection, retrieval quality monitoring, latency tracing",
          "Handle long documents: late chunking, summary hierarchies, LLM-based document maps"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Agent Systems Intermediate",
          "Production infra: async processing, message queues"
        ],
        "diagramsNeeded": [
          "Agentic RAG: query → planner agent → (decompose into sub-queries) → retrieval per sub-query → context synthesis → generator → critique agent → (if low confidence: retrieve more) → final answer",
          "Multi-hop retrieval: question → first retrieval → answer entity extraction → second retrieval on entity → combine → final answer",
          "GraphRAG: document → entity extraction → relationship extraction → knowledge graph construction → community detection → hierarchy-aware retrieval",
          "RAG observability: query log → retrieval trace (which chunks, what scores) → generation trace (prompt tokens, output tokens) → faithfulness check (LLM-as-judge) → dashboard",
          "Cache layer: query embedding → semantic cache (find similar past queries) → cache hit → return cached answer (no LLM call)"
        ],
        "caseStudies": [
          "Microsoft GraphRAG: how they built community-level summarization on top of a knowledge graph — why flat vector retrieval fails for \"what are the main themes of this corpus\" questions",
          "Perplexity Pro: multi-step reasoning with tool use — how they combine web search, code execution, and calculator within a RAG generation loop",
          "Cohere's enterprise RAG platform: document access control at retrieval time, per-user filtering, compliance audit trails in financial services"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design a RAG system for a law firm: 50M documents, strict access control (attorneys can only retrieve documents relevant to their cases), audit requirements. Walk through the full architecture.",
          "What is multi-hop retrieval? When is it necessary, and what are its failure modes?",
          "How do you detect when your RAG system is hallucinating? What signals can you use without ground truth?",
          "Your RAG system works well on questions within a single document but fails on cross-document synthesis. What's the root cause and how do you fix it?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze frontier RAG: Self-RAG, CRAG, Adaptive RAG, Multi-Vector RAG",
          "Design a research experiment comparing retrieval strategies on a specific domain",
          "Propose a novel contribution to RAG evaluation or retrieval quality",
          "Understand the open problem of long-context LLMs vs RAG"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: RAG (Lewis 2020), Self-RAG (Asai 2023), FLARE (Jiang 2023), GraphRAG (Edge 2024), Contextual Retrieval (Anthropic 2024)"
        ],
        "diagramsNeeded": [
          "Self-RAG: training pipeline (SFT on special reflection tokens) → inference (adaptive retrieval + critique) — compare to naive always-retrieve",
          "FLARE: active retrieval — generate until confidence drops below threshold → retrieve → continue generation — the confidence signal and retrieval trigger",
          "Long-context vs RAG comparison: 1M-token context model directly vs hierarchical RAG — accuracy, cost, and latency comparison",
          "Speculative RAG: draft answer → retrieve supporting evidence → verify draft → revise if unsupported"
        ],
        "caseStudies": [
          "CRAG (Corrective RAG): the retrieval evaluator that triggers web search fallback — how they trained the evaluator and what the threshold should be",
          "Atlas (Izacard 2022): end-to-end jointly trained retriever + FiD generator — how training the retriever jointly with generation beats frozen DPR",
          "Lost in the Middle (Liu 2023): the finding that LLMs perform poorly when relevant context is in the middle of a long context window — implications for chunk ordering in RAG"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "llm-serving",
    "number": 5,
    "name": "LLM Serving",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Explain the difference between LLM training and LLM inference",
          "Understand the autoregressive generation loop: one token at a time, KV cache reuse",
          "Know what throughput and latency mean in LLM serving context",
          "Describe what a KV cache is and why it's the memory bottleneck"
        ],
        "prerequisites": [
          "Know what a transformer is (attention, feedforward, layers)",
          "Basic Python + REST APIs"
        ],
        "diagramsNeeded": [
          "Autoregressive generation loop: input tokens → KV cache → attention → feedforward → next token → append → repeat",
          "KV cache memory anatomy: batch_size × num_heads × seq_len × head_dim × num_layers × 2 (K+V) × dtype_bytes",
          "Single-request serving path: API call → tokenizer → model forward pass → sampling → detokenizer → streaming response",
          "Memory hierarchy: GPU HBM (fast, scarce) → CPU DRAM (slow, abundant) — where KV cache lives"
        ],
        "caseStudies": [
          "HuggingFace Text Generation Inference (TGI): the reference architecture for serving a single Llama-3 model — batching, KV cache, streaming",
          "Ollama: how they made local LLM serving accessible on consumer hardware — quantization defaults, memory mapping",
          "OpenAI ChatGPT API: what's publicly known about request handling — batching, model routing, response streaming"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "What is a KV cache? Why does it save compute?",
          "What's the difference between TTFT and TPS (tokens per second)?",
          "Why does LLM inference get slower as the sequence gets longer?",
          "What is quantization and why does it matter for serving?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design an LLM serving system: batching strategy, concurrency control, memory management",
          "Understand continuous batching (Orca) and why it outperforms static batching",
          "Implement PagedAttention: why fragmentation destroys KV cache utilization without it",
          "Reason about quantization: INT8, INT4, GPTQ, AWQ — quality vs throughput tradeoffs"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Systems: memory management, virtual memory concepts",
          "ML: quantization awareness"
        ],
        "diagramsNeeded": [
          "Static batching waste: pad tokens wasted for short sequences in a fixed-length batch",
          "Continuous batching: token-level scheduling — new requests enter mid-batch as old ones complete",
          "PagedAttention: logical KV blocks → physical block table → non-contiguous physical memory → no fragmentation",
          "vLLM architecture: request pool → scheduler → block manager → model executor → sampler → detokenizer → streamer",
          "Quantization pipeline: FP16 weights → calibration dataset → INT4 quantization with scale factors → quantized inference"
        ],
        "caseStudies": [
          "vLLM's PagedAttention paper: how they achieved 24× more throughput than Hugging Face TGI by solving KV cache fragmentation — the OS-inspired memory management approach",
          "Orca's continuous batching: the throughput improvement from iteration-level scheduling — 36.9× over static batching in their paper",
          "Anyscale (Ray Serve): how they serve multiple LLM models with different resource requirements on the same cluster using fractional GPU allocation"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Why does naive batching in LLM serving waste GPU compute? How does continuous batching fix it?",
          "Explain PagedAttention. What problem does it solve that a contiguous KV cache has?",
          "Walk me through the memory layout of a KV cache for a 70B model serving 100 concurrent requests.",
          "What is AWQ and why does it outperform GPTQ at the same bit width?",
          "How does speculative decoding work? When does it help most?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design a production LLM serving cluster: multi-GPU tensor parallelism, pipeline parallelism, disaggregated prefill/decode",
          "Implement speculative decoding with a draft model — understand when it wins and when it loses",
          "Design a multi-model serving platform: routing, SLO management, cost optimization",
          "Build serving observability: GPU utilization, KV cache hit rate, request queue depth, SLO compliance"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Distributed systems: NCCL, AllReduce, tensor parallelism",
          "Production infra: Kubernetes, GPU cluster management"
        ],
        "diagramsNeeded": [
          "Disaggregated prefill/decode: prefill cluster (large batch, compute-bound) + decode cluster (small batch, memory-bound) + KV migration between clusters",
          "Speculative decoding: draft model generates k tokens → target model verifies in parallel → accept prefix, reject tail → net speedup analysis",
          "Multi-GPU tensor parallelism for serving: QKV split across GPUs, feedforward split, all-reduce between layers — latency modeling",
          "Production serving cluster: load balancer → router (latency-aware) → serving pods (with autoscaling) → GPU fleet → observability stack",
          "Cost optimization architecture: spot instance tier (for non-latency-sensitive batch) + on-demand tier (for interactive) + reserved tier (for stable load)"
        ],
        "caseStudies": [
          "Google's SpecInfer (2024): how speculative decoding with tree verification achieves 2.5× throughput improvement for LLM serving at scale",
          "Anyscale's disaggregated serving: separating prefill and decode phases onto different hardware — why prefill and decode have fundamentally different hardware requirements (compute-bound vs memory-bound)",
          "Together.ai's multi-tenant serving: how they serve 50+ models to thousands of concurrent users — the routing logic, model co-location, and preemption strategy"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design ChatGPT's serving infrastructure for 10M daily active users. How do you handle load spikes during viral moments?",
          "What is disaggregated prefill/decode and why does it improve utilization?",
          "Walk through speculative decoding step by step. Why does the acceptance rate determine speedup? When does it degrade?",
          "How do you decide between scaling vertically (bigger GPUs) vs horizontally (more GPUs) for your serving cluster?",
          "Your LLM serving costs are 3× higher than a competitor. Walk through a systematic approach to reducing them."
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze frontier serving research: Sarathi-Serve, DistServe, MegaScale, Mooncake",
          "Design a serving system optimization for a novel hardware constraint",
          "Understand the emerging disaggregated inference paradigm and its limitations",
          "Propose a research contribution to improve LLM serving efficiency"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: vLLM (2023), Orca (2022), Sarathi-Serve (2024), DistServe (2024), Mooncake (2024)"
        ],
        "diagramsNeeded": [
          "Sarathi-Serve: chunked prefill — splitting long prefill across multiple iterations to prevent head-of-line blocking in decode",
          "DistServe: memory pool disaggregation — global KV pool accessible by any decode worker, eliminating the memory-compute coupling",
          "Mooncake: KV cache-centric scheduling across heterogeneous hardware (SSD + DRAM + HBM) — the prefetch pipeline",
          "MegaScale architecture: ByteDance's LLM training/serving infrastructure at 10K+ GPU scale — the communication topology"
        ],
        "caseStudies": [
          "Chunked prefill in Sarathi-Serve: the head-of-line blocking problem in mixed workloads (long prompts + short prompts), how chunking solves it, the optimal chunk size selection",
          "DistServe's evaluation: under what workload distributions does disaggregated prefill/decode win? When does the KV migration cost exceed the benefit?",
          "PD (Prefill-Decode) disaggregation in production at ByteDance: the engineering challenges of maintaining consistency when KV cache moves between machines"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "agent-systems",
    "number": 6,
    "name": "Agent Systems",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Define what an AI agent is vs a chatbot vs a pipeline",
          "Understand the ReAct loop: Reason → Act → Observe → Reason...",
          "Know the 4 core agent components: perception, memory, planning, action",
          "Describe tool use: what a tool is, how an agent decides to use it, how it handles the result"
        ],
        "prerequisites": [
          "Know how LLMs work and what prompting is",
          "Basic Python"
        ],
        "diagramsNeeded": [
          "ReAct loop: user task → LLM reasons → tool call → tool result → LLM reasons again → final answer",
          "Agent component diagram: perception (input parsing) + memory (context/retrieval) + planning (next action) + action (tool execution)",
          "Tool definition schema: name, description, parameters (JSON Schema), return type — how the LLM \"sees\" tools",
          "Agent vs pipeline comparison: pipeline (fixed DAG of steps) vs agent (dynamic decision about next step)"
        ],
        "caseStudies": [
          "OpenAI ChatGPT with Code Interpreter: the simplest real-world agent — write code, execute it, observe output, revise",
          "LangChain's basic agent: how the ReAct prompt template structures tool use for a simple search + calculator agent",
          "Anthropic's Claude with tool use: the computer use agent — how perception (screenshot) + action (click/type) creates a GUI agent"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "What's the difference between an LLM pipeline and an LLM agent?",
          "Explain the ReAct framework. Why does showing reasoning traces improve reliability?",
          "What are the 4 failure modes of basic agents?",
          "How does a tool call work in the Claude / OpenAI API?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design a multi-agent system: orchestrator + specialized sub-agents + shared memory",
          "Implement agent memory architecture: in-context / external / episodic / semantic memory",
          "Handle agent reliability: retry logic, error recovery, graceful degradation",
          "Design an agent evaluation framework: task completion rate, cost per task, step efficiency"
        ],
        "prerequisites": [
          "Beginner module complete",
          "RAG Systems Beginner",
          "Async Python (async/await)"
        ],
        "diagramsNeeded": [
          "Multi-agent orchestrator pattern: planner agent → task decomposition → assigns sub-tasks to specialized agents (researcher, coder, critic) → result synthesis",
          "Agent memory hierarchy: working memory (context window) → episodic memory (external store of past interactions) → semantic memory (vector search over knowledge) → procedural memory (stored tools/functions)",
          "Agent reliability stack: tool call → retry with exponential backoff → fallback tool → error handler → graceful answer with uncertainty",
          "Reflexion loop: agent acts → evaluator scores result → agent reflects on failure → agent retries with revised plan",
          "Cost tracking: per-agent token counter → per-task total cost → budget enforcement → auto-stop when budget exceeded"
        ],
        "caseStudies": [
          "AutoGPT (2023) failure analysis: what went wrong — infinite loops, inability to recognize when done, cost explosion — and how later systems (OpenDevin, SWE-agent) fixed these",
          "Devin: the first agent to score >13% on SWE-bench — the memory architecture (persistent workspace + scrollback), tool set (bash + browser + code editor), and planning approach",
          "Meta's Toolformer → Meta Llama with tool use: how they progressed from teaching models to use tools via self-supervised data to native function calling"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design a coding agent that can solve LeetCode problems. What tools does it need? What memory architecture?",
          "How do you prevent an agent from getting into an infinite loop?",
          "Walk me through agent memory. What are the four types and when do you use each?",
          "How do you evaluate whether an agent system is reliable enough for production?",
          "An agent makes a tool call that causes a side effect (sends an email, deletes a file). How do you design safe agent actions?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design a production multi-agent platform: agent lifecycle management, sandboxing, observability",
          "Implement agent planning: hierarchical task networks, MCTS-based planning, plan verification",
          "Handle agent security: prompt injection defense, tool sandboxing, capability limiting",
          "Build agent cost optimization: model routing (cheap model for simple steps, expensive for complex), caching, parallel subtask execution"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Systems security basics",
          "Distributed systems: containerization, message queues"
        ],
        "diagramsNeeded": [
          "Production agent platform: API gateway → agent orchestrator → task queue → agent worker pool (sandboxed containers) → tool registry → result aggregator → audit log",
          "Prompt injection defense: user input sanitization → tool output sanitization → canary token injection → anomaly detection on agent reasoning traces",
          "Hierarchical planning: high-level plan (HTN) → task decomposition → subtask parallelization → dependency graph execution",
          "Agent sandboxing: Docker container per agent run → restricted network access → file system isolation → resource limits (CPU, memory, time)",
          "Model router: complexity classifier (fast, cheap model) → route simple steps to Haiku, complex steps to Opus → per-step cost tracking"
        ],
        "caseStudies": [
          "SWE-agent architecture: the Agent-Computer Interface (ACI) — custom tool set designed for software engineering (not generic bash) — why custom tools dramatically improve task completion vs raw terminal access",
          "OpenDevin (2024): the open-source Devin — how they implemented the browser + code editor + bash environment with Docker isolation — security considerations",
          "LangGraph: DAG-based agent orchestration with explicit state machines — why graph-based orchestration prevents the runaway agent problem"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design a customer service agent for an airline: access to booking system, policy documents, payment processing. What are the failure modes? How do you sandbox it?",
          "How do you prevent prompt injection in an agent that reads external web pages as part of its task?",
          "Walk me through the architecture of a coding agent that can complete SWE-bench tasks.",
          "How do you parallelize subtasks in a multi-agent system safely?",
          "A user asks your agent to \"do whatever it takes\" to complete a task. How does your system prevent this from becoming dangerous?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze frontier agent research: world models for planning, process reward models for agent verification, scalable oversight",
          "Understand the fundamental unsolved problem of reliable long-horizon planning",
          "Design a research experiment on agent evaluation: metrics beyond task completion rate",
          "Propose a novel contribution to agent architecture or evaluation"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: ReAct (2022), Reflexion (2023), LATS (2023), SWE-agent (2024), AgentBench (2023), Scalable Oversight (Bowman 2022)"
        ],
        "diagramsNeeded": [
          "Process reward model for agents: training PRMs to evaluate intermediate agent steps (not just final outcomes) — the verification-guided planning loop",
          "Scalable oversight: agent with human spot-checking — debate protocol, amplification, recursive reward modeling",
          "World model for planning: agent maintains a learned environment model → simulates plans before executing → selects highest-value plan — similar to MuZero applied to agentic tasks",
          "Constitutional agent: agent with embedded value system — self-critique against principles before executing potentially harmful actions"
        ],
        "caseStudies": [
          "o1/o3 reasoning as agent planning: how inference-time search over reasoning chains relates to traditional agent MCTS — convergence in approach",
          "Voyager's skill library: the emergent curriculum — what the 100th skill looks like vs the 10th skill, how complexity accumulates — lessons for open-ended agent design",
          "METR's evaluations: how they test whether AI agents can autonomously replicate AI research — the task design challenges"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "youtube-architecture",
    "number": 7,
    "name": "YouTube Architecture",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Identify the 10 major subsystems of YouTube: upload, encoding, CDN, recommendation, search, ads, comments, analytics, live streaming, creator tools",
          "Explain why video storage and delivery is fundamentally different from text data",
          "Understand what a CDN does and why it's essential for video streaming",
          "Know what adaptive bitrate streaming (ABR) is and why it exists"
        ],
        "prerequisites": [
          "Basic networking: HTTP, DNS, TCP",
          "Basic databases: CRUD operations"
        ],
        "diagramsNeeded": [
          "YouTube system overview: upload path + playback path + recommendation path — the 3 major flows",
          "Video upload pipeline: client upload → ingestion → transcoding farm → multiple resolutions (240p/480p/720p/1080p/4K) → storage → CDN distribution",
          "Adaptive bitrate streaming: video player monitors bandwidth → selects highest quality segment that fits → smooth quality switching",
          "CDN architecture: origin server (YouTube data centers) → edge PoPs globally → user's ISP → viewer"
        ],
        "caseStudies": [
          "YouTube's migration from monolith to microservices (2008–2012): what broke first, the order of decomposition, what stayed monolithic longest",
          "YouTube's storage system: Google Colossus (successor to GFS) — how they store 500 hours of video uploaded every minute",
          "YouTube Shorts: the engineering challenge of adding short-form video to a platform built for long-form — new recommendation models, new storage tiers, new encoding pipelines"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "How does YouTube serve video to 2B users without every request hitting the origin server?",
          "What is adaptive bitrate streaming? Why does it exist?",
          "Design the video upload pipeline. What happens between a user clicking \"Upload\" and their video being live?",
          "Why is video storage fundamentally more expensive than text storage?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design YouTube's recommendation system at system level: candidate generation, ranking, policy enforcement",
          "Understand the search indexing challenge for video: transcripts, metadata, engagement signals",
          "Design the comments system: write path (low latency), read path (high throughput), moderation",
          "Reason about creator analytics: real-time counters (views, likes) vs eventual consistency tradeoffs"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Recommendation Systems Intermediate",
          "Distributed systems: consistency, replication"
        ],
        "diagramsNeeded": [
          "YouTube recommendation funnel: 800M videos → collaborative filtering (top 1000) → DNN ranking (top 50) → diversity + freshness filters (top 20) → UI",
          "Search pipeline: video metadata + auto-transcripts + engagement signals → inverted index → BM25 → neural re-ranker → results with thumbnails",
          "Comments system: write path (Bigtable row per video, comment as column) + read path (sorted by engagement, paginated) + moderation (ML classifier → human review queue)",
          "Real-time view counter: approximate counter (Redis increment) → debounced writes to Bigtable → periodic reconciliation with accurate count",
          "Creator Studio analytics: event stream (Kafka) → stream processing (Dataflow) → OLAP store (BigQuery) → creator dashboard with sub-minute delay"
        ],
        "caseStudies": [
          "YouTube's recommendation paper (2016): the two-tower DNN architecture that replaced matrix factorization — the 80 features used, how watch time beats clicks as a label",
          "YouTube's content ID: how they detect 99.9% of copyright infringing content in real time at upload using audio/video fingerprinting",
          "YouTube Chapters: how they extract chapter markers from video descriptions and transcripts, the accuracy challenges, the UX improvement in watch sessions"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design YouTube's real-time view counter. How do you handle a viral video getting 1M views/minute?",
          "Walk me through how YouTube generates personalized recommendations for a logged-out user (no history).",
          "How does YouTube's Content ID work at a system level?",
          "Design the YouTube comments section: write path, read path, moderation, vote counts.",
          "What happens when YouTube's recommendation system over-optimizes for watch time?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design YouTube Live: real-time streaming pipeline, sub-second latency mode, chat at scale",
          "Understand multi-region deployment: data replication strategy, disaster recovery, geo-routing",
          "Design the ads serving system: auction mechanics, targeting, billing, fraud detection",
          "Build the creator monetization system: super chat, memberships, merch shelf — transaction integrity"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Distributed databases: Spanner, Bigtable",
          "Stream processing: Kafka, Dataflow"
        ],
        "diagramsNeeded": [
          "YouTube Live pipeline: OBS → RTMP ingest → transcoding cluster → HLS/DASH packaging → CDN push → viewers (sub-second to 30-second latency modes, tradeoffs)",
          "Live chat at scale: 100K concurrent chatters → Firestore (real-time pub/sub) → spam filtering → rate limiting → rendering at client",
          "Ads auction: ad request → targeting signals → eligible ad selection → second-price auction → winner serves → impression event → billing pipeline",
          "Multi-region architecture: US / EU / APAC regions → each region serves local users → cross-region replication for metadata → regional failover with 30-second RTO",
          "Creator payment pipeline: view events → verified view count → CPM computation → monthly payout calculation → payment processor → creator bank"
        ],
        "caseStudies": [
          "YouTube's \"Super Chat\" transaction system: how they handle real-time money collection + display in chat + creator revenue split with strict financial consistency",
          "YouTube's abuse detection: how they detect view count manipulation (bot views) using behavioral signals — device fingerprinting, click patterns, IP anomalies",
          "YouTube's move to VP9 and AV1 codecs: the 35–50% bandwidth reduction at equal quality — the transcoding backlog challenge when introducing a new codec for 800M+ videos"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design YouTube Live for a major sports event with 100M concurrent viewers. How do you achieve sub-10-second latency globally?",
          "How does YouTube detect that a video is getting bot-inflated view counts?",
          "Walk me through the YouTube ads pipeline from ad request to revenue recorded in a creator's dashboard.",
          "How does YouTube ensure creators are paid correctly? What consistency guarantees does the billing system need?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze next-generation video recommendation: multimodal retrieval, video-to-video embedding, temporal modeling",
          "Understand the content moderation at scale problem: the 500 hours/minute upload rate vs human review capacity",
          "Propose a research contribution to video understanding or recommendation"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: YouTube DNN (Covington 2016), YouTube MultiTask (Zhao 2019), VideoMAE (2022), Video-LLaVA (2023)"
        ],
        "diagramsNeeded": [
          "Video embedding pipeline: frame sampling → CLIP visual encoder → audio encoder → transcript encoder → late fusion → unified video embedding",
          "Multimodal video recommendation: video embedding + user history embedding → ANN retrieval → multimodal re-ranker using both visual and behavioral signals",
          "LLM-augmented content moderation: video transcript → toxicity classifier → visual frame analyzer → LLM-based policy adjudicator → human review queue prioritization",
          "Temporal attention in video understanding: ViViT / TimeSFormer — spatial + temporal attention over frame sequence"
        ],
        "caseStudies": [
          "YouTube's VideoMAE: self-supervised video pretraining — how they learn video representations without labels by predicting masked spatiotemporal patches",
          "YouTube's automated chapters using LLMs: using transcript + title + description → LLM → chapter timestamps — the evaluation challenge (no ground truth for most videos)",
          "YouTube's responsible AI report: the algorithmic amplification problem — how recommendation systems can inadvertently promote harmful content and the technical mitigations"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "netflix-architecture",
    "number": 8,
    "name": "Netflix Architecture",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Identify Netflix's major subsystems: content delivery, recommendation, encoding, search, payments",
          "Understand why Netflix uses a microservices architecture and how it differs from a monolith",
          "Know what Chaos Engineering is and why Netflix invented it (Chaos Monkey)",
          "Describe Netflix's CDN (Open Connect) and why they built their own"
        ],
        "prerequisites": [
          "Basic networking, basic web architecture"
        ],
        "diagramsNeeded": [
          "Netflix request path: user click → AWS API gateway → microservices → recommendation service + catalog service + streaming service → video player",
          "Netflix Open Connect: ISP-hosted appliances pre-loaded with popular content → local delivery without backbone transit",
          "Content ingestion: studio master → encoding (multiple codecs + bitrates + languages) → storage (S3) → CDN distribution",
          "Microservices dependency graph: API gateway → title service → user service → recommendation service → streaming service → billing service"
        ],
        "caseStudies": [
          "Netflix's AWS migration (2008–2016): moving from a single Oracle database to 700+ microservices — the 7-year journey, what forced them off the monolith (the database failure of 2008)",
          "Netflix Open Connect: why they partnered with ISPs to put their hardware in ISP data centers — the 95% of traffic served locally, the cost reduction",
          "Chaos Monkey (2011): why Netflix deliberately kills random production instances — the philosophy of building for failure"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Why did Netflix build their own CDN instead of using Akamai or CloudFront?",
          "What is Chaos Engineering? Why is deliberately breaking production systems valuable?",
          "How does Netflix serve video to 238M subscribers simultaneously without buffering?",
          "What is a circuit breaker in microservices? When does it activate?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design Netflix's recommendation system: how they personalize the homepage for 238M users",
          "Understand Netflix's A/B testing at scale: experimentation platform, metric selection, analysis",
          "Design the billing system: payment processing, dunning management, revenue recognition",
          "Reason about multi-region active-active deployment"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Recommendation Systems Intermediate",
          "Payments basics: payment processors, webhooks"
        ],
        "diagramsNeeded": [
          "Netflix homepage personalization: user profile → recommendation models (collaborative filtering + content-based + trending) → row assembly → image personalization (which thumbnail to show) → A/B test layer → rendered homepage",
          "Netflix A/B test platform: treatment assignment (user-level hashing) → experience serving → metric collection (Kafka events) → statistical analysis (Causal Impact) → rollout decision",
          "Billing pipeline: subscription event → Zuora billing → payment processor (Stripe/Adyen) → payment result → dunning scheduler → subscriber status update",
          "Multi-region active-active: US / EU / APAC → each handles writes for local users → Cassandra cross-region replication → conflict resolution (last write wins with vector clocks)",
          "Content ranking algorithm: relevance score + freshness score + diversity constraint → ranked catalog per user per context (browsing vs searching vs autoplay)"
        ],
        "caseStudies": [
          "Netflix's recommendation thumbnails: A/B testing showed that the thumbnail shown for a title is more important than the title's actual quality for click-through — how they personalize thumbnails per user with a contextual bandit",
          "Netflix's password sharing crackdown (2023): the architecture change — device fingerprinting, household detection, account sharing detection — and the subscriber impact",
          "Netflix's \"What we watch\" transparency report: how they decided on an 18-hour threshold for defining a \"view\" — the policy and technical implications"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "How does Netflix personalize the thumbnail shown for the same title to different users?",
          "Design Netflix's A/B testing platform. How do they run 100+ experiments simultaneously without interference?",
          "Walk me through the Netflix billing system when a user's credit card payment fails.",
          "How does Netflix decide which titles to recommend in the \"Top 10 in Your Country\" row vs the \"Because You Watched X\" row?",
          "Design Netflix's search system. How is it different from Google?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design Netflix's encoding pipeline at scale: VMAF quality metrics, per-title encoding, AV1 transition",
          "Understand Netflix's observability stack: metrics (Atlas), logging, distributed tracing (Edgar)",
          "Design device-specific streaming: TVs, phones, smart TVs — different codec support, bitrate ladders, DRM requirements",
          "Handle DRM at scale: Widevine, FairPlay, PlayReady — license server architecture"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Video encoding basics: codecs, bitrate, resolution, GOP structure",
          "Cryptography basics: symmetric encryption, key management"
        ],
        "diagramsNeeded": [
          "Per-title encoding pipeline: encode quality ladder per title → VMAF quality assessment → optimize bitrate ladder for this specific content (animation vs action vs drama) → store multiple versions",
          "Netflix Edgar (distributed tracing): service calls annotated with trace IDs → trace aggregation → waterfall visualization → latency bottleneck identification",
          "DRM architecture: client → license server request → verify entitlement (subscriber has right to this content) → generate encrypted key → client decrypts and plays",
          "Netflix device matrix: 2000+ supported devices → device capability registry → per-device streaming profile (codec, max resolution, HDR support, audio format)",
          "Adaptive streaming quality: BOLA algorithm — buffer occupancy-based adaptation → select bitrate that maximizes quality given buffer level"
        ],
        "caseStudies": [
          "Netflix per-title encoding: how encoding Animated content differently from Action movies saved 20% bandwidth at equal VMAF quality — the content-aware encoding pipeline",
          "Netflix's VMAF metric: why they invented a better video quality metric than PSNR or SSIM — the human visual system model + machine learning on crowdsourced quality ratings",
          "Netflix AV1 transition: migrating 800M hours/month of streaming to AV1 — the 30% bandwidth reduction, the encoding time cost, the device support challenges"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design Netflix's encoding pipeline for a new title. How do you determine the optimal bitrate ladder?",
          "Walk me through what happens when a Netflix subscriber in Singapore clicks play on a title.",
          "How does DRM work? Explain Widevine's architecture.",
          "Netflix has 2000+ supported devices. How do they test and certify streaming quality on each device?",
          "Your CDN is having an outage in Europe. How does Netflix gracefully degrade?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze the frontier of adaptive streaming: reinforcement learning for ABR, viewport-aware encoding for VR",
          "Understand the tension between recommendation personalization and content diversity",
          "Propose a research contribution to video quality assessment or streaming optimization"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: VMAF (Li 2018), Pensieve ABR (2017), Netflix Artwork Personalization (2017), Netflix Calibrated Chaos Engineering (2020)"
        ],
        "diagramsNeeded": [
          "Pensieve RL-based ABR: state (buffer level + bandwidth history + chunk sizes) → LSTM policy network → bitrate selection → reward (SSIM + rebuffering penalty + smoothness)",
          "360° video streaming: viewport prediction → selective encoding (higher quality in predicted viewport) → client-side viewport tracking → on-demand quality switch for new viewport",
          "Federated learning for recommendation: local model updates on Netflix devices → differential privacy noise → aggregation without raw data leaving device",
          "Causal recommendation analysis: counterfactual reasoning — \"what would this user have watched without our recommendations?\" — the instrumental variable approach"
        ],
        "caseStudies": [
          "Netflix's reinforcement learning for live streaming bitrate adaptation: how they improved startup time by 20% and reduced rebuffering by 12% vs heuristic ABR",
          "Netflix's research into \"recommendation diversity vs engagement\" tradeoff — the internal study showing that users who see more diverse content have higher long-term retention despite lower short-term engagement",
          "Netflix's COVID demand spike (2020): 70% traffic increase in 2 weeks — how the auto-scaling and Open Connect handled it, what failed first"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "tiktok-architecture",
    "number": 9,
    "name": "TikTok Architecture",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Explain what makes TikTok's \"For You Page\" (FYP) fundamentally different from Instagram or YouTube recommendations",
          "Understand why TikTok can make accurate recommendations with no user history (the cold-start solution)",
          "Know what video transcoding at scale means for short-form video",
          "Describe TikTok's content moderation pipeline"
        ],
        "prerequisites": [
          "Basics of recommendation systems (what collaborative filtering is)",
          "Basic social networks: follows, likes, shares"
        ],
        "diagramsNeeded": [
          "FYP pipeline overview: new user → no history → trending pool + content signals → first personalization loop → rapid personalization after 10 interactions",
          "Short video upload pipeline: phone upload → object storage → transcoding (multiple resolutions) → CDN → worldwide delivery within 60 seconds",
          "Like/share/comment event pipeline: user action → event bus → real-time feature update → recommendation model feature refresh",
          "Content moderation tiers: automated (ML classifier) → human review queue → appeal process"
        ],
        "caseStudies": [
          "TikTok's cold-start solution: why they can make great recommendations to brand new users — the reliance on item-side signals (video content, audio, hashtags) rather than user history",
          "ByteDance's recommendation research papers: the Monolith feature system, the GDBT-based ranking, how they evolved to deep learning",
          "TikTok's global CDN: why they built a dedicated CDN for short video (different caching behavior than long-form video) — hot video 100% cache hit, long-tail delivery challenge"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "How does TikTok recommend content to a brand new user with no history?",
          "What's the difference between TikTok's FYP and Instagram's algorithm?",
          "Why is short-form video recommendation harder than long-form recommendation?",
          "What signals does TikTok use to determine if a video is \"good\"?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design TikTok's recommendation system end-to-end: candidate generation, scoring, diversity, real-time features",
          "Understand TikTok's creator monetization: how virality is incentivized, creator fund architecture",
          "Design TikTok's search: hashtag search, sound search, text search, user search",
          "Reason about the feedback loop problem: when recommendation drives creation patterns"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Recommendation Systems Intermediate"
        ],
        "diagramsNeeded": [
          "TikTok recommendation funnel: 1B+ videos → multi-channel retrieval (collaborative + content + trending) → first-stage coarse ranker → deep ranking model → policy filters (diversity, safety, repeat suppression) → 15–20 video sequence",
          "Real-time feature pipeline: user watches video → Kafka event → Flink stream processing → feature update in Redis → recommendation model reads fresh features (< 1 second delay)",
          "Creator monetization: view event → verified view (fraud filtered) → creator fund computation → daily payout calculation → payment",
          "FYP diversity controller: prevent showing 3 videos from same creator → prevent same hashtag > 2 in a row → inject 1 \"discovery\" video (outside comfort zone) per 10 videos",
          "Social graph service: follow/follower storage (Cassandra) → friend video boost → following feed vs FYP blending → creator notification service"
        ],
        "caseStudies": [
          "TikTok's \"completion rate\" signal: why watching 100% of a video is worth more than a like — the behavioral economics of attention — and how this shaped content length distribution on the platform",
          "TikTok LIVE: the real-time gift economy — virtual currency, gift animation rendering, creator earnings calculation — the system design of a real-time money transfer product",
          "Douyin (Chinese TikTok) architecture: how ByteDance runs Douyin and TikTok on shared infrastructure with content and data segmentation for regulatory compliance"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Walk me through TikTok's FYP recommendation system from user swipe to next video appearing.",
          "How does TikTok implement real-time feature updates? Why does sub-second latency matter for recommendation?",
          "Design TikTok's diversity controller. What rules prevent the feed from becoming an echo chamber?",
          "How does TikTok detect that a video is \"going viral\" before it goes viral?",
          "Design TikTok LIVE: 100K concurrent viewers per streamer, real-time gift animations, creator earnings calculation."
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design TikTok's content understanding pipeline: video understanding, audio fingerprinting, text overlay extraction",
          "Implement TikTok's safety and trust system: hate speech detection, misinformation flagging, underage user protection",
          "Design the global architecture: US/EU/APAC data sovereignty, content localization, regulatory compliance",
          "Handle trending detection: identify emerging trends in real time, amplify strategically"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Video/audio ML: CLIP, Whisper, content understanding models",
          "Compliance: GDPR, COPPA basics"
        ],
        "diagramsNeeded": [
          "Video understanding pipeline: video frames → CLIP visual encoder → audio → Whisper transcript → text overlay OCR → hashtag/caption NLP → unified video embedding",
          "Trending detection system: view velocity graph (views/minute) per video → anomaly detection → trending score → trending pool injection → global trending page",
          "Data sovereignty architecture: US user data stays in US (Oracle TikTok) → EU user data in EU (GDPR compliance) → content sharing between regions vs data isolation",
          "Trust and Safety pipeline: upload → automated content analysis (hate, NSFW, violence, spam) → risk score → auto-remove (high risk) + human queue (medium risk) + approve (low risk)",
          "Age verification: age gate at signup → behavioral signals for underage users → COPPA-compliant data handling → restricted feed for <13 users"
        ],
        "caseStudies": [
          "TikTok's Project Texas: the US data sovereignty effort — Oracle Cloud infrastructure, US-based data governance, how they separated US user data from ByteDance access",
          "TikTok's content moderation at scale: 1B users uploading 100M+ videos/day — the ratio of automated vs human moderation, regional policy differences (what's banned in Turkey vs US vs India)",
          "TikTok's sound recommendation: how the audio signal drives discovery — recommending the same audio across creators (sounds go viral before videos) — the audio embedding and trending detection system"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "How does TikTok understand the content of a video without watching it manually?",
          "Design TikTok's trending detection system. How do you distinguish \"going viral\" from \"high baseline popularity\"?",
          "TikTok is required to store EU user data exclusively in the EU. How does this affect the recommendation system?",
          "How does TikTok prevent underage users from accessing adult content?",
          "Your content moderation system is removing 0.1% of videos incorrectly (false positives). At 100M uploads/day, that's 100K wrongly removed videos. How do you handle this?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze the societal impact of recommendation algorithms: political polarization, mental health, attention fragmentation",
          "Design a research experiment to measure algorithmic amplification of harmful content",
          "Propose a technically rigorous intervention for recommendation systems that reduces harm without destroying engagement",
          "Understand the technical challenges of cross-cultural content moderation"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: Algorithmic amplification papers (Huszár 2022), TikTok's own research disclosures, FYP audit papers"
        ],
        "diagramsNeeded": [
          "Causal model of recommendation harm: recommendation → content exposure → attitude change → behavioral change — identifying the causal pathway for measurement",
          "Counterfactual recommendation audit: compare FYP feed vs chronological feed vs random feed → measure downstream metrics (time spent, sentiment, information diversity)",
          "Value-sensitive recommendation: multi-objective optimization including wellbeing proxy signals (session length declining = user unhappy) + engagement signals",
          "Cross-cultural moderation: content policy ontology with 50 categories → region-specific thresholds per category → localized human review teams → appeal translations"
        ],
        "caseStudies": [
          "Mozilla Foundation's TikTok algorithm audit (2023): their crowdsourced research on political content amplification — the methodology and findings",
          "TikTok's \"Screen Time\" research: their own internal study on whether time limits improve user wellbeing — what they found and chose not to publish",
          "Instagram/Facebook's internal research on recommendation effects on teenage girls — the parallel from Meta that informs TikTok design questions"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "uber-architecture",
    "number": 10,
    "name": "Uber Architecture",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Identify Uber's core systems: dispatch, routing, pricing, driver/rider apps, payments, fraud detection",
          "Understand the two-sided marketplace problem: matching supply (drivers) to demand (riders) in real time",
          "Know what geospatial indexing means and why it's fundamental to Uber",
          "Describe the trip lifecycle from a systems perspective: request → match → route → complete → settle"
        ],
        "prerequisites": [
          "Basic networking, databases",
          "Basic probability: supply/demand"
        ],
        "diagramsNeeded": [
          "Trip lifecycle: rider requests → dispatcher finds nearby drivers → driver accepts → route computed → navigation updates → trip completes → payment processes",
          "Geospatial index: city divided into H3 hexagonal cells → driver positions indexed by cell → radius search = query nearby cells",
          "Supply-demand matching: rider request arrives → find available drivers within 5km → rank by ETA + rating → offer to best driver → accept/decline logic",
          "Surge pricing logic: demand/supply ratio per cell → dynamic multiplier → ride price = base × surge multiplier"
        ],
        "caseStudies": [
          "Uber's migration from monolith to microservices (2014–2018): the original architecture (Python monolith, MySQL) → the business logic that drove decomposition (separate domains: dispatch, pricing, payments, driver)",
          "Uber's surge pricing PR crisis: the technical reality of dynamic pricing vs the public perception — how the algorithm actually works vs how it was explained",
          "Uber's H3 geospatial library: why they open-sourced their hexagonal hierarchical grid system and how it's now used by Airbnb, Safegraph, and others"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "How does Uber find nearby drivers efficiently? Why not just check all drivers?",
          "What is surge pricing and how does it actually work technically?",
          "Design the Uber dispatch system. What happens in the 3 seconds between a rider requesting and a driver being matched?",
          "Why is GPS accuracy a fundamental constraint on Uber's accuracy?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design Uber's dispatch system at city scale: latency requirements, matching algorithms, optimization objectives",
          "Understand Uber's ETA prediction: traffic-aware routing, ML models for historical travel time",
          "Design the payments system: ride payment, driver payout, refunds, fraud detection",
          "Reason about the driver experience: location updates, earnings visibility, trip requests"
        ],
        "prerequisites": [
          "Beginner module complete",
          "Algorithms: Dijkstra, A*, shortest path"
        ],
        "diagramsNeeded": [
          "Dispatch system: rider request → supply service (driver locations in Redis, indexed by H3) → demand service → matching service (rank drivers by ETA) → offer service (driver notification) → trip service",
          "ETA prediction: road graph (pre-computed, updated hourly) + real-time traffic (probe data from driver GPS) → graph edge weight update → Dijkstra on live graph → ETA = path cost",
          "Driver location pipeline: driver sends GPS every 4 seconds → Kafka → stream processor → Redis geo-index update (< 100ms) → dispatch service reads fresh positions",
          "Payments pipeline: trip ends → fare calculation → rider payment (Stripe) → platform fee deduction → driver earnings update → weekly settlement → bank transfer",
          "Fraud detection: GPS trace analysis (impossible speed detection) → device fingerprinting → velocity checks (same card used in 2 cities simultaneously) → rules engine + ML model"
        ],
        "caseStudies": [
          "Uber's DISCO (2017): the matchmaking system that optimized for batch matching instead of greedy matching — 20% improvement in ETA by considering global optimal assignments within 500ms",
          "Uber's Michelangelo ML platform: how they built the ML infrastructure to serve 1M+ predictions/second for surge, ETA, fraud, and demand forecasting on a unified platform",
          "Uber Eats architecture: how the same dispatch and routing infrastructure was adapted for food delivery — restaurant wait time as additional ETA component, multi-stop delivery"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Walk me through Uber's dispatch system. How does a driver get selected for a trip?",
          "How does Uber predict arrival time accurately when traffic changes minute by minute?",
          "Design the Uber payments system. How does money flow from rider to driver?",
          "Your ETA predictions are consistently 3 minutes too short on Friday evenings. How do you diagnose and fix this?",
          "How does Uber detect GPS spoofing (a driver falsifying their location to appear near airport)?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design Uber's real-time city-level supply/demand forecasting: predict demand 5 minutes ahead at H3-cell level",
          "Implement dynamic pricing optimization: not just surge, but personalized pricing, promotional discounts, driver incentives",
          "Design Uber's global architecture: 70+ countries, local regulatory compliance, localized payment methods",
          "Handle disaster scenarios: what happens when the dispatch service goes down mid-city?"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "Time series forecasting: ARIMA, Prophet, LSTM for time series",
          "Distributed systems: fault tolerance, circuit breakers"
        ],
        "diagramsNeeded": [
          "Demand forecasting pipeline: historical trip data → feature engineering (weather, events, time) → Prophet + deep learning ensemble → 5-minute demand forecast per H3 cell → dispatch system uses forecast for proactive driver positioning",
          "Driver incentive engine: real-time earnings gap analysis → identify underserved zones → incentive calculation (surge bonus) → push notification to nearest off-duty drivers → acceptance rate monitoring",
          "Global regulatory compliance layer: routing engine → country-specific rules (surge caps, minimum fares, accessibility requirements) → localized payment methods (WeChat Pay, Paytm, M-Pesa)",
          "Disaster recovery: primary dispatch cluster fails → health check fails → traffic failover to secondary region → driver location state reconstructed from Kafka replay → < 60 second recovery"
        ],
        "caseStudies": [
          "Uber's demand forecasting with Prophet: how they predict city-wide demand 5 minutes ahead — the feature importance (concerts, sports events, weather, historical patterns) and accuracy metrics",
          "Uber's driver positioning: proactive driving incentives that position drivers near predicted demand before it materializes — 15% reduction in rider wait time",
          "Uber's India launch: custom architecture for 2G network reliability — SMS-based fallback, reduced GPS accuracy handling, cash payment integration"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design Uber's demand forecasting system. How do you predict ride demand 15 minutes ahead at a city-block level?",
          "The dispatch service goes down in a major city. Walk through the failure cascade and recovery procedure.",
          "How does Uber decide where to send driver incentive notifications to maximize coverage of demand?",
          "Design Uber's international payment architecture: 70 countries, 50 local payment methods, multiple currencies.",
          "How does Uber prevent a race condition where 2 drivers are assigned to the same rider?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze the frontier of marketplace optimization: multi-objective matching, counterfactual demand estimation, equilibrium pricing",
          "Design a research experiment measuring the social welfare effects of ride-hailing algorithms",
          "Understand the open problem of autonomous vehicle dispatch: how Uber's architecture changes when vehicles drive themselves"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: Uber DISCO (2017), Uber DeepETA (2022), Uber Michelangelo (2017), two-sided market theory papers"
        ],
        "diagramsNeeded": [
          "Multi-objective matching: rider ETA + driver idle time + platform revenue + environmental cost (routing to minimize emissions) → Pareto optimal matching",
          "AV dispatch architecture: AV fleet management (charge level, maintenance status) + ride demand + routing → dispatch with vehicle state constraints",
          "Counterfactual demand estimation: in a surge scenario, what would demand have been without the surge (price elasticity measurement) — instrumental variable approach using weather as instrument"
        ],
        "caseStudies": [
          "Uber's DeepETA (2022): a graph neural network that models road network as a graph, achieved 15% MAPE improvement over gradient boosting — the architecture and training setup",
          "Lyft's RL-based dispatch: how they framed matching as a reinforcement learning problem, with reward = driver earnings + rider satisfaction — the training environment setup",
          "Shared rides (Uber Pool / UberX Share): the matching algorithm for grouping riders — the NP-hard nature of the problem, the approximation algorithms used, why shared rides remain a hard product problem"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "chatgpt-architecture",
    "number": 11,
    "name": "ChatGPT Architecture",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Explain what ChatGPT is architecturally: a fine-tuned GPT-4 with RLHF, served via an API and chat interface",
          "Understand the three training stages: pretraining → SFT → RLHF",
          "Know what session management means for a stateless LLM: how context is passed back with each request",
          "Describe the streaming response architecture: tokens appearing one by one"
        ],
        "prerequisites": [
          "Know what an LLM is and how transformers work at a conceptual level",
          "Basic web architecture: HTTP, WebSockets"
        ],
        "diagramsNeeded": [
          "ChatGPT system overview: browser → OpenAI API → model serving cluster → GPT-4 → streaming response → browser renders tokens",
          "RLHF training pipeline: GPT-4 base → SFT on human demonstrations → reward model from human comparisons → PPO optimization",
          "Session context management: conversation history → token counting → context window management → truncation when approaching limit",
          "Streaming token delivery: LLM generates token → Server-Sent Events (SSE) → browser receives partial response → renders incrementally"
        ],
        "caseStudies": [
          "ChatGPT's launch (November 2022): the 1M users in 5 days moment — what the serving infrastructure had to handle, what actually broke (capacity limits, rate limiting)",
          "OpenAI's API architecture: how the same underlying models serve both ChatGPT.com and the developer API with different SLOs",
          "GPT-4V's multimodal serving: what changes when the input includes images — the vision encoder, increased compute per request, image upload handling"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "What's the difference between GPT-4 (base) and ChatGPT?",
          "How does ChatGPT remember what you said 10 messages ago?",
          "What happens architecturally when you type a message in ChatGPT and see the response appear word by word?",
          "Why does ChatGPT have a context limit?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design ChatGPT's serving infrastructure: latency budget, concurrency model, load balancing",
          "Understand OpenAI's model routing: GPT-4o vs GPT-4o-mini — the cost/quality routing decision",
          "Design the safety layer: content policy enforcement before generation and after generation",
          "Reason about the memory problem: why ChatGPT doesn't remember conversations across sessions"
        ],
        "prerequisites": [
          "Beginner module complete",
          "LLM Serving Intermediate",
          "Basic safety/alignment concepts"
        ],
        "diagramsNeeded": [
          "OpenAI serving stack: API gateway → rate limiter → model router (GPT-4o vs mini based on complexity) → GPU cluster → output filter → response",
          "Safety pipeline: user input → moderation classifier (before sending to model) → model generation → output moderation (after generation) → return to user",
          "Conversation memory architecture: in-context memory (within session) → memory feature (explicitly stored facts) → custom instructions (persistent user preferences)",
          "Model routing logic: request complexity classifier → latency SLO → cost threshold → route to GPT-4o-mini (simple) or GPT-4o (complex)",
          "Rate limiting: per-user token bucket (requests/minute + tokens/minute) → tier-based limits → queue for burst → 429 response when exceeded"
        ],
        "caseStudies": [
          "OpenAI's outage retrospectives: the Dec 2022 capacity crisis, the March 2023 partial outage — what failed, how they communicated, what mitigations they put in place",
          "ChatGPT's memory feature (2024): the architecture of long-term memory — how they decide what to save, where it's stored, how it's retrieved and injected into context",
          "OpenAI's Custom Instructions: the persistent system prompt feature — how user preferences are stored and prepended to every conversation"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "How does OpenAI decide whether to route your request to GPT-4o vs GPT-4o-mini?",
          "Walk me through the safety stack in ChatGPT. How does OpenAI prevent the model from producing harmful content?",
          "Design a conversation memory system for ChatGPT. What should be remembered? What should be forgotten?",
          "How does ChatGPT handle a user who is consuming 1000× more API calls than average?",
          "What is the \"alignment tax\" — the tradeoff between making a model safer and making it more capable?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design ChatGPT's multi-modal serving: text + vision + audio + image generation in a unified interface",
          "Understand the tool use serving architecture: code interpreter sandboxing, web search integration, function calling",
          "Design a production safety evaluation system: red-teaming, automated safety benchmarks, canary testing",
          "Handle OpenAI's scale: 100M+ daily users, multi-model serving, cost optimization at $1B+ annual inference spend"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "LLM Serving Advanced",
          "Multi-modal models: GPT-4V, Whisper, DALL-E"
        ],
        "diagramsNeeded": [
          "GPT-4o unified multi-modal serving: audio input → Whisper ASR → text → GPT-4o → text output → TTS output / code → sandbox → output",
          "Code Interpreter architecture: generated Python code → Docker sandbox → execution with timeout → stdout/stderr/plots → returned to model context",
          "Web search integration: model decides to search → tool call to search API → results returned as context → model answers with citations",
          "Production safety evaluation pipeline: new model checkpoint → automated safety benchmark (MMLU, TruthfulQA, safety evals) → red-team eval → canary deployment (1% traffic) → full deployment",
          "Cost optimization: prefix caching (reuse KV cache for common system prompts) → batching similar requests → quantization for speed → spot GPU instances for batch jobs"
        ],
        "caseStudies": [
          "OpenAI's Code Interpreter safety: the sandboxing approach — why they chose Docker, what operations are blocked (network, filesystem persistence), how they handle the 10-minute timeout",
          "ChatGPT's \"memory\" rollout: the A/B test results showing that users with memory enabled have higher retention — the privacy considerations that delayed rollout",
          "OpenAI's red-teaming process: how they use both automated and human red-teaming to find model failure modes before deployment — the pipeline from finding to fixing"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Design ChatGPT's Code Interpreter. How do you sandbox user-generated code safely?",
          "How does OpenAI test whether a new model is safer than the previous one?",
          "Walk me through how ChatGPT's web search integration works architecturally.",
          "OpenAI's inference costs are $1B+ per year. What are the top 5 optimization levers they use?",
          "How do you A/B test a new model in ChatGPT without exposing users to a worse experience?"
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze the alignment research frontier: Constitutional AI, RLHF vs DPO, scalable oversight, interpretability",
          "Understand the open problem of measuring AI capability vs alignment simultaneously",
          "Design a research contribution to safety evaluation methodology",
          "Analyze emergent capabilities: what abilities appear suddenly at scale and why"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: InstructGPT (2022), Constitutional AI (2022), DPO (2023), Measuring Massive Multitask Language Understanding (2021), Sparks of AGI (2023)"
        ],
        "diagramsNeeded": [
          "DPO vs RLHF comparison: RLHF (reward model + PPO) vs DPO (direct optimization on preference pairs without explicit reward model) — training stability and data efficiency",
          "Scalable oversight: debate protocol — two AI models argue for/against claims, human judge (with limited time) evaluates — how this scales human oversight to superhuman tasks",
          "Interpretability pipeline: activation analysis → sparse autoencoders → feature visualization → circuit tracing — the mechanistic interpretability toolchain",
          "Emergent capability measurement: per-task accuracy across model scales → emergent threshold detection → sharp phase transition identification"
        ],
        "caseStudies": [
          "DPO vs RLHF empirical comparison: where DPO wins (simpler, more stable) vs where RLHF wins (fine-grained reward modeling for complex tasks) — the Llama 3 instruction tuning report",
          "Anthropic's interpretability research: golden gate Claude — how they found and steered a feature representing the \"Golden Gate Bridge\" in Claude's residual stream",
          "The emergent capabilities debate: Schaeffer et al. (2023) arguing that emergence is a measurement artifact vs Wei et al. defending the phenomenon — the implications for capability prediction"
        ],
        "handsOnProjects": [],
        "interviewQuestions": []
      }
    ]
  },
  {
    "slug": "deepseek-architecture",
    "number": 12,
    "name": "DeepSeek Architecture",
    "modules": [
      {
        "level": "Beginner",
        "learningObjectives": [
          "Explain what makes DeepSeek architecturally novel vs GPT-4: MoE, MLA, DeepSeek-R1's reasoning approach",
          "Understand the cost story: DeepSeek-V3 trained for $6M vs GPT-4's estimated $100M — what made this possible",
          "Know what Mixture of Experts is and why it enables larger models without proportional compute increase",
          "Describe what Multi-Head Latent Attention (MLA) is at a conceptual level"
        ],
        "prerequisites": [
          "Know what a transformer is (attention + feedforward)",
          "Basic scaling intuition: bigger model = more compute"
        ],
        "diagramsNeeded": [
          "MoE vs dense transformer: dense (all feedforward neurons active) vs MoE (top-K experts selected, others inactive) — FLOPs comparison",
          "DeepSeek-V3 high-level architecture: transformer blocks with MoE feedforward layers + MLA attention",
          "DeepSeek-R1 reasoning: base model → reasoning traces via RL → chain-of-thought before answering — the \"think before you speak\" architecture",
          "Cost breakdown: pretraining compute (GPU-hours) → hardware (H800 cluster) → total cost — DeepSeek vs estimated GPT-4"
        ],
        "caseStudies": [
          "DeepSeek's disruption moment (Jan 2025): how a Chinese lab trained a GPT-4-class model for 1/10th the cost — the Nvidia stock drop, the geopolitical implications, what the technical report revealed",
          "DeepSeek MoE architecture: why they use 256 experts with top-8 selection — the routing algorithm and load balancing challenges",
          "DeepSeek-R1: the first open-source model to approach o1-level reasoning — how pure reinforcement learning without supervised reasoning traces produced chain-of-thought behavior"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "What is Mixture of Experts? How does it allow a 671B parameter model to use only 37B active parameters?",
          "Why was DeepSeek-V3's training cost so much lower than GPT-4?",
          "What is DeepSeek-R1 and how is it different from DeepSeek-V3?",
          "What is Multi-Head Latent Attention (MLA) and what problem does it solve?"
        ]
      },
      {
        "level": "Intermediate",
        "learningObjectives": [
          "Design a Mixture of Experts LLM from architecture choices to training recipe",
          "Understand DeepSeek's MLA: how compressing the KV cache into a latent vector reduces memory 6× without quality loss",
          "Implement expert load balancing: why top-K routing without balancing collapses onto 2-3 popular experts",
          "Reason about DeepSeek-R1's RL training: GRPO algorithm, reward function design, the cold-start problem in reasoning RL"
        ],
        "prerequisites": [
          "Beginner module complete",
          "LLM Serving Intermediate",
          "Reinforcement learning basics (policy gradient)"
        ],
        "diagramsNeeded": [
          "MoE layer internals: token → router (softmax over expert logits) → top-K selection → dispatch to K expert feedforward networks → weighted combine → output",
          "Load balancing auxiliary loss: count tokens per expert → compute load imbalance metric → add auxiliary loss term to training objective → force equal expert utilization",
          "MLA architecture: [Q, K, V] → compress (K, V) into low-rank latent vector cKV → store compressed KV cache → decompress during attention → 6× memory reduction",
          "DeepSeek-R1 training pipeline: base model → GRPO with math/code reward → reasoning model (no SFT) → alternatively: base model → cold-start SFT on reasoning traces → GRPO → rejection sampling → SFT → GRPO",
          "GRPO algorithm: group of output samples → compute relative rewards → group-normalized advantage → policy gradient update"
        ],
        "caseStudies": [
          "DeepSeek MoE routing: the fine-grained expert design (256 experts vs 8 in Mixtral) — why more finer-grained experts with larger top-K works better — the token routing patterns observed in their paper",
          "DeepSeek's hardware-aware design for H800 (not H100): the A100/H800 has half the NVLink bandwidth of H100 — their communication-computation overlap strategy compensated for this",
          "DeepSeek-R1-Zero: training only with RL (no SFT cold start) — the emergent behaviors they observed (self-correction, \"aha moment\" when switching strategies mid-solution)"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Explain MoE routing. What happens if you don't use load balancing? How does the auxiliary loss fix this?",
          "What is Multi-Head Latent Attention? Draw the key and value computation with and without MLA.",
          "Walk me through DeepSeek-R1's training procedure. Why does pure RL (without SFT) work for reasoning?",
          "What is GRPO and how does it differ from PPO for LLM training?",
          "Why does DeepSeek achieve lower inference cost than a dense model of the same total parameter count?"
        ]
      },
      {
        "level": "Advanced",
        "learningObjectives": [
          "Design DeepSeek-V3's training infrastructure: FP8 mixed precision, DualPipe pipeline parallelism, all-to-all communication for MoE",
          "Understand DeepSeek's speculative decoding with MTP (Multi-Token Prediction): how they generate 2+ tokens per forward pass",
          "Design the serving infrastructure for a 671B MoE model: tensor parallelism, expert parallelism, KV cache management for MLA",
          "Analyze the cost structure: why $6M training cost was achievable — hardware utilization, precision, pipeline efficiency"
        ],
        "prerequisites": [
          "Intermediate module complete",
          "LLM Serving Advanced",
          "Mixed precision training: FP16, BF16, FP8"
        ],
        "diagramsNeeded": [
          "FP8 mixed precision training: forward pass in FP8 → loss in BF16 → backward pass in FP8 → weight update in BF16 → FP8 quantization errors and how they accumulate",
          "DualPipe: two pipeline directions (forward on one micro-batch, backward on another) interleaved — computation-communication overlap — bubble reduction vs 1F1B",
          "Expert parallelism for serving: 256 experts distributed across 32 GPUs → all-to-all dispatch (token routing across GPUs) → expert compute → all-to-all combine → next layer",
          "MTP (Multi-Token Prediction) for speculative decoding: main model outputs next token + draft head outputs next-2 token prediction → verify in single forward pass → 1.8× throughput",
          "DeepSeek serving cost analysis: active parameters per forward pass (37B) × FP8 × H800 FLOPS → tokens/second/GPU → cost per million tokens"
        ],
        "caseStudies": [
          "DeepSeek's FP8 training stability: how they maintained stable training at FP8 precision — the tile-wise quantization approach, the gradient scaling strategy, where they fell back to BF16",
          "DualPipe vs 1F1B pipeline: DeepSeek's custom pipeline schedule that reduced bubble rate to near-zero — the communication-computation overlap that made it work on H800 (limited NVLink)",
          "DeepSeek-V3 vs Llama 3.1 serving cost: at equal output quality, DeepSeek-V3 is 10-20× cheaper to serve — the breakdown of where the savings come from (MoE active parameters + MLA KV cache + MTP)"
        ],
        "handsOnProjects": [],
        "interviewQuestions": [
          "Walk me through DeepSeek-V3's training stack: what's novel about the FP8 training, the pipeline schedule, and the MoE implementation?",
          "How do you serve a 671B parameter MoE model with only 37B active parameters? What does the GPU layout look like?",
          "Explain Multi-Token Prediction for speculative decoding. How is it different from a separate draft model?",
          "What is DualPipe and why does it matter more on H800 than H100?",
          "DeepSeek cost $6M to train vs GPT-4's $100M estimate. Walk through 5 specific engineering decisions that explain the gap."
        ]
      },
      {
        "level": "Research",
        "learningObjectives": [
          "Analyze the fundamental research contributions of DeepSeek-R1: RL for reasoning without reasoning supervision",
          "Design an experiment to understand why MoE scaling laws differ from dense scaling laws",
          "Propose a novel contribution to efficient LLM training or architecture",
          "Understand the geopolitical and competitive implications for AI research — and why open weights matter"
        ],
        "prerequisites": [
          "Advanced module complete",
          "Read: DeepSeek-V3 Technical Report (2024), DeepSeek-R1 Technical Report (2025), MoE survey (2022), GRPO paper"
        ],
        "diagramsNeeded": [
          "DeepSeek-R1 emergent reasoning analysis: distribution of response formats (thinking tokens vs direct answer) before vs after RL training — what the RL pressure produces",
          "MoE scaling laws: dense scaling (chinchilla) vs MoE scaling (evidence from DeepSeek, Mixtral, Switch Transformer) — do the exponents differ?",
          "Reasoning RL generalization: does RL training on math reasoning generalize to coding? Science? Commonsense? — transfer learning diagram",
          "Open weights impact diagram: DeepSeek weights release → fine-tuning community → specialized models (medical, legal, code) → 6-month capability cascade"
        ],
        "caseStudies": [
          "DeepSeek-R1 ablations: with cold start vs without, with rejection sampling vs without — the paper's own ablations that reveal which components are load-bearing",
          "The distillation result: DeepSeek-R1 distilled into Qwen-7B and Llama-3-8B — that a 7B model trained on DeepSeek-R1 outputs beats o1-mini — the implications for scaling vs distillation as a capability vector",
          "Community response to DeepSeek-R1 release: the fine-tuning wave (medical reasoning, code reasoning, math tutoring models) — what open weights enable that closed weights don't"
        ],
        "handsOnProjects": [
          "DeepSeek-R1 showed that RL alone (without supervised reasoning traces) can produce strong chain-of-thought reasoning. What is the theoretical explanation? Does this scale?",
          "MoE models have 2× the parameters for the same FLOPs as a dense model. Does the extra capacity consistently help? What do the scaling law papers say about this?",
          "If you were at a lab competing with DeepSeek, what architectural or training innovation would you pursue to reclaim the cost advantage?",
          "What are the open research problems in MoE training stability? Why do experts sometimes collapse?",
          "Propose a follow-up experiment to DeepSeek-R1's paper that could falsify or confirm their claims about RL-emergent reasoning."
        ],
        "interviewQuestions": []
      }
    ]
  }
];
