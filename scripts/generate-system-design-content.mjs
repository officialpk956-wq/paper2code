#!/usr/bin/env node

/**
 * Builds the canonical long-form system-design articles from SD_SYSTEMS.
 * The TypeScript metadata is the checklist; profiles below supply the teaching
 * model, math, operational vocabulary, and primary-source reading direction.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = fs.readFileSync(path.join(ROOT, "src/data/content/systemDesign.ts"), "utf8");
const literal = source.match(/export const SD_SYSTEMS[^=]*=\s*(\[[\s\S]*\]);/)?.[1];
if (!literal) throw new Error("Could not parse SD_SYSTEMS");
const systems = Function(`"use strict"; return (${literal});`)();

const profiles = {
  "vector-databases": {
    description: "Design vector-search infrastructure from embedding ingestion through billion-scale approximate nearest-neighbor retrieval, filtering, sharding, and online operations.",
    tags: ["vector-search", "ann", "embeddings", "hnsw"],
    plain: "A vector database stores learned representations and retrieves items whose vectors are close to a query vector. Its job is to make that similarity operation fast, filterable, fresh, and dependable at a scale where comparing every vector is impossible.",
    components: ["embedding API", "write log", "vector store", "ANN index", "metadata index", "query coordinator", "reranker"],
    constraints: "recall, tail latency, memory, write freshness, filtering selectivity, and embedding-version compatibility",
    formula: String.raw`For vectors $x,y\in\mathbb{R}^d$, cosine similarity is $s(x,y)=\frac{x^\top y}{\lVert x\rVert_2\lVert y\rVert_2}$. A raw float32 collection needs approximately $4Nd$ bytes before graph edges, metadata, replicas, and allocator overhead. Exact scan is $O(Nd)$ per query; ANN trades exactness for a far smaller candidate set.`,
    baseline: "exact brute-force k-nearest-neighbor search on a labeled sample",
    sources: ["Malkov and Yashunin, Efficient and Robust Approximate Nearest Neighbor Search Using HNSW (2018)", "Jégou, Douze, and Schmid, Product Quantization for Nearest Neighbor Search (2011)", "FAISS documentation and research papers"],
  },
  "search-engines": {
    description: "Build lexical, semantic, and hybrid search systems covering crawling, indexing, ranking, freshness, personalization, and web-scale operations.",
    tags: ["information-retrieval", "bm25", "ranking", "indexing"],
    plain: "A search engine converts a short, ambiguous query into an ordered set of useful documents. It must discover or ingest content, build searchable indexes, retrieve candidates cheaply, rank them accurately, and explain operational failures such as stale or missing results.",
    components: ["crawler/ingest", "parser", "inverted index", "dense index", "query understanding", "candidate retrieval", "ranker"],
    constraints: "relevance, freshness, p99 latency, index size, abuse resistance, and privacy",
    formula: String.raw`BM25 scores a document $D$ for query term $q_i$ using $\operatorname{IDF}(q_i)\frac{f(q_i,D)(k_1+1)}{f(q_i,D)+k_1(1-b+b|D|/\operatorname{avgdl})}$. The saturation term prevents repeated keywords from increasing score without bound, while length normalization avoids automatically favoring long documents.`,
    baseline: "a deterministic BM25 index with judged queries",
    sources: ["Robertson and Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond (2009)", "Brin and Page, The Anatomy of a Large-Scale Hypertextual Web Search Engine (1998)", "Introduction to Information Retrieval by Manning, Raghavan, and Schütze"],
  },
  "recommendation-systems": {
    description: "Design multi-stage recommendation systems from feedback logs and candidate generation through ranking, experimentation, safety, and long-term value.",
    tags: ["recommendation", "ranking", "two-tower", "experimentation"],
    plain: "A recommendation system chooses a small, ordered set of items for one user and context from a catalog that may contain millions of possibilities. Unlike search, the user may not state a query, so the system must infer intent while managing feedback loops and long-term welfare.",
    components: ["event log", "feature platform", "candidate generators", "ranking model", "policy/diversity layer", "experiment service", "feedback loop"],
    constraints: "candidate recall, ranking quality, freshness, diversity, fairness, exploration, and long-term satisfaction",
    formula: String.raw`A two-tower retriever often uses $s(u,i)=e_u^\top e_i$, while the ranker estimates $\hat y=f(e_u,e_i,x_{ui})$. For implicit feedback, the logged label is exposure-dependent: $P(\text{click})=P(\text{examined})P(\text{click}\mid\text{examined})$, which is why position bias must be measured.`,
    baseline: "popular and recently popular items evaluated alongside a simple matrix-factorization model",
    sources: ["Covington, Adams, and Sargin, Deep Neural Networks for YouTube Recommendations (2016)", "Koren, Bell, and Volinsky, Matrix Factorization Techniques for Recommender Systems (2009)", "Recommender Systems Handbook"],
  },
  "rag-systems": {
    description: "Engineer retrieval-augmented generation from document ingestion and retrieval through grounding, evaluation, access control, and production reliability.",
    tags: ["rag", "retrieval", "grounding", "llm"],
    plain: "A RAG system finds external evidence at request time and gives that evidence to a generator. Its value is not merely adding text to a prompt: it creates an inspectable path from source material to answer, with retrieval and generation evaluated separately.",
    components: ["source connectors", "parser/chunker", "embedding/index pipeline", "retriever", "reranker", "context builder", "grounded generator", "verifier"],
    constraints: "retrieval recall, context precision, faithfulness, freshness, authorization, latency, and auditability",
    formula: String.raw`For a judged relevant set $R$ and retrieved set $K$, $\operatorname{Recall@k}=|R\cap K|/|R|$ and $\operatorname{Precision@k}=|R\cap K|/|K|$. End-to-end answer quality cannot identify which stage failed, so retrieval, context, faithfulness, and task success need separate metrics.`,
    baseline: "a small lexical retriever plus a prompt that requires citations and permits abstention",
    sources: ["Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)", "Gao et al., Retrieval-Augmented Generation for Large Language Models: A Survey", "RAGAS documentation and paper"],
  },
  "llm-serving": {
    description: "Design high-throughput LLM inference with KV-cache management, continuous batching, quantization, parallelism, scheduling, and cost controls.",
    tags: ["llm-serving", "inference", "paged-attention", "gpu"],
    plain: "An LLM serving system turns variable-length prompts into streamed tokens while sharing expensive accelerators across many users. The scheduler, memory manager, kernels, and model partitioning are as important as the model weights.",
    components: ["API gateway", "tokenizer", "request scheduler", "prefill workers", "decode workers", "KV-cache manager", "streaming transport", "telemetry"],
    constraints: "time to first token, inter-token latency, tokens per second, KV memory, fairness, availability, and cost per token",
    formula: String.raw`For batch $B$, sequence length $L$, layers $n_l$, KV heads $n_h$, head dimension $d_h$, and bytes $b$, KV memory is approximately $M_{KV}=2BLn_ln_hd_hb$; the factor two stores keys and values. Throughput improves only while batching gains exceed memory pressure and queueing delay.`,
    baseline: "single-request eager inference with measured TTFT, inter-token latency, throughput, and peak memory",
    sources: ["Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)", "vLLM documentation", "Dao et al., FlashAttention papers"],
  },
  "agent-systems": {
    description: "Build bounded, observable agent systems with tool use, planning, memory, evaluation, sandboxing, and multi-agent coordination.",
    tags: ["agents", "tool-use", "planning", "safety"],
    plain: "An agent system gives a model a goal, state, and permitted actions, then closes the loop around observations and tool results. The core engineering problem is not autonomy by itself; it is useful autonomy inside enforceable limits.",
    components: ["task API", "planner/policy", "tool registry", "permission layer", "sandbox", "memory", "verifier", "trace store"],
    constraints: "task success, step and cost budgets, side effects, prompt injection, recoverability, and traceability",
    formula: String.raw`If step $i$ succeeds with conditional probability $p_i$, a fragile $n$-step plan has success probability roughly $P(\text{success})=\prod_{i=1}^{n}p_i$. Verification, retries, and replanning can help, but also add cost and new failure paths; measure end-to-end task success, not tool-call accuracy alone.`,
    baseline: "a deterministic workflow with the same tools and explicit human decisions",
    sources: ["Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models (2023)", "AgentBench and SWE-bench papers", "OWASP guidance for LLM and agentic application security"],
  },
  "youtube-architecture": {
    description: "Understand YouTube-scale upload, transcoding, storage, CDN delivery, recommendation, live streaming, moderation, and creator economics.",
    tags: ["video", "cdn", "streaming", "recommendation"],
    plain: "YouTube is several systems joined by durable identifiers and event streams: a media-processing platform, a global delivery network, a search and recommendation product, a live service, and a safety and monetization platform.",
    components: ["upload API", "object storage", "transcode farm", "metadata/index", "recommendation", "CDN", "player", "analytics/rights"],
    constraints: "upload durability, encoding cost, startup time, rebuffering, global fan-out, rights enforcement, and accounting correctness",
    formula: String.raw`Raw storage is approximately $S=rT$, where bitrate $r$ includes audio and video and duration is $T$. With renditions $j$, replicas $c_j$, and view traffic $v_j$, total cost combines $\sum_j r_jTc_j$ storage with CDN egress; adaptive bitrate shifts quality to the best rendition the current connection can sustain.`,
    baseline: "single-region upload, one encoded rendition, object storage, and direct HTTP delivery",
    sources: ["YouTube engineering publications on recommendations and video infrastructure", "MPEG-DASH and HLS specifications", "Google SRE books"],
  },
  "netflix-architecture": {
    description: "Study Netflix-scale media encoding, Open Connect delivery, personalization, experimentation, resilience, playback, DRM, and billing.",
    tags: ["streaming", "cdn", "resilience", "experimentation"],
    plain: "Netflix separates a cloud control plane for product logic from a purpose-built delivery plane that places encoded media near viewers. Personalization, playback, billing, and resilience share observability but have different consistency requirements.",
    components: ["client", "API/control plane", "personalization", "encoding", "catalog", "Open Connect CDN", "DRM/license", "billing"],
    constraints: "playback availability, startup latency, encoding efficiency, device diversity, experiment integrity, DRM, and payment correctness",
    formula: String.raw`Availability is measured over user-visible playback attempts, not server uptime alone. If independent serial dependencies have availability $A_i$, path availability is approximately $\prod_i A_i$; fallbacks, cached metadata, and circuit breakers reduce the number of dependencies on the critical playback path.`,
    baseline: "origin-hosted fixed-bitrate streaming with a non-personalized catalog",
    sources: ["Netflix TechBlog and Open Connect documentation", "Netflix papers on per-title encoding and experimentation", "Release It! by Michael Nygard for circuit breakers"],
  },
  "tiktok-architecture": {
    description: "Design a short-video platform around real-time recommendation, media delivery, live streaming, moderation, privacy, and feedback-loop control.",
    tags: ["short-video", "recommendation", "streaming", "moderation"],
    plain: "TikTok's product is a low-latency feedback loop: serve one video, observe dense behavior, update features, and choose the next item. Media infrastructure, recommendation, experimentation, safety, and regional data boundaries all sit on that loop.",
    components: ["video ingest", "content understanding", "candidate service", "real-time features", "ranker", "policy/diversity", "CDN/player", "moderation"],
    constraints: "swipe latency, feature freshness, diversity, creator cold start, moderation recall and precision, privacy, and regional isolation",
    formula: String.raw`A ranking score may combine calibrated predictions: $s=w_1P(\text{watch})+w_2E[\text{watch time}]+w_3P(\text{share})-w_4P(\text{negative feedback})$. The weights encode product policy; optimizing only watch time can damage diversity, safety, or satisfaction.`,
    baseline: "a popularity-and-recency feed with explicit diversity constraints",
    sources: ["ByteDance engineering and recommendation publications", "Research on multi-objective recommendation and counterfactual evaluation", "HLS/WebRTC specifications for video and live delivery"],
  },
  "uber-architecture": {
    description: "Design location-aware dispatch, ETA, pricing, payments, forecasting, fraud detection, and resilient city-scale marketplace infrastructure.",
    tags: ["geospatial", "dispatch", "marketplace", "payments"],
    plain: "Uber coordinates a real-time two-sided marketplace under uncertain location, travel time, demand, and supply. The system must make fast assignments while preserving payment correctness and recovering safely from regional failures.",
    components: ["rider/driver apps", "location stream", "geospatial index", "ETA service", "dispatch optimizer", "pricing", "trip state", "payments"],
    constraints: "dispatch latency, pickup ETA, GPS error, market balance, fraud, regional availability, and financial consistency",
    formula: String.raw`Dispatch can be framed as minimum-cost matching: $\min_x\sum_{r,d}c_{rd}x_{rd}$ subject to each rider and driver being assigned at most once. Cost $c_{rd}$ may combine pickup ETA, cancellation risk, fairness, and marketplace effects; a fast approximate solution is often better than a late exact one.`,
    baseline: "nearest available driver by straight-line distance with a simple state machine",
    sources: ["Uber Engineering publications on H3, Michelangelo, dispatch, and forecasting", "H3 geospatial indexing documentation", "Literature on online bipartite matching and marketplace design"],
  },
  "chatgpt-architecture": {
    description: "Design a conversational AI product spanning routing, inference, context, memory, tools, web retrieval, safety, evaluation, and sandboxing.",
    tags: ["chatgpt", "llm", "routing", "safety"],
    plain: "ChatGPT is a compound system around language models: it manages conversations, selects models and tools, retrieves information, streams generation, applies safety controls, and learns from evaluation and user feedback.",
    components: ["conversation API", "policy/router", "context builder", "model fleet", "tool sandbox", "search/retrieval", "safety stack", "evaluation/telemetry"],
    constraints: "quality, TTFT, context limits, privacy, abuse resistance, tool safety, model availability, and inference cost",
    formula: String.raw`Context must satisfy $T_{system}+T_{history}+T_{retrieval}+T_{tools}+T_{output}\le T_{max}$. The context builder is therefore a budget allocator: retaining one item means evicting, compressing, or declining another, and every choice affects quality and privacy.`,
    baseline: "one stateless model endpoint with a fixed system prompt and no tools",
    sources: ["OpenAI technical reports and system cards", "InstructGPT and GPT-4 technical report", "Primary literature on RAG, tool use, and LLM evaluation"],
  },
  "deepseek-architecture": {
    description: "Study DeepSeek's MoE, MLA, FP8 training, reasoning pipeline, distributed serving, and efficiency as an integrated model-system design.",
    tags: ["deepseek", "moe", "mla", "reasoning"],
    plain: "DeepSeek combines model architecture and systems co-design: sparse experts reduce active compute, latent attention compresses KV state, low-precision training improves throughput, and reasoning training changes how inference compute is used.",
    components: ["data/training pipeline", "MoE router", "expert parallel workers", "MLA attention", "reasoning training", "checkpoint store", "serving router", "telemetry"],
    constraints: "expert balance, communication, numerical stability, KV memory, reasoning quality, throughput, and reproducibility of cost claims",
    formula: String.raw`For $E$ total experts and top-$k$ routing, active expert compute scales with $k$ rather than $E$, but communication and imbalance remain. A simplified load term is $L_e=\sum_t\mathbf{1}[e\in\operatorname{TopK}(x_t)]$; capacity planning must bound $\max_e L_e$, not just the average.`,
    baseline: "a dense transformer with standard multi-head attention trained and served at the same active-compute budget",
    sources: ["DeepSeek-V2 and DeepSeek-V3 technical reports", "DeepSeek-R1 paper", "DeepSeek architecture page at /architectures/deepseek-v2"],
  },
};

const levelPurpose = {
  Beginner: "build the vocabulary and trace one request end to end",
  Intermediate: "make design choices and quantify their tradeoffs",
  Advanced: "operate the system under scale, failure, security, and cost pressure",
  Research: "identify open assumptions, evaluate frontier ideas, and design falsifiable experiments",
};

const relatedLinks = {
  "vector-databases": [
    ["Search Engines", "/system-design/search-engines"],
    ["RAG Systems", "/system-design/rag-systems"],
    ["Retrieval at Scale curriculum", "/learn/ai-system-design/system-design-for-retrieval-at-scale"],
    ["HNSW architecture context", "/architectures/graphsage"],
  ],
  "search-engines": [
    ["Vector Databases", "/system-design/vector-databases"],
    ["Recommendation Systems", "/system-design/recommendation-systems"],
    ["RAG Systems", "/system-design/rag-systems"],
    ["Production RAG curriculum", "/learn/rag-systems/production-rag-architecture-at-scale"],
  ],
  "recommendation-systems": [
    ["YouTube Architecture", "/system-design/youtube-architecture"],
    ["Netflix Architecture", "/system-design/netflix-architecture"],
    ["TikTok Architecture", "/system-design/tiktok-architecture"],
    ["Matrix Factorization", "/architectures/matrix-factorization"],
  ],
  "rag-systems": [
    ["Vector Databases", "/system-design/vector-databases"],
    ["Search Engines", "/system-design/search-engines"],
    ["Agent Systems", "/system-design/agent-systems"],
    ["Factuality and grounding curriculum", "/learn/natural-language-processing/factuality-grounding-and-hallucination"],
  ],
  "llm-serving": [
    ["ChatGPT Architecture", "/system-design/chatgpt-architecture"],
    ["DeepSeek Architecture", "/system-design/deepseek-architecture"],
    ["Distributed LLM engineering", "/learn/llm-engineering/tensor-parallelism-pipeline-parallelism-and-fsdp"],
    ["Flash Attention curriculum", "/learn/llm-engineering/flash-attention-and-memory-efficient-attention"],
  ],
  "agent-systems": [
    ["RAG Systems", "/system-design/rag-systems"],
    ["LLM Serving", "/system-design/llm-serving"],
    ["ChatGPT Architecture", "/system-design/chatgpt-architecture"],
    ["Compound AI Systems curriculum", "/learn/ai-system-design/compound-ai-systems"],
  ],
  "youtube-architecture": [
    ["Recommendation Systems", "/system-design/recommendation-systems"],
    ["Search Engines", "/system-design/search-engines"],
    ["TikTok Architecture", "/system-design/tiktok-architecture"],
    ["Two-Tower Model", "/architectures/two-tower-model"],
  ],
  "netflix-architecture": [
    ["Recommendation Systems", "/system-design/recommendation-systems"],
    ["Search Engines", "/system-design/search-engines"],
    ["ML Platform Engineering curriculum", "/learn/mlops/ml-platform-engineering"],
    ["Two-Tower Model", "/architectures/two-tower-model"],
  ],
  "tiktok-architecture": [
    ["Recommendation Systems", "/system-design/recommendation-systems"],
    ["YouTube Architecture", "/system-design/youtube-architecture"],
    ["Video Generation curriculum", "/learn/computer-vision/video-generation"],
    ["Alignment curriculum", "/learn/natural-language-processing/alignment"],
  ],
  "uber-architecture": [
    ["ML Platform Engineering curriculum", "/learn/mlops/ml-platform-engineering"],
    ["Recommendation Systems", "/system-design/recommendation-systems"],
    ["Real-Time Inference curriculum", "/learn/mlops/real-time-inference-at-scale"],
    ["GraphSAGE", "/architectures/graphsage"],
  ],
  "chatgpt-architecture": [
    ["LLM Serving", "/system-design/llm-serving"],
    ["RAG Systems", "/system-design/rag-systems"],
    ["Agent Systems", "/system-design/agent-systems"],
    ["GPT-4", "/architectures/gpt-4"],
  ],
  "deepseek-architecture": [
    ["LLM Serving", "/system-design/llm-serving"],
    ["ChatGPT Architecture", "/system-design/chatgpt-architecture"],
    ["DeepSeek-V2", "/architectures/deepseek-v2"],
    ["Mixture of Experts curriculum", "/learn/deep-learning/mixture-of-experts-moe"],
  ],
};

function slugLabel(text) {
  return text.replace(/[^a-zA-Z0-9]+/g, " ").trim();
}

function mdxText(text) {
  return text.replaceAll("<", "&lt;");
}

function prerequisites(module) {
  const items = module.prerequisites.length
    ? module.prerequisites.map((item) => `**${item}**`).join(", ")
    : "No formal prerequisites";
  return `:::note\n**Prerequisites:** ${items}. Before continuing, explain each item in your own words and identify where it appears in the request path below.\n:::`;
}

function teachObjective(objective, system, module, profile, index) {
  const stages = profile.components;
  const stage = stages[index % stages.length];
  const renderedObjective = mdxText(objective);
  return `#### Objective ${index + 1}: ${renderedObjective}

${directAnswer(objective, profile)} The capability described by this objective is practical, not a definition to memorize. Begin at the **${stage}** boundary and trace the data entering it, the state it reads, the output contract, and the owner of failures. In ${system.name}, that boundary interacts with ${profile.constraints}. A design is incomplete if it names a component but cannot say how it is measured, versioned, isolated, and recovered.

At the ${module.level.toLowerCase()} level, the useful question is: *what decision can you make after learning this?* Write assumptions before selecting technology. Use a representative workload, identify the cheapest credible baseline (${profile.baseline}), and measure quality and operational cost together. Separate control-plane work—configuration, policy, placement, and metadata—from data-plane work on the live request. This prevents slow management operations from entering the latency-critical path. Finally, test one normal case, one boundary case, and one dependency failure. That small discipline turns the objective into an engineering skill and exposes where an apparently elegant diagram hides queues, stale state, or unsafe fallback behavior.`;
}

function diagramFor(requirement, system, profile, index) {
  const components = profile.components;
  const a = components[index % components.length];
  const b = components[(index + 1) % components.length];
  const c = components[(index + 2) % components.length];
  const d = components[(index + 3) % components.length];
  const renderedRequirement = mdxText(requirement);
  const flowSource = requirement.includes(":") ? requirement.slice(requirement.indexOf(":") + 1) : requirement;
  const flowParts = flowSource
    .split(/\s*(?:→|->)\s*/)
    .map((part) => part.trim().replace(/\s+/g, " "))
    .filter(Boolean);
  const flowDiagram = flowParts.length > 1
    ? flowParts.map((part, partIndex) => `${"  ".repeat(partIndex)}[${part.slice(0, 72)}]${partIndex < flowParts.length - 1 ? "\n" + "  ".repeat(partIndex) + "        |\n" + "  ".repeat(partIndex) + "        v" : ""}`).join("\n")
    : `  [producer / request]
           |
           v
      [${a}] ---- control, version, policy ----> [${b}]
           |                                         |
           | data / candidate / event                | state / cache
           v                                         v
      [${c}] ---- measured result + trace -----> [${d}]
           |                                         |
           +-------- retry / fallback / audit <------+`;
  return `#### Diagram ${index + 1}: ${renderedRequirement}

\`\`\`text
${system.name}: ${slugLabel(requirement)}

${flowDiagram}
\`\`\`

Read the diagram from left to right for the normal path and from the bottom back toward the top for recovery. The label **${renderedRequirement}** determines what each arrow carries; record its schema, freshness requirement, and failure behavior. In an interview, redraw this small path first, then add partitions, replicas, queues, and security boundaries only when the requirements demand them. That order keeps the explanation understandable while still exposing ${profile.constraints}.`;
}

function caseStudy(study, system, module, profile, index) {
  const focusComponent = profile.components[index % profile.components.length];
  const renderedStudy = mdxText(study);
  return `#### ${renderedStudy}

Treat this case as a design investigation rather than a claim that every internal detail is public. The stated scenario—**${renderedStudy}**—creates pressure at the **${focusComponent}** layer of ${system.name}. Start by writing the actor, workload, user-visible success condition, and failure cost. Preserve any numbers given in the prompt as requirements; where numbers are absent, label estimates explicitly rather than presenting them as company facts.

The first design should use ${profile.baseline}. Instrument it for ${profile.constraints}, then locate the first bottleneck with evidence. Introduce one mechanism at a time: partitioning for capacity, caching for repeated work, asynchronous processing for non-critical tasks, and a fallback for dependency failure. Keep authorization and tenant boundaries on the data path rather than filtering after retrieval or generation. For rollout, shadow real traffic, compare against the baseline, inspect important slices, and define a rollback threshold before exposure.

The lesson is the decision process. A successful case study explains why one resource became scarce, why the chosen mechanism addresses that scarcity, and what new failure mode it creates. It closes with a measurement plan: one quality metric, one tail-latency or freshness metric, one cost metric, and one safety or correctness invariant. That makes the story reusable instead of a list of brand-name components.`;
}

function directAnswer(question, profile) {
  const q = question.toLowerCase();
  const rules = [
    [/curse of dimensionality/, "As dimensions grow, distance values concentrate and tree partitions lose pruning power; many points look similarly far away. Exact search therefore touches too much of the collection, while ANN uses navigable graphs, coarse partitions, or compressed codes to inspect a promising subset and measures the resulting recall loss."],
    [/b-tree/, "A B-tree orders scalar keys and prunes by one-dimensional intervals. Vector similarity is a high-dimensional distance relation with no single total ordering that preserves neighborhoods, so a B-tree cannot efficiently discard most candidates. Use an ANN index and a separate metadata index for filters."],
    [/hnsw/, "HNSW is a layered proximity graph. Search enters a sparse upper layer, greedily approaches the query, then descends into denser layers and explores a bounded candidate queue. Parameters such as M and ef trade memory and construction cost for recall and latency."],
    [/l2 distance over cosine/, "Choose L2 when vector magnitude carries meaning or the training objective used Euclidean geometry. Choose cosine when direction matters and magnitude should be ignored. For normalized vectors, ranking by cosine and squared L2 is equivalent, so operational convenience can decide."],
    [/bm25/, "BM25 combines inverse document frequency, term-frequency saturation, and document-length normalization. It improves on raw TF-IDF because repeated terms have diminishing returns and long documents do not win merely by containing more words."],
    [/pagerank/, "PageRank treats links as weighted votes and estimates the stationary probability of a random surfer. It added an authority signal independent of exact query wording, helping distinguish pages that mention the same terms but differ greatly in importance."],
    [/inverted index/, "An inverted index maps each term directly to its postings list, making dictionary lookup approximately constant-time. A multi-term query still has to fetch and combine multiple postings lists; the real cost depends on their lengths and intersection strategy, not merely the number of query words."],
    [/ndcg/, "NDCG discounts relevant results that appear low in the ranking and normalizes by the best possible ordering for that query. It supports graded relevance, making it more informative than binary accuracy when position strongly affects user value."],
    [/cold-start/, "Cold start means interactions are insufficient for a new user or item. Use content features, onboarding preferences, contextual/popularity priors, and controlled exploration; then transition toward personalized signals as evidence accumulates."],
    [/matrix factorization/, "Matrix factorization represents users and items with latent vectors and scores a pair with a dot product plus optional biases. Training moves observed positive pairs closer and sampled or explicit negatives apart, yielding efficient retrieval but limited context modeling."],
    [/explicit and implicit feedback/, "Explicit feedback is a direct rating or preference; implicit feedback is behavior such as view, click, dwell, or purchase. Implicit data is abundant but confounds preference with exposure and interface position, so missing interaction is not a clean negative label."],
    [/position bias/, "Items near the top are more likely to be seen regardless of relevance, so click labels contain both examination and preference. Correct with randomized interventions, inverse-propensity weighting, or models that explicitly estimate examination probability."],
    [/what is rag|rag and why/, "RAG retrieves external evidence and conditions generation on it. It can reduce unsupported claims by supplying current, attributable context, but only if retrieval finds the right evidence and the generator follows it; RAG reduces rather than eliminates hallucination."],
    [/chunk overlap/, "Overlap preserves facts or references that cross an arbitrary chunk boundary. Too little loses context; too much duplicates evidence, inflates the index, and crowds the prompt. Tune it against retrieval and answer metrics on representative documents."],
    [/retrieval precision and retrieval recall/, "Retrieval precision asks what fraction of retrieved items are useful; recall asks what fraction of all needed evidence was retrieved. High recall supplies coverage, while high precision protects the limited context window from distraction."],
    [/kv cache/, "The KV cache stores attention keys and values from earlier tokens so decoding computes only the new token's projections. It removes repeated attention-prefix work, but memory grows with concurrent sequences, layers, token count, KV heads, head dimension, and numeric precision."],
    [/ttft/, "Time to first token includes queueing, tokenization, routing, and prefill. Tokens per second or inter-token latency describes decode after streaming begins. Optimizing one can harm the other, so report both by prompt and output length."],
    [/quantization/, "Quantization stores weights or activations with fewer bits, reducing memory traffic and allowing larger models or batches. The benefit depends on hardware kernels and calibration; always measure quality by task and slice rather than trusting bit width alone."],
    [/continuous batching/, "Naive batching waits for every sequence in a batch to finish, leaving slots idle when lengths differ. Continuous batching admits new work as sequences complete, improving utilization while requiring a scheduler that protects queueing latency and fairness."],
    [/pagedattention/, "PagedAttention stores KV blocks in non-contiguous physical pages behind a logical mapping, similar to virtual memory. It reduces fragmentation and enables sharing, growth, and reclamation without reserving each request's maximum context up front."],
    [/speculative decoding/, "A draft mechanism proposes several tokens and the target model verifies them in parallel with an acceptance rule that preserves the target distribution. Speedup depends on acceptance rate, verification cost, draft cost, and workload; a poor draft can add overhead."],
    [/pipeline and an llm agent|pipeline and.*agent/, "A pipeline follows a developer-defined control flow. An agent chooses among permitted actions from observations and may replan. The distinction is dynamic control, not whether an LLM or tool appears in the system."],
    [/react framework/, "ReAct interleaves reasoning or planning with actions and observations. The useful engineering property is the explicit feedback loop and trace, which allows correction after tool results; hidden reasoning text itself should not be treated as a guarantee of correctness."],
    [/infinite loop/, "Enforce maximum steps, wall-clock and token budgets, repeated-state detection, idempotency keys, and a no-progress evaluator. On exhaustion, return a structured partial result or request human help rather than silently continuing."],
    [/side effect|sends an email|deletes a file/, "Classify tools by effect, separate planning from execution, preview the exact action, require scoped authorization or human confirmation, and use idempotency plus audit logs. A model's textual intention is never the authorization boundary."],
    [/adaptive bitrate/, "Adaptive bitrate streaming encodes several quality renditions and lets the player switch segments according to measured bandwidth and buffer health. It reduces rebuffering under changing networks while using higher quality when conditions permit."],
    [/circuit breaker/, "A circuit breaker stops calls to a dependency after failures cross a threshold, then probes recovery after a timeout. It prevents thread and queue exhaustion from cascading, but requires a meaningful fallback and carefully tuned trip conditions."],
    [/chaos engineering/, "Chaos engineering introduces controlled faults to test a stated resilience hypothesis. It is valuable when blast radius, abort conditions, observability, and expected user impact are defined; random breakage without a hypothesis is not chaos engineering."],
    [/surge pricing/, "Surge pricing estimates local supply-demand imbalance and adjusts price or incentives to improve marketplace balance. Production design needs smoothing, geographic boundaries, caps, transparency, anti-gaming controls, and evaluation of rider and driver outcomes."],
    [/nearby drivers/, "Maintain moving drivers in a geospatial index such as hierarchical hexagonal cells, query nearby cells, then compute network-aware ETA for a shortlist. Scanning all drivers wastes work and cannot meet a short dispatch deadline."],
    [/gpt-4 \(base\).*chatgpt|base\) and chatgpt/, "A base model predicts continuations from pretraining. ChatGPT is a product system around adapted models: instruction/alignment training, conversation state, routing, tools, safety controls, streaming, evaluation, and user interfaces."],
    [/context limit/, "Attention and KV memory grow with the retained token sequence, training only covers bounded lengths, and serving needs predictable resource limits. A context builder must budget system instructions, history, retrieved evidence, tool outputs, and room for generation."],
    [/mixture of experts/, "MoE keeps many expert parameter sets but routes each token to only a small top-k subset. This increases total capacity without activating every parameter, though expert communication, load balance, and memory placement remain system costs."],
    [/multi-head latent attention|\bmla\b/, "MLA compresses key/value information into a lower-dimensional latent representation and reconstructs what attention needs, reducing KV-cache and bandwidth pressure. The comparison must include projection compute, quality, kernel support, and serving layout."],
    [/\bgrpo\b/, "GRPO estimates relative advantage within a group of sampled responses, avoiding a separate learned value model used by PPO-style training. Its behavior depends on reward quality, group diversity, clipping or regularization, and stable reference-policy control."],
  ];
  const match = rules.find(([pattern]) => pattern.test(q));
  if (match) return match[1];
  return `A strong answer begins by fixing the workload and success criterion, then traces the request through ${profile.components.slice(0, 4).join(", ")}. The governing quantitative model is: ${profile.formula} Use it to estimate an order of magnitude before selecting infrastructure. Then quantify ${profile.constraints}, state which data is authoritative, and name the failure and fallback paths. The simplest credible baseline is ${profile.baseline}; compare against it before adding complexity.`;
}

function interviewAnswer(question, system, profile, index) {
  const renderedQuestion = mdxText(question);
  return `#### Q${index + 1}: ${renderedQuestion}

${directAnswer(question, profile)} For **${system.name}**, make the answer operational: draw the critical path, put a budget on each stage, and say what happens when a dependency is slow, stale, unauthorized, or unavailable. Distinguish measured facts from assumptions and avoid invented company internals.

Then close with validation. Use an offline quality set for repeatability, a load or failure test for system behavior, and a guarded online rollout for user impact. Track an end-to-end outcome plus the stage-level metrics needed to localize regressions. This structure answers both halves of a system-design interview: why the mechanism works and how you would operate it responsibly.`;
}

function levelSection(system, module, profile) {
  const objectives = module.learningObjectives
    .map((objective, index) => teachObjective(objective, system, module, profile, index))
    .join("\n\n");
  const diagrams = module.diagramsNeeded
    .map((diagram, index) => diagramFor(diagram, system, profile, index))
    .join("\n\n");
  const cases = module.caseStudies
    .map((study, index) => caseStudy(study, system, module, profile, index))
    .join("\n\n");
  const questions = module.interviewQuestions.length
    ? module.interviewQuestions
        .map((question, index) => interviewAnswer(question, system, profile, index))
        .join("\n\n")
    : "This frontier module has no fixed interview-question list. Turn each learning objective into a research-defense question: state the hypothesis, strongest alternative explanation, decisive experiment, and evidence that would make you abandon the idea.";

  const title = {
    Beginner: "Beginner: Foundations",
    Intermediate: "Intermediate: Design Deep Dive",
    Advanced: "Advanced: Production at Scale",
    Research: "Research: Frontiers",
  }[module.level];

  return `## ${title}

${prerequisites(module)}

At this level, the goal is to **${levelPurpose[module.level]}**. ${profile.plain} Keep one principle visible: a system is not the boxes in its diagram. It is the contracts, queues, ownership boundaries, measurements, and recovery behavior between those boxes.

${objectives}

### Diagrams

${diagrams}

### Case Studies

${cases}

### Interview Questions & Answers

${questions}`;
}

function proposedProjects(system, module, profile) {
  return [
    `**${module.level} project (proposed): Traceable ${system.name} slice.** Implement one request path using ${profile.components.slice(0, 3).join(", ")}. Log correlation IDs and stage latency, inject one dependency failure, and demonstrate a safe fallback. Deliver a diagram, runnable code, load-test result, and one-page decision record.`,
    `**${module.level} project (proposed): Baseline-to-design evaluation.** Build ${profile.baseline}, add one mechanism from this module, and compare quality, p95/p99 latency, resource cost, and a safety or correctness invariant. Include assumptions, raw results, failure examples, and a rollback threshold.`,
  ];
}

function projectsSection(system, profile) {
  const entries = [];
  for (const module of system.modules) {
    const projects = module.handsOnProjects.length
      ? module.handsOnProjects.map((project) => `**${module.level} project:** ${project}`)
      : proposedProjects(system, module, profile);
    entries.push(`### ${module.level}\n\n${projects.map((project) => `- ${project}`).join("\n")}`);
  }
  return `## Hands-On Projects\n\n${entries.join("\n\n")}`;
}

function render(system) {
  const profile = profiles[system.slug];
  if (!profile) throw new Error(`Missing profile for ${system.slug}`);
  const levels = system.modules.map((module) => levelSection(system, module, profile)).join("\n\n");
  const links = relatedLinks[system.slug];
  if (!links || links.length < 4) throw new Error(`Missing related links for ${system.slug}`);
  return `# ${system.name}

${profile.plain}

This article progresses from first principles to research questions. At every level, trace a request, state assumptions, quantify tradeoffs, and preserve a simple baseline. The recurring design pressures are ${profile.constraints}.

${profile.formula}

${levels}

${projectsSection(system, profile)}

## Related Platform Content

${links.map(([label, href]) => `- [${label}](${href})`).join("\n")}

## Further Reading

Prefer primary papers, standards, official engineering publications, and maintained documentation. Start with:

${profile.sources.map((source) => `- ${source}`).join("\n")}

Use sources as evidence, not authority. Record the workload, date, hardware or data setting, and limitations behind every performance claim before transferring it to a different system.
`;
}

let written = 0;
for (const system of systems) {
  const profile = profiles[system.slug];
  const directory = path.join(ROOT, "src/content/system-design", system.slug);
  fs.mkdirSync(directory, { recursive: true });
  fs.writeFileSync(path.join(directory, "content.mdx"), render(system), "utf8");
  fs.writeFileSync(
    path.join(directory, "meta.json"),
    `${JSON.stringify({
      type: "system-design",
      slug: system.slug,
      title: system.name,
      description: profile.description,
      tags: ["system-design", ...profile.tags],
      difficulty: "advanced",
    }, null, 2)}\n`,
    "utf8",
  );
  written += 1;
}

console.log(`Generated ${written} canonical system-design articles and metadata files.`);
