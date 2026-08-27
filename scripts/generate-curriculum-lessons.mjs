#!/usr/bin/env node

/**
 * Generate the remaining curriculum lesson bodies from the canonical metadata.
 *
 * The long-form drafts live in src/content/curriculum. This script is kept so
 * future curriculum additions start with the same pedagogical structure rather
 * than an empty page. It never overwrites an existing lesson.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const curriculumSource = fs.readFileSync(
  path.join(ROOT, "src/data/content/curriculum.ts"),
  "utf8",
);
const curriculumLiteral = curriculumSource.match(
  /export const CURRICULUM[^=]*=\s*(\[[\s\S]*\]);/,
)?.[1];

if (!curriculumLiteral) throw new Error("Could not parse CURRICULUM metadata");
const curriculum = Function(`"use strict"; return (${curriculumLiteral});`)();

const domainGuides = {
  "rag-systems": {
    lens: "retrieval quality, evidence quality, and operational reliability",
    mentalModel: "A RAG system is a controlled evidence pipeline: understand the request, find useful evidence, assemble context, generate an answer, and verify that the answer is supported.",
    diagram: "User query -> query analysis -> retrieval -> reranking -> context builder\n           -> grounded generator -> citation/quality checks -> answer",
    practice: "Build the smallest retrieval pipeline that exposes intermediate results. Inspect retrieved evidence before tuning generation, because a fluent model cannot repair missing or irrelevant context reliably.",
  },
  "ai-agents": {
    lens: "state, actions, feedback, memory, and bounded autonomy",
    mentalModel: "An agent is a feedback-controlled program. It observes state, chooses a permitted action, executes through a tool or environment, evaluates the result, and updates memory before the next step.",
    diagram: "Goal -> planner/policy -> permitted action -> environment\n  ^          |                 |             |\n  +-- memory +-- verifier <----+-- observation+",
    practice: "Represent plans, observations, and tool results as explicit data. This makes failures inspectable and allows limits, retries, and human approval to be applied at the correct boundary.",
  },
  "reinforcement-learning": {
    lens: "sequential decisions, delayed reward, uncertainty, and safety",
    mentalModel: "Reinforcement learning asks how an agent should act now when an action changes both its immediate reward and the states it may encounter later.",
    diagram: "state s_t -> policy -> action a_t -> environment -> reward r_t, state s_(t+1)\n    ^                                                        |\n    +---------------------- learning update ------------------+",
    practice: "Start with a tiny environment where states, actions, rewards, and termination are visible. A small transparent experiment reveals credit-assignment and exploration problems more clearly than a large benchmark.",
  },
  "computer-vision": {
    lens: "visual representation, spatial-temporal structure, conditioning, and evaluation",
    mentalModel: "Modern vision systems turn pixels or video patches into reusable representations, then align, generate, segment, or act from those representations under task-specific conditioning.",
    diagram: "pixels/frames -> visual encoder -> latent tokens -> task or language fusion\n                                              -> prediction/generation/action",
    practice: "Trace tensor shape, coordinate system, and supervision source at every stage. Many apparent model failures are actually resizing, normalization, timestamp, or label-alignment errors.",
  },
  mlops: {
    lens: "repeatability, reliability, observability, governance, and unit economics",
    mentalModel: "MLOps turns an experimental model into a managed product by controlling data, code, artifacts, deployment, monitoring, ownership, and rollback as one lifecycle.",
    diagram: "data -> train -> evaluate -> registry -> deploy -> observe\n    ^       artifacts + lineage       |         |\n    +----------- feedback ------------+--rollback",
    practice: "Define an artifact contract and a measurable service objective before choosing tools. Platforms succeed when they make the safe path easy, not when they merely collect fashionable infrastructure.",
  },
  "ai-system-design": {
    lens: "capacity, latency, throughput, failure isolation, quality, and cost",
    mentalModel: "AI system design is the discipline of turning model behavior into a dependable service by budgeting every scarce resource and making every failure mode observable and recoverable.",
    diagram: "client -> gateway -> scheduler/router -> model or retrieval workers\n                     |             |              |\n                  policy       queues/cache    telemetry\n                     +------------ verifier ------+",
    practice: "Write workload assumptions first: request rate, input/output size, latency target, availability, hardware, and quality floor. Architecture choices only make sense against those numbers.",
  },
  "research-engineering": {
    lens: "reproducibility, evidence, iteration speed, and honest communication",
    mentalModel: "Research engineering converts a question into trustworthy evidence through a traceable chain of assumptions, code, data, experiments, analysis, and communication.",
    diagram: "question -> hypothesis -> implementation -> experiment -> analysis -> claim\n                 ^              artifacts + logs              |\n                 +-------------- revision ---------------------+",
    practice: "Keep the question, configuration, code revision, data version, environment, metrics, and conclusion together. If one link is missing, a result becomes difficult to reproduce or interpret.",
  },
};

const focus = {
  raft: "Retrieval-Augmented Fine-Tuning teaches a model to answer from useful documents while resisting realistic distractors; the key training example contains a question, oracle evidence, distractor evidence, and a grounded answer.",
  "multi-modal-rag": "Multi-modal RAG retrieves across text, tables, charts, page images, and layout. The central challenge is preserving relationships that disappear when a document is flattened into plain text.",
  "production-rag-architecture-at-scale": "Production RAG separates ingestion from serving, versions indexes and embeddings, controls tail latency, evaluates retrieval and generation independently, and degrades safely when dependencies fail.",
  "continual-indexing-and-knowledge-freshness": "Continual indexing propagates source changes through parsing, chunking, embedding, and index publication without serving a mixture of incompatible versions or leaving deleted facts searchable.",
  "rag-for-code": "Code retrieval must understand symbols, call graphs, repository boundaries, generated files, and version history. Lexical names and structural relationships are often as important as embedding similarity.",
  "emergent-cooperation-in-multi-agent-systems": "Cooperation emerges when individual learning incentives make coordination useful. Researchers must separate genuine protocol formation from artifacts caused by centralized training, shared rewards, or privileged information.",
  "continual-learning-agents": "A continual-learning agent turns experience into reusable memory or policy improvement while protecting older skills. The hard problem is deciding what to store, what to consolidate, and what to forget.",
  "simulation-environments-for-agent-training": "Simulation environments provide scalable tasks, feedback, and resettable state for training agents. A useful simulator must be valid enough that success transfers beyond its shortcuts.",
  "world-models-for-agents": "A world model predicts how candidate actions change future state. Agents can imagine several futures, compare them, and execute only the most promising safe action.",
  "formal-specification-and-verification-of-agent-behavior": "Formal specification translates vague expectations into properties over traces, states, and actions. Verification then establishes—or tests with explicit limits—whether an agent can violate those properties.",
  "world-models": "World models learn compact latent dynamics from observations, enabling an RL policy to train on imagined trajectories. The model must preserve reward-relevant details without wasting capacity on irrelevant pixels.",
  "hierarchical-rl": "Hierarchical RL introduces temporally extended skills or options. A high-level controller selects a subgoal while a lower-level policy performs the detailed actions needed to reach it.",
  "safe-rl-and-constrained-mdp": "A constrained MDP optimizes reward while bounding one or more expected safety costs. The constraint is part of the objective, not a warning added after training.",
  "alphazero-and-mcts-with-deep-rl": "AlphaZero alternates planning and learning: MCTS improves the policy for the current position, and neural networks learn from self-play targets produced by that search.",
  "foundation-models-as-world-models": "Foundation models can encode broad regularities about language, vision, and action, but using them as world models requires calibrated predictions, action-conditioned dynamics, and tests for causal rather than verbal plausibility.",
  "vision-language-models": "Vision-language models map visual evidence and language into a shared reasoning process. The system must preserve spatial detail while exposing a token interface that a language model can use.",
  "foundation-models-for-vision": "Vision foundation models learn transferable visual representations at scale, then adapt through prompts or lightweight heads to segmentation, detection, retrieval, and recognition.",
  "video-generation": "Video generation models must model appearance and motion together. Temporal consistency, controllability, physical plausibility, and efficient latent representations distinguish video from independent image generation.",
  "generative-video-editing-and-controlnet": "Controlled generation injects structure such as pose, depth, edges, identity, or motion while preserving the capabilities of a pretrained diffusion model. Video adds the requirement that control remain consistent over time.",
  "embodied-vision": "Embodied vision connects perception to action. The relevant representation must answer not only what is visible, but where it is, how it may change, and what the agent can safely do next.",
  "ml-platform-engineering": "An ML platform provides opinionated, reusable paths for data access, training, evaluation, deployment, and monitoring while preserving lineage and allowing teams to extend rather than bypass the platform.",
  "large-scale-training-infrastructure": "Large-scale training infrastructure keeps thousands of accelerators doing useful synchronized work despite network bottlenecks, stragglers, hardware failures, checkpoint pressure, and numerical instability.",
  "real-time-inference-at-scale": "Real-time inference converts an irregular stream of requests into efficient accelerator batches while protecting time-to-first-token, inter-token latency, fairness, and memory limits.",
  "cost-engineering-and-carbon-accounting-for-ml": "Cost and carbon accounting attribute compute, storage, networking, and energy to experiments and serving traffic, so quality improvements can be compared against their complete operational footprint.",
  "ml-governance-and-compliance": "ML governance assigns ownership and evidence to data use, model behavior, approvals, monitoring, incidents, and retirement. Compliance becomes a repeatable engineering process rather than a launch-time document exercise.",
  "inference-optimization": "Inference optimization reduces latency and cost without crossing a quality floor. It combines measurement, batching, quantization, kernel efficiency, caching, and workload-aware scheduling.",
  "speculative-decoding": "Speculative decoding lets a cheap draft model propose tokens and a target model verify them in parallel. The accepted output distribution remains that of the target model when verification is implemented correctly.",
  "distributed-inference": "Distributed inference partitions model state or work across devices and hosts. Communication topology, synchronization, failure handling, and uneven request shapes determine real performance.",
  "system-design-for-retrieval-at-scale": "Retrieval at scale partitions indexes, routes queries, balances recall against latency, combines lexical and dense signals, and preserves freshness while operating under memory and tail-latency limits.",
  "llm-gateway-design": "An LLM gateway centralizes authentication, quotas, routing, provider fallback, policy checks, caching, observability, and cost attribution without becoming an opaque single point of failure.",
  "evaluation-infrastructure": "Evaluation infrastructure treats prompts, datasets, graders, models, and thresholds as versioned artifacts and turns quality changes into reproducible regression signals.",
  "hardware-aware-architecture-search": "Hardware-aware NAS searches over models using measured or accurately predicted latency, memory, energy, and quality, producing a Pareto frontier rather than a single abstractly efficient network.",
  "custom-cuda-kernels-for-ml": "A custom GPU kernel wins by matching computation to the memory hierarchy: coalesced access, tiling, fusion, occupancy, and numerically stable accumulation matter more than simply writing code in CUDA.",
  "multi-cluster-training-orchestration": "Multi-cluster training coordinates placement, topology, checkpoints, retries, and progress across failure domains where bandwidth and reliability differ sharply within and between clusters.",
  "production-safety-and-alignment-systems": "Production safety uses layered controls before, during, and after generation: policy, input defenses, model behavior, tool permissions, output classifiers, monitoring, incident response, and continuous evaluation.",
  "compound-ai-systems": "A compound AI system combines models, retrieval, tools, programs, memory, and verifiers. Its quality depends on interfaces and feedback loops as much as on any single model.",
  "reading-ml-papers-effectively": "Effective paper reading is a question-driven process: identify the claim, reconstruct the experiment that supports it, inspect assumptions, and decide what evidence would change your mind.",
  "setting-up-a-research-environment": "A research environment makes dependencies, hardware assumptions, seeds, and entry points reproducible. The goal is not merely installation; it is repeatable execution by another person or machine.",
  "git-for-research-projects": "Git connects every result to the exact code that produced it. Small commits, experiment branches, tags, and ignored artifacts turn version control into a scientific instrument.",
  "experiment-tracking-basics-wandb-mlflow": "Experiment tracking records configurations, metrics, artifacts, environment, and code identity so runs can be compared fairly and promising results can be reproduced.",
  "writing-research-code": "Good research code optimizes for correct change: explicit configuration, modular boundaries, tests around invariants, deterministic data flow, and simple experiment entry points.",
  "reproducing-sota-results": "Reproduction tests whether a reported result survives independent implementation. It requires an evidence ledger of paper claims, hidden assumptions, data versions, evaluation details, and deviations.",
  "ablation-study-design": "An ablation changes one causal component at a time while holding the experimental budget and evaluation protocol constant, revealing which parts of a method actually support the claimed improvement.",
  "hyperparameter-search-at-scale": "Scalable search allocates trials intelligently, stops weak runs early, and records the search process. The validation set remains part of optimization, so a final untouched test is essential.",
  "dataset-curation-and-preprocessing-pipelines": "Dataset pipelines define inclusion, provenance, normalization, deduplication, quality filtering, splitting, and versioning. Every transformation changes what the model can learn and what an evaluation means.",
  "writing-technical-research-reports": "A research report creates a traceable argument from question to evidence. Readers should be able to distinguish observation, interpretation, limitation, and proposed next step.",
  "open-source-library-contribution": "A strong contribution solves a maintainer-recognized problem, follows project conventions, includes tests and documentation, and keeps its review surface small enough to reason about.",
  "large-scale-pretraining-experiments": "Pretraining experiments manage data mixtures, optimizer stability, distributed efficiency, checkpoint recovery, and online diagnostics while preserving enough control to learn from expensive runs.",
  "scaling-law-modeling": "Scaling-law modeling fits simple relationships between loss, model size, data, and compute, then validates whether extrapolation remains stable across held-out scales and alternative fits.",
  "efficient-fine-tuning-research": "PEFT research asks which low-dimensional parameter updates preserve adaptation quality while reducing memory, communication, storage, and interference across tasks.",
  "evaluation-research": "Evaluation research studies whether a measurement is valid, reliable, sensitive, contamination-resistant, and representative of the capability or risk it claims to measure.",
  "implementing-novel-architectures": "Novel architecture implementation translates equations and prose into tensor contracts, initialization, numerics, kernels, tests, and controlled comparisons against a trusted baseline.",
  "first-principles-research-taste": "Research taste is disciplined problem selection: important enough to matter, specific enough to test, neglected for a reason you understand, and tractable with available evidence and resources.",
  "mechanistic-interpretability-2": "Expert mechanistic interpretability uses causal interventions, sparse feature decompositions, and circuit hypotheses to explain computations rather than merely correlate activations with labels.",
  "alignment-research-methods": "Alignment research turns uncertain safety hypotheses into falsifiable experiments spanning preference learning, oversight, adversarial testing, interpretability, and governance of deployment decisions.",
  "research-paper-writing-for-top-venues": "A strong paper states one precise contribution, positions it honestly, supports it with decisive experiments, exposes limitations, and makes the reasoning easy for skeptical reviewers to audit.",
  "building-research-infrastructure-at-scale": "Research infrastructure multiplies iteration speed through shared data, compute, environments, orchestration, observability, and evaluation while preserving flexibility for unusual experiments.",
};

const sourceAliases = {
  "continued-pre-training-and-domain-adaptation-at-scale": "continued-pre-training",
  "flash-attention-and-memory-efficient-attention": "flash-attention",
  "tensor-parallelism-pipeline-parallelism-and-fsdp": "tensor-and-pipeline-parallelism",
  "continual-learning-and-catastrophic-forgetting": "continual-learning",
};

function list(items, fallback) {
  return items.length ? items.map((item) => `**${item}**`).join(", ") : fallback;
}

function codeFor(domain, topic) {
  if (domain.slug === "rag-systems") return `\`\`\`python
from dataclasses import dataclass
from math import sqrt

@dataclass
class Passage:
    text: str
    vector: list[float]
    source: str

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    return dot / max(norm_a * norm_b, 1e-12)

def retrieve(query_vector, passages, k=2):
    scored = [(cosine(query_vector, p.vector), p) for p in passages]
    return sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]

corpus = [
    Passage("Refunds take five business days.", [0.9, 0.1], "policy-v3"),
    Passage("Passwords require twelve characters.", [0.1, 0.9], "security-v2"),
]
results = retrieve([1.0, 0.0], corpus)
context = "\\n".join(f"[{p.source}] {p.text}" for _, p in results)
assert "policy-v3" in context
print(context)
\`\`\``;
  if (domain.slug === "ai-agents") return `\`\`\`python
from dataclasses import dataclass, field

@dataclass
class AgentState:
    goal: str
    observations: list[str] = field(default_factory=list)
    steps_left: int = 4

def choose_action(state):
    if not state.observations:
        return {"tool": "inspect", "argument": state.goal}
    return {"tool": "finish", "argument": state.observations[-1]}

def run_agent(goal, tools):
    state = AgentState(goal)
    while state.steps_left > 0:
        action = choose_action(state)
        if action["tool"] == "finish":
            return action["argument"]
        if action["tool"] not in tools:
            raise ValueError("Action is outside the allowlist")
        observation = tools[action["tool"]](action["argument"])
        state.observations.append(observation)
        state.steps_left -= 1
    raise RuntimeError("Step budget exhausted")

tools = {"inspect": lambda goal: f"verified evidence for: {goal}"}
print(run_agent("check the deployment", tools))
\`\`\``;
  if (domain.slug === "reinforcement-learning") return `\`\`\`python
from collections import defaultdict
import random

Q = defaultdict(float)
alpha, gamma, epsilon = 0.2, 0.95, 0.1
actions = (-1, 1)

def choose(state):
    if random.random() < epsilon:
        return random.choice(actions)
    return max(actions, key=lambda action: Q[state, action])

def step(state, action):
    next_state = max(0, min(4, state + action))
    reward = 1.0 if next_state == 4 else -0.01
    return next_state, reward, next_state == 4

for episode in range(500):
    state = 0
    for _ in range(20):
        action = choose(state)
        next_state, reward, done = step(state, action)
        target = reward if done else reward + gamma * max(Q[next_state, a] for a in actions)
        Q[state, action] += alpha * (target - Q[state, action])
        state = next_state
        if done:
            break

assert choose(0) == 1
print("learned start action:", choose(0))
\`\`\``;
  if (domain.slug === "computer-vision") return `\`\`\`python
import torch
from torch import nn

class TinyVisionProjector(nn.Module):
    def __init__(self, channels=3, width=64, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(width, output_dim)

    def forward(self, images):
        features = self.encoder(images).flatten(1)
        embeddings = self.project(features)
        return nn.functional.normalize(embeddings, dim=-1)

model = TinyVisionProjector()
batch = torch.randn(4, 3, 224, 224)
output = model(batch)
assert output.shape == (4, 128)
assert torch.allclose(output.norm(dim=-1), torch.ones(4), atol=1e-5)
print(output.shape)
\`\`\``;
  if (domain.slug === "mlops") return `\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-service
  labels:
    model-version: "2026-08-27"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-service
  template:
    metadata:
      labels:
        app: model-service
    spec:
      containers:
        - name: server
          image: registry.example/model@sha256:REPLACE_ME
          resources:
            requests: {cpu: "2", memory: "8Gi"}
            limits: {cpu: "4", memory: "12Gi"}
          readinessProbe:
            httpGet: {path: /ready, port: 8080}
          env:
            - {name: MODEL_VERSION, value: "2026-08-27"}
\`\`\``;
  if (domain.slug === "ai-system-design") return `\`\`\`python
from dataclasses import dataclass
from math import ceil

@dataclass
class CapacityPlan:
    requests_per_second: float
    service_seconds: float
    concurrency_per_replica: int
    target_utilization: float = 0.70

    def replicas(self):
        # Little's Law: average in-flight work = arrival rate × service time.
        inflight = self.requests_per_second * self.service_seconds
        usable_slots = self.concurrency_per_replica * self.target_utilization
        return max(1, ceil(inflight / usable_slots))

    def with_headroom(self, failure_replicas=1):
        return self.replicas() + failure_replicas

plan = CapacityPlan(
    requests_per_second=120,
    service_seconds=0.8,
    concurrency_per_replica=16,
)
print("steady replicas:", plan.replicas())
print("N+1 replicas:", plan.with_headroom())
assert plan.with_headroom() > plan.replicas()
\`\`\``;
  return `\`\`\`python
from dataclasses import dataclass, asdict
from hashlib import sha256
import json

@dataclass(frozen=True)
class Experiment:
    question: str
    code_revision: str
    data_version: str
    seed: int
    learning_rate: float

    @property
    def run_id(self):
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return sha256(payload).hexdigest()[:12]

def record_result(experiment, metrics):
    record = {"run_id": experiment.run_id, **asdict(experiment), "metrics": metrics}
    print(json.dumps(record, indent=2, sort_keys=True))
    return record

run = Experiment(
    question="Does the proposed change improve validation loss?",
    code_revision="abc1234",
    data_version="dataset-v3",
    seed=7,
    learning_rate=3e-4,
)
record = record_result(run, {"validation_loss": 1.82})
assert record["run_id"] == run.run_id
\`\`\``;
}

function render(domain, topic) {
  const guide = domainGuides[domain.slug];
  const idea = focus[topic.slug];
  if (!guide || !idea) throw new Error(`Missing teaching profile for ${domain.slug}/${topic.slug}`);
  const prerequisites = list(topic.prerequisites, "no formal prerequisites");
  const unlocks = list(topic.unlocks, "independent synthesis and deeper work in this domain");
  const why = topic.why?.trim() || `${topic.title} supplies a missing conceptual bridge between theory and reliable practice.`;
  const primaryPrereq = topic.prerequisites[0] || "the foundations already developed in this curriculum";
  const primaryUnlock = topic.unlocks[0] || "independent project work";

  return `# ${topic.title}

## 1. Overview

${idea} In plain language, this topic is about making one difficult part of ${domain.name.toLowerCase()} explicit enough to reason about, test, and improve. The curriculum marks it as **${topic.level}** because competent work requires more than recognizing terminology: you must connect mechanisms to measurements and know when the method's assumptions stop holding.

Why does this deserve approximately ${topic.studyTime} of study? ${why} That motivation becomes practical when a system behaves unexpectedly. Instead of treating the outcome as magic, you can identify the relevant inputs, state, algorithm, resource limit, and evaluation signal. Throughout this lesson, use ${guide.lens} as the organizing lens. The aim is not memorizing one fashionable implementation. It is developing a durable mental model that transfers to new models, datasets, tools, and production constraints.

:::note
You have understood this lesson when you can explain the mechanism without jargon, sketch its data flow, name a useful metric, and design a small experiment that could prove your explanation wrong.
:::

## 2. Prerequisites in Context

The listed prerequisites are ${prerequisites}. They are not ceremonial checkboxes. **${primaryPrereq}** supplies the vocabulary needed to follow the main transformation in this lesson. The remaining prerequisites contribute complementary views: algorithmic prerequisites explain what computation is performed; statistical prerequisites explain what evidence supports a conclusion; and systems prerequisites explain what changes when the method meets finite memory, latency, data, or human-attention budgets.

Do a short readiness check before proceeding. For each prerequisite, write one sentence defining it and one sentence connecting it to ${topic.title}. If a prerequisite is a tool, reproduce a minimal workflow. If it is mathematical, explain the objects and assumptions in an equation. If it is architectural, trace one request or batch through the components. You do not need perfect mastery, but you should be able to notice when a later argument depends on it. That habit prevents a common advanced-learning failure: copying a technique while missing the condition that made it work.

## 3. Core Concept

${guide.mentalModel} For **${topic.title}**, the specific focus is this: ${idea}

A useful way to reason about the topic is to separate four layers. First, define the **contract**: inputs, outputs, permitted behavior, and success criteria. Second, identify the **mechanism** that transforms input into output. Third, expose the **feedback signal** used to learn, select, or operate the mechanism. Fourth, map the **constraints**—data quality, compute, latency, memory, safety, and human review. Most confusing explanations mix these layers. Most brittle implementations optimize one layer while silently violating another.

The practical question is therefore not “Does this technique work?” but “Under which workload, assumptions, metric, and failure budget does it improve the complete system?” Keep a baseline nearby. A complex approach earns its place only when a controlled comparison shows a meaningful benefit and its operational cost remains acceptable.

## 4. Theoretical Foundations

The architecture can be summarized as:

\`\`\`text
${guide.diagram}
\`\`\`

Two simple equations keep the discussion honest. For an evaluated outcome, define improvement over a baseline as

$$
\\Delta = M_{\\text{new}} - M_{\\text{base}},
$$

where $M$ is a clearly specified quality metric measured on the same examples and protocol. Report uncertainty, not just $\\Delta$. For a service or experimental pipeline, Little's Law provides a capacity check:

$$
C \\approx \\lambda W,
$$

where $\\lambda$ is arrival rate, $W$ is average time in the system, and $C$ is average concurrent work. The formula is simple, but it catches impossible designs: if work arrives faster than available capacity can complete it, queues grow and tail latency eventually dominates.

These equations do not replace topic-specific analysis. They create guardrails. ${idea} A complete theory states what is held fixed, which random variables or workloads are sampled, and what observation would falsify the proposed mechanism.

## 5. Worked Example

Imagine a team considering ${topic.title} for a service that processes 100 representative cases per hour. Begin with a deliberately boring baseline and record quality, median latency, 95th-percentile latency, resource use, and failure categories. Next, change only the mechanism central to this topic. Keep the dataset, split, prompt or configuration, hardware, and evaluator fixed. Suppose quality moves from 0.72 to 0.79, while average service time moves from 0.40 seconds to 0.55 seconds. The quality delta is $0.07$, and estimated concurrent work rises from $100 \\times 0.40 / 3600$ to $100 \\times 0.55 / 3600$.

The arithmetic is not the conclusion. Inspect which examples improved and which regressed. Determine whether the gain came from the intended mechanism or from leakage, extra information, a changed evaluator, or retry behavior. Then repeat across seeds, slices, or traffic windows. This example models the essential discipline: pair a causal story with controlled evidence and operational accounting.

## 6. Code / Implementation

The following small implementation is intentionally transparent. It exposes inputs and outputs, includes an assertion, and can be extended into a topic-specific experiment. Run it before adding frameworks or distributed infrastructure.

${codeFor(domain, topic)}

Treat this as a scaffold, not a production implementation. Add structured logs around every boundary, validate shapes and schemas, fix random seeds where appropriate, and persist the configuration beside results. Then make one topic-specific modification suggested by the core concept and predict the result before running it. Prediction turns coding into an experiment: a mismatch between prediction and observation reveals a gap in the mental model.

## 7. Real-World Applications

${topic.title} matters wherever decisions must remain dependable beyond a clean benchmark. In a small prototype, the mechanism can be inspected manually. At production or research scale, it must survive noisy inputs, distribution shift, partial failures, changing dependencies, limited budgets, and users who do not behave like curated test data. ${guide.practice}

Three application patterns recur. **Assistive systems** use the method while keeping a person responsible for consequential decisions. **Automated pipelines** use it behind explicit validation, rollback, and alerting boundaries. **Research platforms** use it to make experiments faster without weakening evidence quality. The correct architecture differs across these contexts because the cost of a false result, slow response, or unavailable component differs. Before adoption, write a one-page decision record with the workload, baseline, expected gain, safety boundary, owner, and rollback trigger. That document is often more valuable than another layer of abstraction.

## 8. Common Pitfalls & Misconceptions

The first misconception is that a sophisticated name guarantees a sophisticated result. It does not; implementation details, data, and evaluation usually dominate. Second, teams often compare a tuned new method against an untuned baseline, which measures effort rather than merit. Third, an average metric can hide catastrophic behavior on a rare but important slice. Fourth, offline success may not survive queueing, stale state, user feedback loops, or domain shift. Fifth, observability added after launch rarely exposes the intermediate state needed to diagnose failures.

For ${topic.title}, pay special attention to this mechanism: ${idea} Turn that statement into a test. Create an adversarial or boundary case, state the expected behavior, and capture enough intermediate output to explain the result. Use a fallback when confidence or evidence is insufficient. Complexity should be introduced one reversible step at a time; otherwise, when quality changes, you will not know which component caused it.

:::warning
Never infer safety, fairness, or factual reliability from a single aggregate benchmark. Test the actual deployment slices and define escalation paths before automating consequential decisions.
:::

## 9. How This Connects

The learning path into this lesson comes through ${prerequisites}. In particular, ${primaryPrereq} gives you the first handle on the mechanism; the other prerequisites make the analysis more realistic by adding implementation, measurement, or operational constraints. Revisit them whenever an explanation relies on a term you can name but cannot derive, implement, or test.

From here, the curriculum identifies ${unlocks}. **${primaryUnlock}** is the most immediate next step because it uses the mechanism developed here as a building block rather than an isolated trick. Build one small artifact before moving on: a notebook, trace viewer, benchmark, design document, or reproducible run. Then read the next topic while asking which assumptions carry forward and which change. If the unlock list is empty, treat this lesson as a synthesis point: connect it backward to prerequisites and outward to a real project where its constraints become concrete.

## 10. Check Your Understanding

**Q1: Explain ${topic.title} in one paragraph without using its name.**

A strong answer describes the input, transformation, output, and reason for using it. It should also mention the main constraint or failure mode. If the answer is only a list of products or acronyms, revisit the core concept.

**Q2: Why is ${primaryPrereq} a real prerequisite?**

It provides a concept or mechanism used directly by this lesson. The important connection is not chronological; it is explanatory. You should be able to point to a step in the diagram or implementation that becomes unclear without it.

**Q3: What comparison would demonstrate that the method helps?**

Use the simplest credible baseline, hold the evaluation protocol fixed, and measure both target quality and operational cost. Report uncertainty and inspect slices. The test should distinguish the proposed mechanism from extra data, compute, retries, or evaluator leakage.

**Q4: What is the most dangerous deployment mistake?**

Deploying without observable intermediate state and a rollback boundary is especially dangerous. When a failure appears, the team cannot localize it or safely reduce impact. Aggregate success metrics do not substitute for traceability.

**Q5: When should you avoid this technique?**

Avoid it when the baseline already meets requirements, assumptions do not match the workload, required evidence is unavailable, or added complexity costs more than the measured gain. A justified non-adoption decision demonstrates understanding, not lack of ambition.

## 11. Further Reading

Start with primary sources and official documentation rather than summaries. Search for the original paper or specification that introduced **${topic.title}**, then compare it with at least one careful reproduction or systems report. Read claims alongside experimental tables, ablations, and limitations. For implementation details, prefer maintained project documentation and source code at a pinned release.

A productive reading sequence is: (1) one survey for vocabulary, (2) the primary paper for mechanism and assumptions, (3) a reproduction for hidden implementation details, (4) a production or safety report for operational failures, and (5) the next curriculum topic, ${primaryUnlock}. Keep a claim-evidence notebook. For every important statement, record the source, experimental setting, and whether the evidence is causal, correlational, theoretical, or anecdotal. That practice makes further reading cumulative rather than a collection of disconnected facts.
`;
}

let created = 0;
let aliased = 0;
const refreshGenerated = process.argv.includes("--refresh");
for (const domain of curriculum) {
  for (const topic of domain.topics) {
    const outputDir = path.join(ROOT, "src/content/curriculum", domain.slug, topic.slug);
    const outputFile = path.join(outputDir, "content.mdx");
    const isGeneratedDomain = Object.hasOwn(domainGuides, domain.slug);
    if (fs.existsSync(outputFile) && !(refreshGenerated && isGeneratedDomain)) continue;

    const alias = sourceAliases[topic.slug];
    if (alias) {
      const sourceFile = path.join(
        ROOT,
        "src/content/curriculum",
        domain.slug,
        alias,
        "content.mdx",
      );
      if (fs.existsSync(sourceFile)) {
        fs.mkdirSync(outputDir, { recursive: true });
        fs.copyFileSync(sourceFile, outputFile);
        aliased += 1;
        continue;
      }
    }

    fs.mkdirSync(outputDir, { recursive: true });
    fs.writeFileSync(outputFile, render(domain, topic), "utf8");
    created += 1;
  }
}

console.log(`Created ${created} lessons and copied ${aliased} existing drafts to canonical slugs.`);
