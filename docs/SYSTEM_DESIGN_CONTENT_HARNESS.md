# System Design Content Harness — All Batches

## STATUS (completed 2026-08-27)
`src/data/content/systemDesign.ts` defines 12 systems (`SD_SYSTEMS`), each
with 4 metadata modules — Beginner / Intermediate / Advanced / Research —
carrying `learningObjectives`, `prerequisites`, `diagramsNeeded`,
`caseStudies`, `handsOnProjects`, `interviewQuestions`. All 12 canonical
systems now have routed long-form prose and validated metadata.
`/system-design/[slug]/page.tsx` reads those exact slugs through
`getMdxContent('system-design', slug)`.

Run `npm run audit:system-design` after edits. It cross-checks every source
objective, prerequisite, requested diagram, case study, interview question,
and hands-on project; it also validates metadata, minimum prose/case-study
lengths, control characters, and complete MDX/KaTeX compilation.

**Do not confuse with:** `src/content/system-design/` already has 11
existing `content.mdx` files (`chatgpt-system-design`, `github-copilot`,
`netflix-recommendation`, `perplexity`, `multi-agent`, `single-agent`,
`basic-rag`, `agentic-rag`, `advanced-rag`, `recommendation-engine`,
`tiktok-recommendation`, `youtube-recommendation`). Those use a different
slug set that no route currently reads by slug — leave them alone, this
harness only targets the 12 `SD_SYSTEMS` slugs below.

**Build gotcha — do not skip `meta.json`:** unlike the Curriculum harness,
`system-design` IS a validated content type in
`scripts/generate-content-index.mjs`. If a new
`src/content/system-design/{slug}/` directory is missing `meta.json`, the
`prebuild` script hard-fails (`process.exit(1)`) and breaks the whole build.
Every file created here needs a sibling `meta.json`.

## HOW TO USE
Copy **MASTER PREAMBLE** + one **BATCH** block into Antigravity per session.
Each batch creates content.mdx + meta.json for the listed systems. Antigravity
has repo access — instruct it to open `src/data/content/systemDesign.ts` and
read the named system's 4 modules directly; don't let it invent different
learning objectives, case studies, or interview questions than what's
already there — those are the spec.

---

## MASTER PREAMBLE (paste at the top of every prompt)

You are writing one long-form MDX system design article per system for
paper2code, an educational ML platform. Each article must teach a reader who
starts at Beginner all the way to Research-frontier understanding of that
system, in one file, structured as four progressively deeper sections.

Before writing each file: open `src/data/content/systemDesign.ts`, find the
system by `slug`, and read its 4-entry `modules` array in order
(`modules[0]` = Beginner, `modules[1]` = Intermediate, `modules[2]` =
Advanced, `modules[3]` = Research). Every `learningObjectives`,
`caseStudies`, `diagramsNeeded`, and `interviewQuestions` entry in that
system's data must be covered somewhere in the corresponding section — treat
the array as a checklist, not inspiration.

Each system produces two files:
- `src/content/system-design/{slug}/content.mdx`
- `src/content/system-design/{slug}/meta.json`

Use GitHub-flavored Markdown with KaTeX math (`$inline$`, `$$block$$`).

**`content.mdx` required section structure (copy exactly):**

```
# [System Name]

## Beginner: Foundations
### Diagrams
### Case Studies
### Interview Questions & Answers

## Intermediate: Design Deep Dive
### Diagrams
### Case Studies
### Interview Questions & Answers

## Advanced: Production at Scale
### Diagrams
### Case Studies
### Interview Questions & Answers

## Research: Frontiers
### Diagrams
### Case Studies
### Interview Questions & Answers

## Hands-On Projects
```

**Content rules:**
- The prose under each level heading (before its `### Diagrams` subsection) must teach to every item in that module's `learningObjectives`, and must state its `prerequisites` as a short callout at the top (`:::note Prerequisites ... :::`).
- `### Diagrams` must render one labeled ASCII/text diagram per entry in that module's `diagramsNeeded` — not fewer. If `diagramsNeeded` has 4 entries, there are 4 diagrams.
- `### Case Studies` must narrate every entry in that module's `caseStudies` as a short (150+ word) story with concrete numbers where the source text implies them (e.g. request volume, latency targets) — don't just restate the one-line description.
- `### Interview Questions & Answers` must answer every question in that module's `interviewQuestions`, in full, multi-sentence answers — these are already written questions, just answer them.
- `## Hands-On Projects` aggregates every module's `handsOnProjects`. Several modules have an empty `handsOnProjects: []` in the source — when a module's list is empty, propose 2-3 concrete hands-on projects appropriate to that level yourself and label them `(proposed)`.
- Include real equations in KaTeX wherever the system has one (e.g. HNSW search complexity, KV-cache memory formula, recommendation ranking score) — don't skip math just because this is "system design," these are ML systems.
- Every level's prose section is substantive — minimum 300 words per level (this is a long-form article, not a lesson card).

**`meta.json` required shape:**

```json
{
  "type": "system-design",
  "slug": "{slug}",
  "title": "{Name from SD_SYSTEMS}",
  "description": "{1-2 sentence summary}",
  "tags": ["system-design", "..."],
  "difficulty": "advanced"
}
```

`difficulty` must be exactly one of `beginner` | `intermediate` | `advanced`
(lowercase — this is validated). Use `"advanced"` for all 12: every article
here culminates in Research-tier content, so `advanced` is the closest valid
bucket. `slug` must match the directory name exactly and be kebab-case.

---

## BATCH SD-1 — Retrieval & Ranking Systems

**Create content.mdx + meta.json for:**
- `src/content/system-design/vector-databases/`
- `src/content/system-design/search-engines/`
- `src/content/system-design/recommendation-systems/`
- `src/content/system-design/rag-systems/`

**Context:** These four share a lot of vocabulary (embeddings, ANN indexes,
ranking). Don't let `rag-systems` re-explain vector-database internals from
scratch — its Beginner section should link to `vector-databases` for that
and focus on the retrieval-augmentation loop itself (retrieve → rerank →
inject → generate). `recommendation-systems` should cross-reference the
`ai-system-design`/`recommendation-systems`-adjacent curriculum topics if
present (see the Curriculum Content Harness) rather than duplicating theory.

---

## BATCH SD-2 — Serving & Agent Infrastructure

**Create content.mdx + meta.json for:**
- `src/content/system-design/llm-serving/`
- `src/content/system-design/agent-systems/`
- `src/content/system-design/youtube-architecture/`
- `src/content/system-design/netflix-architecture/`

**Context:** `llm-serving` Advanced/Research sections must cover continuous
batching, PagedAttention, and KV-cache memory formulas explicitly — this is
the most technically dense system in the whole set per its source
`learningObjectives`. `youtube-architecture` and `netflix-architecture` are
full consumer-scale case-study systems — their Beginner sections should
still start from first principles (CDN basics, recommendation basics) before
narrating the named company's specific design in later tiers.

---

## BATCH SD-3 — Consumer-Scale & Frontier Model Systems

**Create content.mdx + meta.json for:**
- `src/content/system-design/tiktok-architecture/`
- `src/content/system-design/uber-architecture/`
- `src/content/system-design/chatgpt-architecture/`
- `src/content/system-design/deepseek-architecture/`

**Context:** `chatgpt-architecture` and `deepseek-architecture` Research
sections should reference the platform's own architecture pages where
relevant (`src/content/architectures/gpt-4`, `mixtral-8x7b`,
`deepseek-v2`) instead of re-deriving MoE/MLA math — link out for the
model internals, stay focused on the *system* (serving fleet, routing,
safety layers, cost) here. `uber-architecture` should lean on its
`ml-platform-engineering` curriculum-topic overlap (Domain 10) for the
platform-engineering angle and link to it rather than duplicating.

---

## QUALITY CHECKLIST (Antigravity must verify before finishing each system)

- [ ] All 4 levels present as H2 sections in the exact order Beginner → Intermediate → Advanced → Research
- [ ] Every `learningObjectives` item from that system's source data is taught somewhere in the matching level
- [ ] Diagram count under `### Diagrams` matches the `diagramsNeeded` count for that module — no fewer
- [ ] Every `caseStudies` entry is narrated at 150+ words, not just restated
- [ ] Every `interviewQuestions` entry has a full multi-sentence answer under `### Interview Questions & Answers`
- [ ] Empty `handsOnProjects` arrays are backfilled with 2-3 proposed projects, labeled `(proposed)`
- [ ] `meta.json` exists, `type` is `"system-design"`, `slug` matches the directory name, `difficulty` is lowercase `"advanced"`
- [ ] No section is a stub — each level's prose is 300+ words
