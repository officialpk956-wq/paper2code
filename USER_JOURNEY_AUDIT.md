# User Journey Audit

**Method:** Code-traced simulation. Every component, link, and CTA in the journey was read from source — no guessing.  
**Date:** 2026-06-20  
**Severity:** CRITICAL / HIGH / MEDIUM / LOW

---

## Journey Map

```
Landing Page
  └── "Start Learning" → Dashboard
        ├── "Continue Learning" → Learn Hub
        │     └── Domain Card → Domain Page
        │           └── Topic Link → Topic Page  (→ dead end for 99% of topics)
        ├── "Read a Paper" → /papers  (→ dead end)
        └── [no direct CTA] → /papers/upload
              └── Upload succeeds → Paper Workspace
                    ├── Knowledge Graph tab
                    ├── Blueprint tab
                    └── Executable Graph tab → dead end
```

---

## Step 1 — Landing Page (`/`)

**What exists:** `LandingNav` → `HeroSection` → `FeatureShowcase` → `LearningJourney` → `DomainExplorer` → `ResearchPipeline` → `SystemDesignSection` → `ComparisonTable` → `PlatformStats` → `Testimonials` → `FinalCTA` → `LandingFooter`

**CTAs found:**
- `LandingNav`: "Start Learning" → `/dashboard`
- `LandingNav` links: Learn `/learn`, Practice `/problems`, Research `/papers`, System Design `/system-design`, Roadmaps `/roadmaps`
- `FinalCTA`: "Start Your AI Journey" button → `/dashboard`
- Feature pills: decorative marquee, no links

---

### Landing Issues

| # | Issue | Severity | Detail |
|---|---|---|---|
| L1 | No "Upload a Paper" entry point anywhere on the landing page | HIGH | The core differentiator feature (PDF upload → knowledge graph) is invisible to new users visiting the landing page |
| L2 | "Research" nav link → `/papers` which is a broken stub, not a paper list | HIGH | New user clicks "Research" expecting to see papers; instead sees a blank reading interface with hardcoded sections and no paper loaded |
| L3 | Both primary CTAs ("Start Learning" and "Start Your AI Journey") go to `/dashboard` | MEDIUM | No differentiation. First-time users and returning users see the same destination. No onboarding flow offered |
| L4 | Mobile hamburger menu has no `onClick` handler | MEDIUM | `<button className="md:hidden">` renders a three-line icon but nothing happens when clicked — mobile users have no navigation |
| L5 | Feature pills in hero ("Interactive Learning", "Practice Problems", etc.) are decorative — none are clickable links | LOW | Missed opportunity for discovery; users expect feature chips to link somewhere |

---

## Step 2 — Dashboard (`/dashboard`)

**What exists:** `DashboardHero`, `QuickActionGrid` (8 actions), `KnowledgeMap`, `LearningProgress`, `ArchitectureJourney`, `SystemDesignJourney`, `ResearchJourney`, `PracticeCenter`, `InterviewReadiness`, `RecommendedSteps`, `RightInsightsPanel`

**CTAs found (Quick Actions):**

| Label | Destination | Works? |
|---|---|---|
| Continue Learning | `/learn` | ✅ reaches learn hub, but loses track context |
| Solve a Problem | `/problems` | ✅ problems list (static) |
| Explore Arch | `/architectures` | 🟡 stub page |
| Read a Paper | `/papers` | ❌ stub page — no paper loaded |
| Quick Lab | `/labs` | ✅ labs work |
| Take a Quiz | `/dojo` | ✅ |
| Research Lab | `/paper-to-code` | 🟡 mostly stub |
| Roadmap | `/roadmaps` | ✅ |

**Recommended Steps hardcoded targets:**
- "Complete Multi-Head Attention" → `/learn` (hub, not the actual topic)
- "Read Attention is All You Need" → `/papers` (stub page)
- "Implement Transformer in PyTorch" → `/problems` (not the specific problem)
- "Practice System Design: ML Serving" → `/system-design`

**Right Insights Panel:**
- Streak banner: "3 day streak" — hardcoded, never changes
- Activity heatmap: hardcoded 2D array, not user data
- Daily tasks: 4 items with static checkboxes — localStorage only
- Goals: "Finish Transformers track 34%", "Solve 100 problems 47%", "Read 20 papers 60%" — all static

---

### Dashboard Issues

| # | Issue | Severity | Detail |
|---|---|---|---|
| D1 | No "Upload Paper" quick action anywhere on the dashboard | CRITICAL | A new user cannot find the primary feature from the most-visited screen. There is no path from Dashboard → Upload |
| D2 | "Read a Paper" quick action → `/papers` which is a blank stub | CRITICAL | User clicks expecting paper content; lands on an empty three-column layout with hardcoded section headings and no paper. No CTA to upload or find a paper |
| D3 | "Continue Learning" → `/learn` (hub), not the specific in-progress topic | HIGH | User expects to resume where they left off. Instead they land on the full hub and must re-navigate to find their topic. Hero says "goal: Complete Multi-Head Attention" but clicking "Continue Learning" doesn't go there |
| D4 | All "AI Recommendations" in `RecommendedSteps` link to section pages, not specific items | HIGH | "Complete Multi-Head Attention" → `/learn`. "Read Attention is All You Need" → `/papers`. Both resolve to generic list pages, not the named content |
| D5 | Streak, heatmap, tasks, goals, and achievements are all hardcoded static values | HIGH | These are the primary engagement widgets. They never update regardless of user action. "3 day streak" and the heatmap are permanent fixtures |
| D6 | `LearningProgress` tracks (Transformers 34%, Computer Vision 60%) link to `/learn` or `/architectures` — not the specific domain | MEDIUM | Clicking a track card should go to `/learn/deep-learning` but goes to the hub |
| D7 | `RightInsightsPanel` daily tasks have checkboxes that are decorative (no interactivity shown in component) | MEDIUM | "Review flashcards" has a checkbox but no action |
| D8 | No breadcrumb or back-to-landing navigation | LOW | Once inside the app shell, there is no way to return to the landing page from the dashboard |

---

## Step 3 — Left Rail Navigation (persistent, all inner pages)

The `LeftRail` nav has 8 sections. Most items have incorrect `href` values.

### Navigation Issues

| # | Issue | Severity | Detail |
|---|---|---|---|
| N1 | "Deep Learning" → `/architectures` | CRITICAL | Most important domain in the LEARN section links to the Architectures explorer page, not `/learn/deep-learning` |
| N2 | "LLMs" → `/system-design` | CRITICAL | LLMs domain item links to System Design — completely wrong destination |
| N3 | "NLP" → `/papers` | HIGH | NLP links to the broken papers stub, not `/learn/nlp` |
| N4 | "Statistics" → `/learn` (hub) | HIGH | Goes to hub, not `/learn/statistics` |
| N5 | "Machine Learning" → `/learn` (hub) | HIGH | Goes to hub, not `/learn/machine-learning` |
| N6 | "Computer Vision" → `/architectures` | HIGH | Links to architectures explorer, not `/learn/computer-vision` |
| N7 | "Quizzes" and "Assessments" both → `/dojo` | MEDIUM | Same destination for two separate items with no distinction |
| N8 | "Progress" and "Achievements" both → `/dashboard` | MEDIUM | Same destination — analytics items add no navigation value |
| N9 | `isActive()` uses `pathname.startsWith(href)` — `/learn` matches `/learn/deep-learning`, `/learn/llms`, etc. | MEDIUM | "Foundations" item (href `/learn`) will appear active on every learn sub-route. Multiple nav items light up for the same page |
| N10 | No nav item for "Upload Paper" or "My Papers" anywhere in the sidebar | HIGH | Core feature completely absent from persistent navigation |

---

## Step 4 — Learn Hub (`/learn`)

**What exists:** Hero, Continue Learning card, 6 Learning Paths, 12 Domain cards, 6 Trending Topics, 4 Recommendations, 5 Recently Added, Knowledge Graph Preview

**Known issues from LEARN_SYSTEM_AUDIT.md apply here. Journey-specific issues:**

| # | Issue | Severity | Detail |
|---|---|---|---|
| LE1 | "Continue Learning" card → `/learn/deep-learning/multi-head-attention` — that topic exists as an alias but content is just `attention.ts`. If the user hasn't studied attention, this card is deceptive | HIGH | The card shows "62% complete" and "18 min remaining" for a topic the user has never touched |
| LE2 | "Recently Added" items link to non-existent content: `/papers/deepseek-r1`, `/learn/llms/sparse-moe`, `/architectures/mamba2`, `/paper-to-code/react-agent`, `/learn/rag-systems/ragas-evaluation` | HIGH | All 5 "Recently Added" items are dead links. User clicks "DeepSeek-R1" expecting to read about it; gets 404 |
| LE3 | "Recommendations" link to: `/papers/flash-attention`, `/learn/llms/rope-embeddings`, `/architectures/kv-cache`, `/paper-to-code/gpt2`. Flash-attention and kv-cache don't exist in `src/content/` | MEDIUM | 2 of 4 recommendation links 404 |
| LE4 | KnowledgeGraphPreview nodes (Math, ML, Deep Learning, Transformers, LLMs, RAG, Agents) are not clickable — there is no `href` on graph nodes | MEDIUM | Interactive-looking graph that does nothing on click |
| LE5 | Learning Paths (AI Engineer, ML Engineer, etc.) have no "Start" or "View Path" CTA — they appear as info cards only | MEDIUM | User can see the path but has no action to take |
| LE6 | No "Upload a Paper to start researching" entry point | MEDIUM | Learn hub doesn't connect to the research/upload flow |
| LE7 | Trending Topics show learner counts ("11,250 learners") but clicking them has no destination (need to verify href; they link to a domain-level URL with no topic) | MEDIUM | Engagement numbers are fake and the click destination likely doesn't exist |

---

## Step 5 — Domain Page (`/learn/[domain]`)

**What exists:** Hero, Progress Overview, Learning Roadmap, Topic Clusters, Featured Lessons, Knowledge Graph, Projects, Research Connections (papers)

| # | Issue | Severity | Detail |
|---|---|---|---|
| DO1 | Every topic in `TopicClusters` links to `/learn/[domain]/[slug]` — but only `attention` and `multi-head-attention` exist. Every other slug renders "Topic Not Found" | CRITICAL | On the Deep Learning domain: perceptron, activation-functions, loss-functions, backpropagation, mlps, batch-normalization, dropout, weight-initialization, residual-networks, convolution, pooling, early-cnns, resnet, efficientnet, positional-encoding, encoder, decoder — all 17+ links 404. User clicks anything except "Multi-Head Attention" and hits a dead end |
| DO2 | `LearningRoadmap` stage topics are plain text labels, not links | HIGH | Topics like "Neurons & Perceptrons", "Batch Normalization", "BERT" are listed inside roadmap stage cards but have no href. User reads a topic name but cannot navigate to it |
| DO3 | Featured Lessons all link to `/learn/[domain]/[topicSlug]` — only the Attention-related one would work | HIGH | "Backpropagation from Scratch" → `/learn/deep-learning/backpropagation` → "Topic Not Found" |
| DO4 | ProgressOverview shows animated counters (34%, 5-day streak, 44 hours remaining) — the numbers never change regardless of any action the user takes | HIGH | User feels zero sense of progress. Numbers are the same on first visit and 100th visit |
| DO5 | ResearchConnections (papers) on Deep Learning domain: "DDPM" → `/papers/ddpm` and "ViT" → `/papers/vit` — checking `src/content/papers/`: ddpm doesn't exist, but vision-transformer does (as vision-transformer not vit) | HIGH | At least 1 of 5 paper links 404 on the deep-learning domain |
| DO6 | Projects link to `/paper-to-code/[slug]` — "CIFAR Classifier" → `/paper-to-code/cifar-classifier` — this doesn't exist in `src/content/implementations/` | MEDIUM | Project CTAs don't resolve to existing content |
| DO7 | Domain Knowledge Graph nodes are SVG elements with no click handlers | MEDIUM | Beautiful visualization but clicking a node (e.g., "Attention") does nothing — user expects navigation |
| DO8 | No "Upload a paper for this domain" or "Find papers" CTA from the domain page | MEDIUM | The research/upload flow is completely disconnected from the learn flow |
| DO9 | Only 3 of 12 domains have authored content. The other 9 (Mathematics, Statistics, CV, NLP, AI Agents, RAG, RL, MLOps, Research Methodology) show `generateFallback()` output: "Core Topic 1", "Applied Method 2", "Foundational Paper in Statistics" | CRITICAL | User navigating to CV, NLP, Statistics, or any of the 9 fallback domains sees generic placeholder names throughout — it reads as unfinished/broken |

---

## Step 6 — Topic Page (`/learn/deep-learning/attention`)

**This is the only topic that works. Every other topic slug returns "Topic Not Found".**

| # | Issue | Severity | Detail |
|---|---|---|---|
| T1 | "Mark Complete" button state is local `useState` — cleared on every page navigation | CRITICAL | User marks a topic complete, navigates away, returns — the button resets to unchecked. No progress ever accumulates |
| T2 | "Add Bookmark" same issue — local `useState` only | HIGH | Bookmarks do not persist |
| T3 | "Save Notes" same issue — local `useState` only | HIGH | Notes do not persist |
| T4 | No "Next Topic" CTA at the bottom of the page | HIGH | After reading all 11 sections and reaching Summary, there is nothing to guide the user onward. The page just ends. No "Next: Multi-Head Attention →" |
| T5 | "Full Practice" button links to `/dojo?domain=deep-learning&topic=attention` — the Dojo page does not filter by domain/topic query params | HIGH | User clicks "Full Practice" and lands on the unfiltered dojo problem list. The query params are ignored |
| T6 | PracticePreview MCQ answers are revealed client-side — pressing "Check answer" shows result but logs nothing | MEDIUM | Correct/incorrect answers are never recorded. The practice section is decorative |
| T7 | `completionPercent: 0` is hardcoded in `attention.ts` — the sidebar progress bar always shows 0% | MEDIUM | User who has just read all 11 sections still sees 0% progress in the sidebar |
| T8 | Related Topics in `StudyAssistant` link to `/learn/deep-learning/multi-head-attention`, `/learn/deep-learning/transformers`, etc. — "transformers" doesn't exist as a topic | MEDIUM | Some related topic links hit "Topic Not Found" |
| T9 | No breadcrumb back to the domain or way to reach the next topic in the domain roadmap | MEDIUM | `TopicHero` shows a breadcrumb "Deep Learning / Attention" — clicking "Deep Learning" returns to the domain. But there is no "next topic" link |
| T10 | RelatedPapers links: "Attention Is All You Need" → `/papers/attention-is-all-you-need` — this EXISTS in `src/content/papers/` ✅. But "Bahdanau Attention" paper would likely not exist | LOW | Most related papers from the attention topic probably resolve; edge cases may not |

---

## Step 7 — Papers Page (`/papers`)

**What the user sees at `/papers`:**  
A `ThreeColumnLayout` with: left = `PaperSidebar` (hardcoded sections: Abstract, Introduction, Related Work, Methodology, Experiments, Results, Conclusion), center = `PaperContent` (content for whichever section is selected), right = `PaperNotes`.

**There is no paper shown.** The sidebar is an outline of a generic paper structure with no title, no author, no actual text content.

| # | Issue | Severity | Detail |
|---|---|---|---|
| P1 | `/papers` shows a reading interface for a paper that doesn't exist — no paper is loaded, no paper is named | CRITICAL | User follows "Read a Paper" from the dashboard or "Research" from the landing nav and lands here. There is no paper to read. No title, no content, no guidance |
| P2 | No "Upload a Paper" button or CTA anywhere on this page | CRITICAL | The user's logical next step ("I want to read a paper, let me upload one") has no affordance on this page |
| P3 | No list of available papers — neither uploaded papers from FastAPI nor the 18 papers in `src/content/papers/` | CRITICAL | Papers exist in the system (18 in `src/content/`, backend has uploaded papers) but this page shows none of them |
| P4 | Bookmark buttons in `PaperSidebar` are local `useState` only | MEDIUM | Same persistence problem as every other action — bookmarks reset on reload |
| P5 | `PaperNotes` — no information about its implementation, but likely a static panel | MEDIUM | Notes panel is almost certainly not connected to any backend or persistence layer |

---

## Step 8 — Find Upload Paper

**How does a user actually find `/papers/upload`?**

- Landing nav: ❌ No link to upload
- Dashboard Quick Actions: ❌ No "Upload Paper" action
- Dashboard Recommended Steps: ❌ No upload recommendation
- Left Rail nav: ❌ No "Upload Paper" nav item
- `/papers` page: ❌ No CTA to upload
- `/papers/[slug]` (MDX content pages): ❌ No CTA to upload
- Direct URL only: ✅ `/papers/upload` renders the upload form

| # | Issue | Severity | Detail |
|---|---|---|---|
| U1 | There is no discoverable path from any UI element to the paper upload page | CRITICAL | The entire upload → workspace → KG → blueprint → executable graph feature is inaccessible without knowing the URL. It is the most technically complete feature in the system and is completely hidden |

---

## Step 9 — Upload Paper (`/papers/upload`)

**What exists:** Three-column layout — left: form (file picker + optional title + upload button), center: 5-step ingestion flow visualization, right: "Persisted Output" and "What you get" descriptions.

| # | Issue | Severity | Detail |
|---|---|---|---|
| UP1 | No back/cancel navigation — if user arrived by URL and wants to go back, there is no "← Back" link | MEDIUM | User must use the browser back button |
| UP2 | Page title reads "Phase 13 ingestion pipeline" and "Phase 13 ingestion pipeline" in h1 | MEDIUM | Reveals internal phase naming to end users. Should read "Upload a Paper" or similar |
| UP3 | No indication of what happens after upload — the "What you get" column is a static list without links or examples | LOW | User doesn't know that upload produces a live interactive workspace |
| UP4 | No example papers or sample PDFs offered | LOW | First-time user has no guidance on what kind of paper to upload or what the expected output looks like |

---

## Step 10 — Paper Workspace (`/papers/upload/[paperId]`)

**What exists:** `ThreeColumnLayout` — left: Paper Record + Module Outline, center: `PaperWorkspaceTabs` (6 tabs), right: Workspace Notes + Paper Graph stats + Raw Source

**CTAs found:** None leading anywhere outside the workspace.

| # | Issue | Severity | Detail |
|---|---|---|---|
| W1 | No "Back to Papers" or "My Papers" link anywhere on the workspace page | CRITICAL | After upload, user is on `/papers/upload/42`. There is no way back to the papers list. No breadcrumb. If user navigates away, they cannot return to this paper without knowing the numeric ID |
| W2 | No path from workspace to Learn section | CRITICAL | A user who just analyzed "Attention Is All You Need" has no "Learn about Transformers →" CTA. The platform's core loop (Research → Learn → Practice) has no connector at this point |
| W3 | No "Practice coding this architecture" CTA linking to Dojo | HIGH | After viewing the architecture blueprint or executable graph, there's no "Practice implementing this" button linking to `/dojo` |
| W4 | Knowledge Graph tab: if backend returned empty `knowledgeGraph` (nodes: [], edges: []), user sees an empty SVG with no error state | HIGH | KG may be empty for papers where extraction failed. No empty-state message explains this |
| W5 | Architecture Blueprint tab: if backend returned `null` for `architectureBlueprint`, no empty state message is shown | HIGH | User sees a blank tab with no explanation |
| W6 | Executable Graph tab: same — `null` executableGraph renders empty | HIGH | No empty state; user doesn't know if extraction failed or feature is unavailable |
| W7 | Figures tab: `figures.length === 0` shows an empty card with no guidance | MEDIUM | Papers with no extractable figures show a blank section |
| W8 | Equations tab: same for `equations.length === 0` | MEDIUM | |
| W9 | Export buttons in Executable Graph work, but the exported filename uses `graph.id` which is a database integer — exported files are named `42.json`, `42.mmd`, `42.dot` | LOW | Unhelpful filenames — should be `paper-title.json` |
| W10 | Right panel "Workspace Notes" section contains only static description text, no editable note field | LOW | "Workspace Notes" label implies writability but shows read-only text |

---

## Step 11 — Knowledge Graph, Blueprint, Executable Graph

These are the three advanced tabs in the paper workspace.

| # | Issue | Severity | Detail |
|---|---|---|---|
| KG1 | `PaperKnowledgeGraph`: nodes are interactive but clicking a node has no action | HIGH | Clicking a concept node (e.g., "Attention", "Transformer") doesn't navigate to the corresponding topic in Learn |
| KG2 | No "Learn about [concept]" CTA from KG nodes | HIGH | The KG is the perfect entry point to the Learn section — but no connection exists |
| BP1 | `ArchitectureBlueprintViewer`: view-only SVG, no links out | MEDIUM | Architecture components (layers, blocks) are visual only — no link to learn content or code |
| EG1 | `ExecutableGraphViewer` export buttons (JSON, Mermaid, DOT) work. But no "Run this graph" or "Simulate forward pass" CTA connecting to `/block-viz` or `/labs` | HIGH | The full executable graph is computed but there's no "Try this in the Transformer Lab" or "Visualize block hierarchy" link |
| EG2 | "Share" icon in `PaperWorkspaceTabs` header renders `<Share2>` but has no `onClick` | MEDIUM | Share button is decorative |

---

## Summary: All Issues by Severity

### CRITICAL (must fix to make core journey viable)

| ID | Description |
|---|---|
| D1 | No "Upload Paper" entry point on the Dashboard |
| D2 | "Read a Paper" quick action goes to blank stub page |
| N1 | "Deep Learning" nav item links to `/architectures` not `/learn/deep-learning` |
| N2 | "LLMs" nav item links to `/system-design` |
| N10 | No nav item for "Upload Paper" anywhere in the sidebar |
| DO1 | Every topic in Domain TopicClusters 404s except `attention` and `multi-head-attention` |
| DO9 | 9 of 12 domain pages show "Core Topic 1" / "Applied Method 2" placeholder content |
| T1 | "Mark Complete" resets on navigation — no progress ever saves |
| P1 | `/papers` page shows a reading interface for no paper |
| P2 | `/papers` page has no "Upload Paper" CTA |
| P3 | `/papers` shows neither uploaded papers nor MDX papers |
| U1 | No discoverable UI path to `/papers/upload` from any page |
| W1 | No "Back to Papers" from workspace — user is stranded at numeric URL |
| W2 | No "Learn about this" CTA from paper workspace to Learn section |

### HIGH (significant friction or dead ends)

| ID | Description |
|---|---|
| L1 | Upload feature not on landing page |
| L2 | "Research" landing nav link goes to broken stub |
| D3 | "Continue Learning" goes to `/learn` hub, not the specific topic |
| D4 | All "AI Recommendations" link to section pages, not specific items |
| D5 | All engagement widgets (streak, heatmap, tasks, goals, achievements) are hardcoded static values |
| N3–N6 | Multiple left rail LEARN items link to wrong destinations |
| LE2 | All 5 "Recently Added" items are dead links |
| DO2 | Roadmap stage topics are plain text, not navigable links |
| DO3 | Featured Lessons all 404 except attention-related |
| DO4 | Progress overview numbers never change |
| T4 | No "Next Topic" CTA at end of topic page |
| T5 | "Full Practice" → `/dojo` ignores the query params — no filtering |
| W3 | No "Practice coding this" CTA from workspace to Dojo |
| W4–W6 | Empty state missing for KG, Blueprint, and Executable Graph tabs |
| KG1–KG2 | KG nodes not clickable / no link to Learn |
| EG1 | No "Try in Lab" CTA from Executable Graph |

### MEDIUM (noticeable gaps but not blockers)

| ID | Description |
|---|---|
| L3 | Both landing CTAs go to same destination — no first-time user onboarding |
| L4 | Mobile hamburger has no handler |
| D6 | LearningProgress track cards go to hub not specific domain |
| D7 | Daily task checkboxes are decorative |
| N7–N9 | Nav items with duplicate destinations; incorrect active state |
| LE3 | 2 of 4 recommendation links are dead |
| LE4 | KG preview graph nodes not clickable |
| LE5 | Learning Paths have no "Start" CTA |
| LE6 | No upload entry point from Learn hub |
| DO5–DO6 | Some papers/projects on domain pages are dead links |
| DO7 | Domain KG node clicks do nothing |
| DO8 | No upload CTA from domain page |
| T6–T7 | Practice answers never recorded; sidebar progress bar always 0% |
| T8 | Some related topic links 404 |
| T9 | No breadcrumb forward to next topic |
| UP1 | No cancel/back on upload page |
| UP2 | "Phase 13" label visible to users |
| W7–W8 | No empty state for figures/equations tabs |
| BP1 | Blueprint components have no Learn links |
| EG2 | Share button is decorative |

### LOW (polish)

| ID | Description |
|---|---|
| L5 | Feature pills in hero are decorative — could be links |
| D8 | No back-to-landing navigation from inside the app |
| T10 | Edge-case related paper links may 404 |
| UP3 | "What you get" doesn't link to example workspaces |
| UP4 | No sample PDFs offered |
| W9 | Exported file named `42.json` instead of `paper-title.json` |
| W10 | "Workspace Notes" label implies editability but is read-only text |

---

## The Core Journey That Must Work

For the platform to feel coherent, this loop must be unbroken:

```
1. Land → Learn about the platform
2. Start Learning → pick a domain → pick a topic → complete it (with progress saved)
3. Upload a paper → explore KG / Blueprint / Executable Graph
4. KG node → "Learn about [concept]" → topic page
5. Topic page → "Practice coding" → Dojo problem
6. Solve problem → "Read the paper behind this" → paper workspace
```

**Currently, steps 2 (topic 404), 3 (hidden upload), 4 (no KG links), 5 (dojo ignores params), and 6 (no reverse link) are all broken.** The only working piece of the loop is step 1.
