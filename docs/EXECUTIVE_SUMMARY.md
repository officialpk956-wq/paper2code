# Executive Summary — Platform Reality Verification
**Verified:** 2026-06-20 | **Method:** Direct source code reads, no reliance on prior audit documents

---

## The Core Problem

The platform's primary feature — upload a paper, analyze it, learn from it — is invisible to users and unreachable from any navigation element.

Upload Paper appears in: **zero** landing CTAs, **zero** left-rail nav items, **zero** dashboard quick-actions. A user who correctly guesses `/papers/upload` and successfully completes a PDF upload lands in the workspace — with no way to get back, no recommendations for what to do next, and no path from the knowledge graph to the learn section.

---

## What Actually Works (verified)

| Feature | Status |
|---|---|
| PDF upload → FastAPI → DB → workspace | ✅ End-to-end working |
| Paper workspace: 6 tabs with real data | ✅ Working |
| KG nodes are clickable, panel appears | ✅ Click infrastructure exists |
| `/learn/deep-learning/attention` topic | ✅ 11 sections, good content |
| `/learn/deep-learning`, `/learn/machine-learning`, `/learn/llms` domain pages | ✅ 3 of 12 render real content |
| `src/content/` directory | ✅ EXISTS: 84 content slugs (31 arch, 19 papers, 9 impl, 12 sysdesign, 8 problems, 1 roadmap, 2 interview, 2 math) |
| Dojo problems | ✅ Working |
| FastAPI: 30+ routes implemented | ✅ Backend is solid |

---

## What Is Broken (verified)

| # | Issue | Severity | Evidence |
|---|---|---|---|
| 1 | 6 of 8 LeftRail LEARN section hrefs are wrong (e.g., Deep Learning → /architectures, NLP → /papers) | **P0** | `left-rail.tsx`, hrefs confirmed |
| 2 | Upload Paper absent from landing page, sidebar, dashboard | **P0** | `hero-section.tsx`, `left-rail.tsx`, `quick-actions.tsx` |
| 3 | All 9 "Recently Added" + "Recommendations" items on Learn hub are dead links | **P0** | `data/learn/recommendations.ts` — all 9 slugs verified absent from `src/content/` |
| 4 | `/papers` page is a blank stub (PaperSidebar + PaperContent for a paper that doesn't exist) | **P0** | `src/app/papers/page.tsx` |
| 5 | KG node panel has no "Learn This Concept" CTA | **P1** | `PaperKnowledgeGraph.tsx` lines 329-348: panel exists, no learn button |
| 6 | Topic page ends with no Practice CTA, no next-topic link | **P1** | `[topic]/page.tsx` — no end-of-content section |
| 7 | StudyAssistant (Mark Complete, Bookmark, Notes) uses pure `useState` — resets on every navigation | **P1** | `StudyAssistant.tsx` |
| 8 | `POST /api/progress/update` schema mismatch for topic completion | **P1** | `server.py` line 844: `{paper_id: int, module_id: int}` — no topic slug |
| 9 | `GET /api/papers` response missing `created_at` field | **P1** | `server.py` line 339: field exists on Paper model, not included in response |
| 10 | All 11 topic routes except "attention" return "Topic Not Found" | **P1** | `data/topics/index.ts`: only 1 topic registered |
| 11 | Paper workspace has no back-navigation to /papers | **P2** | `papers/upload/[paperId]/page.tsx` |
| 12 | Dojo problems have `relatedPapers` data but never surface it post-solve | **P2** | `data/problems.ts` grep: `relatedPapers: ["attention-is-all-you-need"]` |

---

## The Right Three Moves (in order)

**Move 1 — Make the core feature discoverable (P0, < 1 day)**

Fix the 6 broken LeftRail LEARN hrefs. Add Upload Paper to LeftRail + Dashboard. Replace 9 dead links with 9 working links from `src/content/`. This alone removes the most trust-breaking issues.

Files: `left-rail.tsx`, `quick-actions.tsx`, `data/learn/recommendations.ts`

**Move 2 — Build the Research Hub (P1, 2–3 days)**

Rewrite `/papers` to show the paper list from `GET /api/papers`. Add a proxy route (`src/app/api/papers/route.ts`). Add `created_at` to backend response. Add Upload CTA to the landing hero. This turns the blank stub into the platform's actual front door for research.

Files: `papers/page.tsx`, new `api/papers/route.ts`, `server.py`, `hero-section.tsx`

**Move 3 — Close the loop from KG to Learn (P1, 1–2 days)**

Add "Learn This Concept →" CTA to the KG node panel. The click infrastructure already exists — the panel just needs the button. Add a concept-to-topic mapping function. Add a "Practice This" section at the bottom of topic pages, linking to problems tagged with related concepts.

Files: `PaperKnowledgeGraph.tsx`, new `conceptToTopicMap.ts`, topic page component

---

## Schema / Architecture Corrections

Two previously-documented plans contain incorrect assumptions:

1. **PHASE_16_BACKEND_PLAN.md** assumes `POST /api/progress/update` can be wired to topic completion. **It cannot.** The schema requires integer `paper_id` + `module_id` DB record IDs — not string topic slugs. Phase 16G must use localStorage for topic progress.

2. **PHASE_16_ANALYSIS.md** inherited the old FRONTEND_REALITY_AUDIT claim that `src/content/` doesn't exist. **It does exist.** 84 content slug directories are present and content loader is functional. The correct blockers are wrong navigation hrefs and missing CTAs, not absent content.

---

## One-Line Status Per Route

| Route | User-Accessible | Content Present | Actions Available |
|---|---|---|---|
| `/` (landing) | ✅ | ✅ | No upload CTA |
| `/dashboard` | ✅ | ✅ (hardcoded) | Wrong deep links, no upload |
| `/learn` | ✅ | ✅ | 9/9 recommendation links dead |
| `/learn/deep-learning` | ✅ | ✅ | All topics except "attention" dead |
| `/learn/deep-learning/attention` | ✅ | ✅ | Mark complete doesn't persist |
| `/learn/[other-domain]` | ✅ | Fallback only | All topic links dead |
| `/learn/[domain]/[other-topic]` | ✅ | ❌ | 404 topic not found |
| `/papers` | ✅ | ❌ | Blank stub, no upload CTA |
| `/papers/upload` | Reachable by URL only | ✅ | Upload works |
| `/papers/upload/[id]` | ✅ | ✅ | 6 tabs render, dead end after |
| `/dojo` | ✅ | ✅ | No post-solve recommendations |
| `/architectures` | ✅ | ❌ | Stub, no content list |
