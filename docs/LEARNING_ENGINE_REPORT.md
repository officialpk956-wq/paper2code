# Learning Engine Report
**Date:** 2026-06-17  
**Build status:** ✅ — `tsc --noEmit`, `next lint`, `next build` all clean  
**New modules:** 8 library files, 1 hook, 2 updated components, 1 updated page

---

## 1. Progress Engine Coverage (`src/lib/progress/`)

Tracks completion state for **9 content types**:

| Type | Totals Used |
|------|------------|
| paper | 18 |
| architecture | 24 |
| problem | 110 |
| implementation | 12 |
| roadmap | 6 |
| roadmap-node | 120 |
| system-design | 8 |
| tensor-trace | 6 |
| math | 16 |

**API surface:** `markStarted()`, `markInProgress()`, `markCompleted()`, `getProgress()`, `getCompletionRate()`, `getCompletedCount()`, `isCompleted()`, `isStarted()`, `getInProgress()`, `getAllCompleted()`, `getTotalTimeSpentMs()`, `getKnowledgeCoverage()`

**Backward compatibility:** `markCompleted('problem', slug)` mirrors writes to the legacy `p2c:completed-problems` key so existing code still works.

---

## 2. Recommendation Engine Coverage (`src/lib/recommendations/`)

Generates 5 categories of recommendations:

| Category | Signal Used |
|----------|------------|
| `continue_learning` | `getInProgress()` — actively started content |
| `ready_now` | Completed papers/architectures unlock related content |
| `skill_gap` | `getMissingPrerequisites()` — prerequisites not met |
| `advanced_challenge` | Mastery sum ≥ 6 on related content |
| `suggested_revision` | Low mastery (< 2) on completed content |

**Functions:** `recommendProblems()`, `recommendPapers()`, `recommendArchitectures()`, `recommendRoadmap()`, `recommendTrack()`, `getRecommendations()` (aggregate feed, limit 8)

---

## 3. Graph Connectivity Report (`src/lib/learning-graph/`)

Graph built from 3 sources:
- **EVOLUTION_NODES** (22 paper/architecture nodes, parent→child directed edges)
- **ROADMAPS** (all 6 roadmaps × their phase nodes, prerequisites)
- **PROBLEMS** (110 nodes, soft edges to relatedPapers + relatedArchitectures)

**Functions:** `getPrerequisites()`, `getNextTopics()`, `getMissingPrerequisites()`, `isUnlocked()`, `getUnlockedContent()`, `getLearningPath()`, `getGraphStats()`

**LearningPath:** BFS backward from target collects all prerequisite nodes in topological order, returning the complete path with `completedIds` and `remainingMinutes`.

**Unlock rule:** A node is unlocked when all its `prerequisites[]` entries return `isCompleted() === true`. Manual override: marking any node complete in the progress engine immediately unlocks its successors.

---

## 4. Analytics Report (`src/lib/analytics/`)

All metrics computed from real tracked data — zero hardcoded values:

| Metric | Source |
|--------|--------|
| Total Study Hours | `getTotalStudyMs()` from session history |
| Today / Week / Month | `getDailyStudy()` aggregates |
| Streak | Consecutive days with `totalMs > 0` |
| Completion Rates | `getCompletionRate()` per type |
| Mastery Average | `getAllMasteryLevels()` mean |
| Weekly Velocity | `getAllCompleted()` filtered by last 7 days |
| Graph Coverage | `getGraphStats()` |

**Heatmap data:** `getHeatmapData(30)` returns 30 days × `{date, totalMs, level: 0–4}`, bucketed relative to the user's personal max.

**Velocity trend:** `getLearningVelocityTrend(4)` returns 4 weekly completion counts.

---

## 5. Search Improvements (`src/lib/search/engine.ts`)

Extended `SearchResult` with optional fields:
```
whyThisResult?: string
difficulty?: string
estimatedMinutes?: number
completionState?: 'completed' | 'in_progress' | 'not_started'
prerequisites?: string[]
```

New `searchWithContext(query, context)`:
- Boosts in-progress content (+4)
- Deprioritises well-mastered content (−2) except low-mastery completions (+1 for revision)
- Applies tag-overlap mastery bonus (up to +3)
- Adds `whyThisResult` explanation per result

`buildSearchContext()` reads from `progress` + `persistence` at call time (SSR-safe).

---

## 6. Roadmap Intelligence Report (`src/lib/roadmap-intelligence/`)

Converts static roadmap nodes to **adaptive state**:

| Adaptive State | Condition |
|----------------|-----------|
| `mastered` | Completed + mastery ≥ 3 |
| `completed` | Completed (mastery 2+) |
| `needs_revision` | Completed but mastery < 2 |
| `in_progress` | Static state = in_progress |
| `available` | Prerequisites satisfied, not yet done |
| `locked` | Prerequisites not satisfied |

Per roadmap: `totalProgress`, `estimatedCompletionDays` (based on remaining minutes ÷ 60min/day assumption), `nextNode` (first available/in-progress), `suggestedRevisions` list.

**Phase F — Calibration system:** Problem calibration engine (`src/lib/problem-calibration/`) scores all 110 problems into Easy/Medium/Hard/Expert tiers, computes per-user `readinessScore` (0–100) based on prerequisite completion + mastery, and surfaces `getNextChallenge()` calibrated to current completion percentile.

---

## 7. User Journey Examples

### New user, zero completions
- Dashboard shows no recommendations → CTA to `/problems`
- "Up Next" falls back to Multi-Head Attention (beginner-friendly hardcoded fallback)
- Roadmap shows all phases as `available`/`locked` per prerequisite graph

### User completes 5 beginner problems
- `getNextChallenge()` returns first `medium`-tier problem with highest readinessScore
- `recommendProblems()` surfaces related intermediate problems
- `getUnlockedContent()` expands based on prerequisites now met

### User completes "attention-is-all-you-need" paper
- All BERT and GPT nodes unlock in the learning graph
- `recommendProblems()` boosts problems with `relatedPapers: ['attention-is-all-you-need']`
- Knowledge graph shows the BERT/GPT successor nodes as `unlocked` (bright ring)

### User studies 3 consecutive days
- `getStudyStreak()` returns 3
- Dashboard right panel shows "3 days 🔥"
- `getLearningVelocityTrend()` shows increasing completion count in current week

---

## 8. Production Readiness Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| TypeScript coverage | ✅ 100% | Strict mode, zero `any` |
| Build | ✅ Pass | Static + SSG output |
| Lint | ✅ Pass | Zero warnings |
| SSR safety | ✅ | All localStorage reads behind `isClient()` |
| Backward compat | ✅ | Old `p2c:completed-problems` key mirrored |
| No mocked data | ✅ | All dashboard metrics from real engines |
| Test coverage | ⚠️ | Unit tests not yet written |
| Session accuracy | ⚠️ | Active time = total elapsed (not idle-deducted) |

**Overall: Production-ready for initial deployment. Recommend adding unit tests for the progress + recommendation engines before high-traffic use.**

---

## Files Created / Modified

| File | Phase | Action |
|------|-------|--------|
| `src/lib/progress/index.ts` | A | Created |
| `src/lib/study-sessions/index.ts` | E | Created |
| `src/lib/learning-graph/index.ts` | B | Created |
| `src/lib/recommendations/index.ts` | C | Created |
| `src/lib/roadmap-intelligence/index.ts` | F | Created |
| `src/lib/problem-calibration/index.ts` | G | Created |
| `src/lib/analytics/index.ts` | I | Created |
| `src/hooks/use-study-session.ts` | E | Created |
| `src/lib/search/engine.ts` | H | Extended |
| `src/components/knowledge/knowledge-graph.tsx` | D | Rewritten with real data |
| `src/app/dashboard/page.tsx` | All | All fake metrics replaced |
