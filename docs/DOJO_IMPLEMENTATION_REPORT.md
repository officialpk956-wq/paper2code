# DS Coding Dojo — Implementation Report

**Date:** 2026-06-18  
**Build status:** ✅ TypeScript clean · 10/10 Python tests pass · All pages 200 OK

---

## Files Created

### Backend (Next.js API Routes)

| File | Purpose |
|------|---------|
| `src/app/api/dojo/run/route.ts` | POST — run code against visible test cases (8s timeout) |
| `src/app/api/dojo/submit/route.ts` | POST — run code against all test cases, return verdict + save |

### Frontend Pages

| File | Route | Purpose |
|------|-------|---------|
| `src/app/dojo/page.tsx` | `/dojo` | Problem list (server component) |
| `src/app/dojo/[slug]/page.tsx` | `/dojo/:slug` | Problem detail page (server component) |

### Components (`src/components/dojo/`)

| File | Purpose |
|------|---------|
| `DojoProblemList.tsx` | Filterable problem catalog with category/difficulty/search filters |
| `DojoProblemPage.tsx` | Main 2-column layout orchestrator (client component) |
| `TheoryPanel.tsx` | Left panel — Problem / Theory / Hints tabs with collapsible hint cards |
| `DojoEditor.tsx` | Monaco editor with Ctrl+Enter shortcut, Run/Submit buttons |
| `TestResultPanel.tsx` | Test case results — individual pass/fail rows with expand |
| `SubmissionHistory.tsx` | Submission history list with code viewer |

### Library (`src/lib/dojo/index.ts`)

Submission tracking + progress integration.

### Tests

| File | Coverage |
|------|---------|
| `tests/test_dojo_service.py` | 10 tests across 5 test classes |

## Files Modified

| File | Change |
|------|--------|
| `src/components/layout/left-rail.tsx` | Added Terminal icon + Dojo nav item under Learn section |
| `package.json` | Added `@monaco-editor/react` dependency |
| `.claude/launch.json` | Updated to use `C:\node.exe` directly for preview server |

---

## API Endpoints

### `POST /api/dojo/run`

Runs code against **visible test cases only** (fast feedback).

**Request body:**
```json
{
  "code": "def matrix_multiply(A, B): ...",
  "testCases": [{ "input": {"A": [...], "B": [...]}, "output": [...], "visible": true }],
  "functionName": "matrix_multiply",
  "visibleOnly": true
}
```

**Response:**
```json
{
  "results": [
    { "index": 0, "passed": true, "actual": [[19,22],[43,50]], "expected": [[19,22],[43,50]], "runtime_ms": 1.2, "error": null }
  ],
  "totalMs": 234
}
```

**Error codes:** 400 (bad input), 408 (timeout), 500 (execution error)

---

### `POST /api/dojo/submit`

Runs code against **all test cases** (visible + hidden). Returns verdict.

**Request body:** Same as `/run` without `visibleOnly`

**Response:**
```json
{
  "results": [...],
  "status": "accepted",
  "passedTests": 3,
  "totalTests": 3,
  "runtimeMs": 1.8,
  "totalMs": 412
}
```

**Status values:** `accepted` | `wrong_answer` | `error` | `timeout`

---

## Database Schema

No database dependency — submissions are stored in **localStorage** with the key `dojo:submissions` (keyed by problem slug). Schema:

```typescript
interface Submission {
  id: string;            // "sub_<timestamp>_<random>"
  slug: string;
  code: string;
  status: 'accepted' | 'wrong_answer' | 'error' | 'timeout';
  passedTests: number;
  totalTests: number;
  runtimeMs: number | null;
  submittedAt: string;   // ISO 8601
  results: TestResult[];
}
```

Accepted submissions mirror-write to the global progress engine via `markCompleted('problem', slug)` so the dashboard and learning graph reflect dojo completions.

---

## Frontend Pages

### `/dojo` — Problem List

- **Filters:** category dropdown, difficulty toggle (All / Easy / Medium / Hard), text search
- **Stats bar:** solved/total count, counts by difficulty
- **Table:** # (with ✓/○ solved/attempted indicators), title + tags, category, difficulty badge, estimated time
- **Pagination:** not yet (21 problems fit on one page)

### `/dojo/[slug]` — Problem Detail

**2-column LeetCode layout:**

| Panel | Content |
|-------|---------|
| Left (42%) | Problem / Theory / Hints / History tabs |
| Right (58%) | Monaco editor (top 60%) + Test results (bottom 40%) |

**Tabs:**
- **Problem** — description, learning objectives, topic tags, visible examples + hidden test count
- **Theory** — step-by-step explanation, complexity analysis, common mistakes, related math tags
- **Hints** — 3 progressive hint cards (collapsible `<details>`), solution outline, interview discussion
- **History** — submission history list with code viewer

**Editor features:**
- Monaco editor with Python syntax highlighting
- JetBrains Mono font, line 22 height, tab=4
- Ctrl+Enter keyboard shortcut → Run
- ▶ Run button (visible tests only, green)
- ↑ Submit button (all tests, purple)
- Spinner animation during execution

**Results panel:**
- "Test Output" tab — individual test case rows (expand for input/expected/actual)
- "Submission" tab — verdict banner with pass count + runtime, all test results

---

## Test Coverage

```
tests/test_dojo_service.py — 10/10 PASS

TestMatrixMultiplication
  ✓ test_correct_solution_passes_all
  ✓ test_wrong_solution_fails
  ✓ test_runtime_error_captured

TestSoftmax
  ✓ test_softmax_sums_to_one

TestSigmoid
  ✓ test_sigmoid

TestVisibleOnlyFilter
  ✓ test_all_cases_run_on_submit
  ✓ test_visible_only_on_run

TestSubmissionStatus
  ✓ test_accepted
  ✓ test_wrong_answer
  ✓ test_runtime_error
```

---

## Code Execution Engine

The test runner generates a Python script that:
1. Imports the user's function definition
2. Parses test cases via `json.loads()` (handles JSON `true`/`false`/`null` → Python `True`/`False`/`None`)
3. Converts list inputs to numpy arrays when numpy is available
4. Calls the function with keyword arguments matching input keys
5. Compares output via `_deep_equal` with tolerance 1e-5 for floats
6. Outputs JSON results to stdout

**Key fixes applied:**
- JSON embedding bug: `json.dumps()` produces lowercase `true`/`false` which aren't valid Python. Fixed by wrapping in `json.loads("""...""")`.
- Monaco keyboard shortcut: uses `onMount(editor, monaco)` second parameter, not `window.monaco`.

---

## Progress Integration

- `saveSubmission()` → localStorage + mirrors accepted submissions to `markCompleted('problem', slug)`
- `getProblemProgress()` → reads from localStorage, checked against `isCompleted()` from global progress engine
- Dashboard stat widgets will reflect dojo completions automatically

---

## Remaining Work

| Item | Priority | Notes |
|------|----------|-------|
| Expand to 110+ problems | High | Add NumPy, Stats, Probability, ML Metrics categories per PDF spec |
| Execution sandbox | High | Current: runs Python directly (dev only). For prod: Docker sandbox or Pyodide (WASM) |
| Pagination | Medium | Currently renders all 21 problems |
| Discussion tab | Medium | Placeholder — needs forum/comments system |
| Acceptance rate tracking | Medium | Needs aggregate stats across all users |
| Dark/light editor theme | Low | Monaco is always dark; should respect app theme |
| Rate limiting | High | API routes have no rate limiting — needed before production |
| `tests/test_dojo_api.py` | Medium | End-to-end API tests against running server |
