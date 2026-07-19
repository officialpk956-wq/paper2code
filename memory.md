# paper2code — Prompt & Results Log

A running log of every prompt handed to Antigravity (or executed directly), what it did, and the verified outcome — with date. Started 2026-07-11. Newest entries at the bottom.

Format per entry:
```
## [date] — <short title>
**Prompt given:** <one-line summary of what was asked>
**What happened:** <what Antigravity/the agent actually did>
**Verified result:** <what was independently checked — pass/fail, real output, not just "it says it worked">
**Follow-up needed:** <anything still open>
```

---

## Context as of 2026-07-11 (recap before logging started)

- Dojo execution pipeline was fully broken in production — Piston was never actually deployable on Render (its `isolate` sandboxing needs privileged `/sys/fs/cgroup` access that Render's managed containers don't grant, confirmed via a real failed deploy attempt: `mkdir: cannot create directory 'isolate/': Read-only file system`).
- Decision made: migrate dojo code execution from Piston to E2B (hosted sandbox SaaS).
- Antigravity rewired `backend/services/e2b_service.py` + `backend/services/dojo_execution_service.py` (exit-code-based pass/fail replacing fragile stderr string-matching, stdin support, per-problem timeout).
- Verified directly against the live E2B API (not just Antigravity's mocked tests): found and fixed a real bug — `e2b_service.py` was forcing E2B's bare `"base"` template, which has no numpy, breaking every one of the 49 dojo problems (they all `import numpy as np`). Fixed by removing the forced default so it uses E2B's own Code Interpreter default template (has numpy/pandas/scipy preinstalled) — verified live, all 4 correctness cases pass (correct+harness, wrong solution rejected, "Error"-in-output false-positive fixed, stdin piping works).
- Also fixed: `.ruff.toml` had invalid pyproject.toml-style syntax (ruff never actually ran, ever); a real health-check bug where a Redis failure could mask a genuine DB outage as "degraded" (200) instead of "unhealthy" (503); a live `NameError` bug in `Scheduler`/`ScheduledTask` (`Optional` used without import); a mutable-default-arg footgun in `lab.py`; `pytest` was never declared in `requirements.txt` so CI had no test runner at all; a JWT secret mismatch where dojo rate-limiting and Sentry user-context verified tokens against the wrong secret (worked locally by coincidence, broke in CI).
- CI (`TensorTonic CI`: lint-and-format, backend-tests, frontend-build, docker-test) is fully green for the first time since the workflow was created.
- Discovered the codebase has 3 separate, uncoordinated Redis env var surfaces (`REDIS_CACHE_URL`/`REDIS_SESSIONS_URL`/`REDIS_RATE_LIMIT_URL` in `redis_config.py` vs plain `REDIS_URL` in `rate_limit.py` and `celery_app.py`) — `REDIS_URL` was never set on Render, so Celery's broker was silently pointing at unreachable `localhost:6379`.

---

## 2026-07-11 — Set E2B_API_KEY and fix REDIS_URL on Render

**Prompt given:** Step-by-step instructions to (1) add `E2B_API_KEY` to the main backend service on Render, (2) diagnose and fix the missing `REDIS_URL` (copy value from the already-working `REDIS_CACHE_URL`), matching what Celery's broker and the rate-limit middleware actually read.

**What happened:** User added `E2B_API_KEY` on `paper2code-1` (triggered a redeploy — confirmed via deploy log, backend seeded 49 dojo problems, started clean). User then added `REDIS_URL` on `paper2code-1` by copying the `REDIS_CACHE_URL` value.

**Verified result:**
```
GET /api/health      → {"status":"healthy","checks":{"database":"healthy","redis":"healthy"}}  HTTP 200
GET /api/health/e2b  → {"status":"ok","e2b":"connected"}                                         HTTP 200
```
Both checked directly via curl against the live backend, not assumed.

**Follow-up needed:** Confirm whether the Celery Background Worker service has actually been created on Render yet (user's "yes i have done it" was ambiguous — may only refer to the Redis fix, not the worker). Once confirmed live, hand off to Antigravity for Phase 4: full Chrome-based Run/Submit sweep across all 49 dojo problems.

## 2026-07-15 — Wire paper ingestion pipeline + ground the AI Tutor (KAG)

**Prompt given:** Two rounds. (1) Wire the real, fully-built `paper_ingestion_service.py` pipeline into the live upload task, replacing the dead thin `run_ingestion()` path that left every non-flagship paper's Knowledge Graph/Blueprint/Executable tabs 404ing. (2) Five small, surgical prompts to ground the AI Tutor in real deterministic facts instead of pure SQL substring matching, to prevent hallucinated architecture claims — grounded in `core/rag/knowledge_graph.py`'s existing `KnowledgeGraph` ontology, with exact file:line diffs specified for each.

**What happened:** Antigravity landed both. Ingestion pipeline: `paper_tasks.py` now calls the real pipeline; added a confidence badge on the Blueprint tab for low-confidence architecture matches. Tutor grounding: added `get_architecture_facts` tool with a genuinely well-reasoned edge-reconstruction (mapped primitive node IDs from `PaperModule.graph_nodes` to module IDs, then resolved `Paper.architecture_graph`'s top-level edges through that mapping) — correctly solved an ambiguity I'd explicitly flagged as unverified rather than guessing wrong. Also caught and fixed a real pre-existing bug on its own initiative: `backend/routers/learning.py` and `assessment.py` each had a byte-identical stale duplicate of `_get_tutor_callbacks` that would have kept the old unhardened behavior even after `tutor.py` was fixed — replaced both with an import from the single source.

**Verified result:**
- 1 of 5 prompts (system-prompt hedge instruction) silently did not land — caught by diffing the actual file against what was specified, not by trusting the "done" report. Fixed directly.
- Synthetic test (not mocked) of the new `get_architecture_facts` reconstruction logic against a fake conv2d→linear paper: correctly surfaced the real `REQUIRES_FLATTEN` rule from `KnowledgeGraph.verify_topology()` — the exact adversarial case the whole feature was built to catch.
- Found and fixed a lint failure (`tutor.py` import sorting) that would have broken `lint-and-format` CI, same pattern as prior E2B commits — none of Antigravity's changes are being run through ruff before landing; worth telling it to do so going forward.
- Removed a stray `=2.0.0` debris file (leftover from an unquoted `pip install x>=2.0.0` in a shell that treated `>` as redirection).
- Full 1488-test suite: 0 failed, 1 skipped (up from 1482 baseline — new tests in `test_agentic_tutor.py` + new `test_paper_tasks.py`).
- Both changes committed as `a5d764e1` (ingestion pipeline) and `44188730` (Tutor grounding).

**Follow-up needed:**
- Neither change has been verified live in Chrome yet (real upload → real workspace tabs; real Tutor conversation with a real LLM call). Unit/synthetic verification is strong but not the same as end-to-end proof, per this project's established standard.
- Phase B of the RAG+KAG plan (Qdrant Cloud signup, widen the vector index to full text/code/explanations, wire cross-paper semantic search into the Tutor) is still queued, not started.
- Push these two commits to `main` when ready — currently local only.
