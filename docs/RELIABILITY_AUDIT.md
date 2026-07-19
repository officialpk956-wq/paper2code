# Reliability Audit
**Agent 6 — Reliability Engineer**

## Fault Tolerance & Resilience
- **What happens if LLM APIs fail?** The paper parsing pipeline crashes with an unhandled exception. The user's upload process is permanently stalled with no resume capability.
- **What happens if DB fails?** The app crashes on startup. No circuit breakers or retry policies are implemented using libraries like `tenacity`.
- **State Recovery**: Background processes do not exist. If a deployment occurs while a user's paper is being parsed, that parsing job is killed instantly and lost forever.

## Risk Classification
- **[P0] In-Memory State**: `server.py`. Long-running tasks rely on the HTTP request lifecycle. Fix: Implement Celery/Temporal for durable execution. Effort: 2 weeks.
