# Scalability Audit

### Can this system scale?
**No.** It will break immediately under sustained load.

### What breaks first?
1. The SQLite Database (Write locks).
2. The FastAPI Event Loop (Blocked by synchronous AI and `subprocess` tasks).
3. The LLM API Quotas (No queuing or backpressure handling).

### What must become asynchronous?
- Paper Upload & Parsing
- Knowledge Graph Extraction
- Architecture Reconstruction

### What needs queues?
- Everything in the `core/` pipeline. Must adopt Celery, Temporal, or RabbitMQ.
