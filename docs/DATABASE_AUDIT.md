# Database Audit
**Agent 4 — Database Architect**

## Schema & Query Review
- **[P0] Concurrency Limits**: The system uses SQLite (`tensortonic_dev.db`). SQLite uses database-level write locks. At 1,000 concurrent users attempting to update `LearnerProgress`, transactions will queue, timeout, and fail with `OperationalError: database is locked`.
- **[P1] Missing Indexes**: Foreign key queries and filtering on `entity_type` + `entity_id` in `LearnerProgress` lack compound indexes, guaranteeing full table scans as the dataset grows.
- **[P2] Transaction Management**: Database commits are performed at the end of long-running operations. If an operation fails midway, partial writes or unhandled rollbacks may occur.

## Fix Recommendations
- Migrate immediately to PostgreSQL.
- Add SQLAlchemy composite indexes to `LearnerProgress(learner_id, entity_type, entity_id)`.
