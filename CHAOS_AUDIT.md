# Chaos Audit
**Agent 10 — Chaos Engineer**

## Chaos Test Designs
1. **The Fork Bomb**: Submit `os.fork()` recursively to the Dojo. Result: Server OS crashes within 2 seconds.
2. **The Infinite Payload**: Upload a 5GB text file padded with zeroes to the paper ingestion endpoint. Result: FastAPI process OOMs instantly due to missing request size limits.
3. **The Race Condition**: Send 50 concurrent `POST /api/progress` requests for the same user and problem. Result: SQLite database locking failures and potential data corruption.
4. **The Network Drop**: Sever the connection to OpenAI mid-generation. Result: Unhandled exceptions block the thread pool.
