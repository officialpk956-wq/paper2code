# Backend Confidence Report

## What would prevent Paper2Code from surviving production traffic tomorrow?
1. **[P0] Dojo RCE**: The server will be compromised within 5 minutes of hitting the internet.
2. **[P0] Synchronous AI Pipelines**: The server will crash under the load of 10 concurrent users.
3. **[P1] SQLite Database**: Database locking will cause massive 500 Internal Server Errors.
4. **[P2] Monolithic Architecture**: Codebase is unmaintainable for a scaling engineering team.

### Effort Estimates
- Fix Dojo Sandbox (Docker/gVisor): 2 weeks
- Async Task Queue (Celery/Redis): 2 weeks
- Migrate to PostgreSQL: 3 days
- Refactor `server.py` to APIRouters: 4 days
