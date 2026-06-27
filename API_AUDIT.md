# API Audit

## Overview
The API layer is currently a monolithic 1800+ line `server.py` file.

### Critical Findings
- **God Object Anti-Pattern**: All routes are crammed into `server.py` instead of using FastAPI `APIRouter`.
- **Blocking the Event Loop**: AI pipelines and `subprocess.run()` calls are executing within synchronous route handlers. In a high-traffic scenario, a single user running an AI extraction will block all other users.
- **Missing Pagination**: Endpoints like `GET /api/problems` return `.all()`. At 1,000,000 users or large datasets, this will cause Out-Of-Memory (OOM) crashes.
- **Lack of Rate Limiting**: No rate limiters exist on expensive endpoints (e.g., Dojo submit, Auth login).
