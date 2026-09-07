# Local Development Guide: Paper2Code

This document describes how to run and test the complete Paper2Code stack locally, including the asynchronous Celery processing pipeline, Redis message broker, FastAPI backend, and Next.js frontend.

---

## 1. Prerequisites

- **Python 3.11+ / Python 3.14**: Active virtual environment with dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```
- **Node.js 18+ / pnpm or npm**: For running the Next.js frontend.
- **Docker Desktop / Docker Engine**: For running PostgreSQL and Redis services.

---

## 2. Infrastructure Setup (Docker Compose)

Start the local backing services (PostgreSQL and Redis) with health checks:

```bash
docker compose up -d postgres redis
```

- **PostgreSQL 16**: Port `5432` (`p2c:p2cdev@localhost:5432/p2c`)
- **Redis 7**: Port `6379` (`redis://localhost:6379/0`) with health check ping.

To verify services are running healthy:
```bash
docker compose ps
```

---

## 3. Running the Stack Locally

### Terminal 1: Celery Worker (Async Task Execution)
On Windows, use the `-P solo` pool to avoid subprocess fork issues:

```bash
celery -A backend.celery_app:celery_app worker --loglevel=info -P solo
```

On Linux / macOS:
```bash
celery -A backend.celery_app:celery_app worker --loglevel=info --concurrency=2
```

### Terminal 2: FastAPI Backend Server
Start the Uvicorn development server:

```bash
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation and Swagger UI will be available at: `http://localhost:8000/docs`

### Terminal 3: Next.js Frontend
Start the Next.js development server:

```bash
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## 4. Testing the Real Async Processing Path (Non-Eager Mode)

When `CELERY_TASK_ALWAYS_EAGER` is set to `False` (default production mode), paper uploads execute asynchronously through the Celery worker via Redis:

1. **Submit Paper via HTTP POST**:
   ```bash
   curl -X POST http://localhost:8000/api/papers/upload \
     -H "Authorization: Bearer <JWT_TOKEN>" \
     -F "file=@sample_paper.pdf" \
     -F "terms_accepted=true" \
     -F "visibility=private"
   ```
   **Response** (`202 Accepted`):
   ```json
   {
    "status": "pending",
    "paper_id": null,
    "task_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "poll_url": "/api/tasks/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "message": "PDF uploaded. Poll poll_url every 1s for the generated code."
   }
   ```

2. **Poll Status Endpoint**:
   ```bash
   curl -X GET http://localhost:8000/api/tasks/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d \
     -H "Authorization: Bearer <JWT_TOKEN>"
   ```
   Task status transitions from `pending` to `running`, then `completed` or
   `failed`. Generation quality (`success` or `needs_review`) is returned in
   `result.generation_status` when the task completes.

3. **Retrieve Generated PyTorch Code & Verification Report**:
   ```bash
   curl -X GET http://localhost:8000/api/papers/1 \
     -H "Authorization: Bearer <JWT_TOKEN>"
   ```

For an actual browser/UI verification (rather than curl), run the stack above
and then opt into the gated Playwright test:

```powershell
$env:RUN_REAL_ASYNC_PAPER_E2E='1'
npx playwright test e2e/paper-upload-async.spec.ts --project=chromium
```

The test logs in through the real API, uploads through the Papers UI, observes
the processing dialog, follows task polling, and requires the workspace
redirect. It is skipped by default so ordinary frontend test runs do not call
LLM or sandbox services.

---

## 5. Running the Test Suite

Always invoke pytest via the active Python module runner:

```bash
# Run all Phase 2 and regression tests:
python -m pytest tests/test_phase2_*.py tests/test_phase1_paper_codegen.py -v

# Run the complete test suite:
python -m pytest tests/ -q
```
