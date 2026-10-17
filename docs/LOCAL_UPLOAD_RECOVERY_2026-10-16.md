# Local upload recovery — 2026-10-16

## Fixed

- Docker Desktop 4.81.0 failed on inaccessible Windows AF_UNIX runtime files in Docker/run/dockerInference and docker-secrets-engine/engine.sock.
- Preserved the affected runtime directories under timestamped .backup-* names. No images, containers, volumes or WSL disk images were deleted.
- Backed up Docker settings-store.json and disabled EnableDockerAI, the optional feature implicated in recreating the inference endpoint. Docker engine 29.6.1 subsequently responded to docker info.
- Started Redis via docker compose up -d redis; verified PONG. Started a local Celery worker with the solo pool so it can read locally stored PDF paths. Backend readiness returned HTTP 200 with ready, broker_reachable and worker_responding all true.
- Added localhost:3003 and 127.0.0.1:3003 to the local .env ALLOWED_ORIGINS setting, retaining other configured origins.

## Upload handling

Both multipart uploads and R2 upload confirmations now publish outside the API event loop, disable publish retries and unnecessary Redis result subscriptions, and use bounded Redis connection/socket timeouts. Application progress continues to use SQL Task records. Dispatch errors mark the Task failed and return a sanitized HTTP 503. Storage accounting and upload rewards only occur after publication succeeds.

The input is retained on dispatch errors because the broker may have accepted the message before a connection failed; immediately deleting it would race a worker. Automated retained-input cleanup and idempotent retries remain separate improvements.

The original unqueued EMSE test task was corrected from pending to failed; the retry uses a separate task ID.

## Validation

- 62 passed, 1 skipped: queue failures, event-loop responsiveness, Celery readiness, upload contracts, storage quotas, intent binding, worker and integration coverage.
- 36 passed: infrastructure tasks (2 existing SQLAlchemy deprecation warnings).
- Ruff: all checks passed; 261 backend/core/worker files already formatted.
- Live browser with Redis down: HTTP 503 in about 4.2 seconds and the specific processing-unavailable message, replacing the generic timeout.
- Screenshot: D:/EMSE_paper2code_test/queue-error-fixed.png.

## Local startup (CMD, separate terminals)

Start Docker Desktop first. Then:

```cmd
cd /d C:\papper2code
docker compose up -d redis
docker compose exec -T redis redis-cli ping
```

```cmd
cd /d C:\papper2code
.venv\Scripts\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

```cmd
cd /d C:\papper2code
.venv\Scripts\python.exe -m celery -A backend.celery_app worker --loglevel=info --pool=solo --concurrency=1
```

```cmd
cd /d C:\papper2code
npm run dev -- --port 3003
```

The solo pool is for this local Windows test; production worker deployment is unchanged. Docker's underlying Windows socket defect may recur after a restart. Runtime backups and settings backup allow recovery without a factory reset.

No commit or push performed.

## Real manuscript result

The repaired browser upload queued task 48ffae67-3edc-4ad3-bf37-3ef44eb65271. The actual worker received the PDF, loaded the embedding model, and called the configured Groq extraction model. The generator returned `No architecture could be detected.` The browser displayed that error. No implementation was produced. This is a generator coverage limitation for this empirical SHAP/defect-prediction paper, distinct from the repaired infrastructure failure. The generator and extraction prompts were not changed in this repair.

Evidence: D:/EMSE_paper2code_test/extraction-result.png.

Final worker outcome: all three configured attempts ended with No architecture could be detected. Groq briefly rate-limited the last attempt; it subsequently completed and returned the same extraction error. The worker remains available.
