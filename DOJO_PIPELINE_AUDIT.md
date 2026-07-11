# Dojo Execution Pipeline Audit Report

## 1. Migration from Piston to E2B (Sandbox Execution)

### Why We Switched
Piston requires container-level privileges to write into `/sys/fs/cgroup` for its isolated CPU/memory sandbox limits (using `isolate`). Render's managed runtime containers (on all tiers, including paid ones) block privileged container operations, causing the self-hosted Piston sandbox service to fail to start in production. 

To overcome this infrastructure blocker, the Dojo execution pipeline has been rewired to **E2B (e2b.dev)**, a hosted sandbox service designed specifically for secure code execution that does not require hosting privileged sandboxing containers on Render.

### Piston Clean-up Decision
Piston has been entirely removed from the local `docker-compose.yml` environment, and its associated `PISTON_URL` environment variables have been cleaned up. Standardizing on E2B for both local development and production ensures environment parity and prevents "works locally, breaks in prod" bugs.

---

## 2. Hardened Sandbox Pass/Fail Logic

The previous Piston implementation relied on fragile string-matching inside `stderr` (e.g., checking if `"AssertionError" not in stderr` or `"Error" not in stderr`), which incorrectly failed valid code containing words like "Error" in print statements (e.g. `print("No Errors found")`).

This has been replaced by a robust, exit-code-based check:
1. **Redirection & Filesystem Execution**: The sandbox writes the combined user code and test harness to `/home/user/solution.py` and the test input to `/home/user/stdin.txt`.
2. **PTY Execution**: The code is run via `sandbox.commands.run("bash -c 'python3 /home/user/solution.py < /home/user/stdin.txt'")`.
3. **Exit Code Validation**: The execution passes if and only if the process returns `exit_code == 0` (indicating all assertions completed successfully without throwing any unhandled exceptions).
4. **Exception Handling**: Catching E2B SDK's `CommandExitException` ensures that assertion failures return `passed = False` and populate the `exit_code` and traceback details without raising unhandled exceptions in the backend.

### Local Verification Results
The new E2B execution logic was verified locally with a throwaway test script:
- **Correct Solution**: Returned `passed: True` and captured correct stdout.
- **Wrong Solution (AssertionError)**: Returned `passed: False` and captured traceback error cleanly.
- **Printed 'Error' String**: Returned `passed: True` (fixing the old fragile string-matching bug).

---

## 3. Production Deployment Guide (Render)

Because we lack direct access to your Render dashboard / API credentials, the following steps must be completed manually:

### Step A: Set environment variables
On your **main backend service** (`paper2code-1-81y5`) on Render, add the following environment variable:
- `E2B_API_KEY` = `<your-e2b-api-key>` (obtain a free key from [e2b.dev](https://e2b.dev))

This will trigger a redeploy of the main API.

### Step B: Verify main API connectivity
Once redeployed, run the following curl command:
```bash
curl -i https://paper2code-1-81y5.onrender.com/api/health/e2b
```
This must return HTTP 200:
`{"status":"ok","e2b":"connected"}`

### Step C: Confirm Celery Worker Deployment
Ensure a separate background worker service is running on Render pointing to your production Redis instance (`REDIS_URL`) and database (`DATABASE_URL`). 
- **Start Command**: `celery -A backend.celery_app.celery_app worker --loglevel=info`
- **Why this is critical**: Graded submissions are queued asynchronously. Without this worker running, all `Submit` requests will hang in `pending` status.

---

## 4. Per-Problem Correctness & Validation Table
*Pending Render deployment of the `E2B_API_KEY` and verification of the Celery worker. Once the production endpoints are confirmed live, we will execute the browser audit and document the pass/fail grading results for all 49 problems here.*
