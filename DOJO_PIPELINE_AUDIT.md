# Dojo Execution Pipeline Audit Report

## 1. Piston Sandbox Service Deployment Guide (Render)

Because we lack direct access to your Render dashboard / API credentials, we cannot deploy the Piston service for you. Below are the exact, step-by-step deployment paths to get the Piston execution engine up and running with **NumPy** persistence.

### Option A: Custom Dockerfile (Recommended — Zero-maintenance)
This option bakes Python 3.10 and NumPy directly into the image at build time, meaning it will survive Render container restarts and scale-downs without needing persistent volumes.

1. Create a file named `piston.Dockerfile` in your repository root with the following content:
   ```dockerfile
   FROM ghcr.io/engineer-man/piston:latest
   
   # Build/install Node dependencies for the ppman CLI
   RUN cd /piston/cli && npm install
   
   # Download and install Python 3.10 package using ppman
   RUN node /piston/cli/index.js ppman install python=3.10.0
   
   # Install numpy directly into the Piston Python package folder
   RUN /piston/packages/python/3.10.0/bin/pip3 install numpy
   ```
2. **Deploy on Render**:
   - Go to [Render Dashboard](https://dashboard.render.com) -> **New +** -> **Web Service**.
   - Connect your GitHub repository.
   - Set the **Docker Path** or **Dockerfile** configuration to `piston.Dockerfile`.
   - Set the environment variables:
     - `PISTON_LIMIT_MEMORY` = `67108864` (64 MB per job)
     - `PISTON_LIMIT_MAX_PROCESS_COUNT` = `32`
   - Render will automatically expose port `2000`.

---

### Option B: Registry Image + Persistent Disk (Alternate Path)
If you deploy directly from the public registry without a custom Dockerfile, the container filesystem is ephemeral, and any package installation (like NumPy) will be wiped on container restart. You must use a Persistent Disk:

1. **Deploy Web Service**:
   - Go to **New +** -> **Web Service** -> **Deploy an existing image from a registry**.
   - Registry Image URL: `ghcr.io/engineer-man/piston`
   - Set environment variables:
     - `PISTON_LIMIT_MEMORY` = `67108864`
     - `PISTON_LIMIT_MAX_PROCESS_COUNT` = `32`
2. **Add a Persistent Disk**:
   - In your newly created Piston service settings on Render, navigate to **Disks**.
   - Click **Add Disk**:
     - **Name**: `piston-packages`
     - **Mount Path**: `/piston/packages`
     - **Size**: `1 GB` (minimum required for runtimes)
3. **Install Python & NumPy**:
   - Once the service is running, send a request to Piston to install Python:
     ```bash
     curl -X POST https://<your-piston-service-url>/api/v2/packages \
       -H "Content-Type: application/json" \
       -d '{"language":"python","version":"3.10.0"}'
     ```
   - Connect to the running container shell (via Render Shell tab or SSH) and run:
     ```bash
     /piston/packages/python/3.10.0/bin/pip3 install numpy
     ```
     Because `/piston/packages` is mounted on the persistent disk, NumPy will persist.

---

### 2. Main API & Routing Integration
Once Piston is live:
1. Update the Environment Variables on your **main backend service** (`paper2code-1-81y5`):
   - Set `PISTON_URL` = `https://<your-piston-service-url>`
2. Saving the env vars will trigger a redeploy of the main API.
3. Verify connectivity:
   ```bash
   curl -i https://paper2code-1-81y5.onrender.com/api/health/piston
   ```
   This must now return HTTP 200 instead of the baseline HTTP 503 connection refused.

---

## 3. Celery Worker (Critical Infrastructure Gap)

*   **Audit Finding**: A Celery worker must be running in production for code-submissions (`Submit` button) to grade.
*   **Action Required**:
    - Ensure a separate Background Worker is deployed on Render pointing to your production Redis instance (`REDIS_URL`).
    - The start command for this service must be:
      ```bash
      celery -A backend.celery_app.celery_app worker --loglevel=info
      ```
    - Without this background worker active, all `Submit` requests will register task IDs but hang forever in `pending` status.

---

## 4. Per-Problem Correctness & Validation Table
*Pending Piston deployment by the user. Once the Piston service is live and `PISTON_URL` is configured, we can proceed to run-through and verify grading for all 49 problems.*
