# Paper2Code Production Deployment Plan

**Timeline:** Weeks 1–2 (10–15 hours total)  
**Target Go-Live:** End of week 2  
**Rollback Plan:** Blue-green deployment, 30-minute rollback window

---

## Pre-Deployment Checklist (Week 1)

### ✅ Code & Configuration

- [ ] **All 1358 tests passing** on main branch
  ```bash
  cd /papper2code && .venv/Scripts/python -m pytest tests/ -x -q
  ```
  Current status: 1358 passing ✅

- [ ] **No unmerged branches** with code
  ```bash
  git branch -v | grep -v "main"
  ```
  Confirm only main exists or all other branches are fully merged

- [ ] **Environment variables documented**
  - [ ] Create `.env.production.example` with all required vars
  - [ ] Verify all sensitive vars use `os.getenv()` (no hardcodes)
  - [ ] List required: `DATABASE_URL`, `REDIS_URL`, `ANTHROPIC_API_KEY`, `SECRET_KEY`, `R2_BUCKET`, `CORS_ORIGINS`

- [ ] **Database migrations prepared**
  - [ ] Run pending Alembic migrations locally
    ```bash
    alembic upgrade head
    ```
  - [ ] Backup current database
  - [ ] Verify rollback path (downgrade script)

- [ ] **Static secrets rotated**
  - [ ] Generate new `SECRET_KEY` (production-strength)
  - [ ] Rotate `OPENID_PROVIDER_CLIENT_SECRET`
  - [ ] Store in secure vault (AWS Secrets Manager, Vault, 1Password)

- [ ] **Logging configured**
  - [ ] Verify structured JSON logging enabled
  - [ ] Sentry DSN set in production config
  - [ ] Log level set to `INFO` (not `DEBUG`)

---

### ✅ Infrastructure Setup

- [ ] **Database**
  - [ ] PostgreSQL 14+ instance running
  - [ ] SSL connection required (not optional)
  - [ ] Automated backups enabled (daily, 30-day retention)
  - [ ] Connection pooling configured (PgBouncer or Supabase built-in)
  - [ ] Max connections: 100 (adjust based on load test)

- [ ] **Redis**
  - [ ] Redis 7+ instance running
  - [ ] Password authentication required
  - [ ] Persistence enabled (RDB snapshots)
  - [ ] Max memory policy: `allkeys-lru`
  - [ ] Monitoring: memory usage, connected clients, ops/sec

- [ ] **Celery Broker**
  - [ ] 2–4 worker processes (start with 2)
  - [ ] Task TTL: 24 hours
  - [ ] Result backend: Redis (with 1-hour expiry)
  - [ ] Dead letter queue monitoring enabled

- [ ] **Storage (R2 or S3)**
  - [ ] Bucket created with private ACL
  - [ ] CORS configured (allow uploads from your domain)
  - [ ] Lifecycle policy: auto-delete uploads >30 days old
  - [ ] Versioning disabled (saves space)

- [ ] **API Key & CORS**
  - [ ] Anthropic API key injected as secret (not in code)
  - [ ] CORS origins set to your domain only (not *)
  - [ ] Rate limiting enabled (100 req/min per IP)

---

### ✅ Security Hardening

- [ ] **SSL/TLS**
  - [ ] Certificate issued (Let's Encrypt or paid CA)
  - [ ] Enabled on all endpoints (force HTTPS redirect)
  - [ ] HSTS header set: `Strict-Transport-Security: max-age=31536000`

- [ ] **Headers**
  - [ ] `Content-Security-Policy` configured (see `SecurityHeadersMiddleware`)
  - [ ] `X-Content-Type-Options: nosniff`
  - [ ] `X-Frame-Options: SAMEORIGIN`
  - [ ] `Referrer-Policy: strict-origin-when-cross-origin`

- [ ] **Database**
  - [ ] All users have minimal required permissions
  - [ ] SQL injection tests passed (parameterized queries verified)
  - [ ] Backups encrypted at rest

- [ ] **Secrets**
  - [ ] No `.env` file in git
  - [ ] `.gitignore` includes `*.env`, `*.pem`, `*.key`
  - [ ] All API keys rotated recently
  - [ ] No test credentials in production config

---

## Load Testing (Week 1, Mid)

### Test Setup

**Tool:** k6 (Grafana's load testing platform)

**Environment:** Staging (identical to production)

**Scenarios:**

#### Scenario 1: Baseline Load (15 minutes)
```
- 10 concurrent users
- Ramp-up: 1 minute
- Sustained: 10 minutes
- Ramp-down: 4 minutes
- Metrics: p50, p95, p99 latency, error rate
```

**Expected Results:**
```
GET /api/problems:        p95 < 100ms
GET /api/papers:          p95 < 200ms
POST /api/dojo/submit:    p95 < 500ms
POST /api/tutor/ask:      p95 < 2000ms (agent latency)
```

#### Scenario 2: Stress Test (30 minutes)
```
- Ramp up to 100 users over 10 minutes
- Hold 100 users for 15 minutes
- Ramp down over 5 minutes
- Identify breaking point
```

**Expected Results:**
```
No 5xx errors
p99 latency < 2 seconds
Success rate > 99.5%
```

#### Scenario 3: Spike Test (10 minutes)
```
- 50 users baseline
- Spike to 200 users for 2 minutes
- Back to 50 users
- Verify recovery time
```

**Expected Results:**
```
Recovery time < 30 seconds
No cascading failures
Queue depth normalizes
```

### Test Execution

**Week 1 (Tuesday):**
```bash
# Install k6
brew install k6  # or download from k6.io

# Run baseline test
k6 run --vus 10 --duration 15m tests/load/baseline.js

# Analyze results
# p95 latency should be < targets above
```

**Week 1 (Wednesday):**
```bash
# Run stress test
k6 run --vus 100 --duration 30m tests/load/stress.js

# If breaking point < 100 users:
#   → Add more DB connections
#   → Scale Celery workers
#   → Enable Redis caching on expensive endpoints
```

**Week 1 (Thursday):**
```bash
# Run spike test
k6 run tests/load/spike.js

# Verify autoscaling works (if using K8s)
# Monitor CPU, memory, network
```

### k6 Test Files to Create

**`tests/load/baseline.js`:**
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  duration: '15m',
  thresholds: {
    http_req_duration: ['p(95)<100', 'p(99)<200'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  // Mix of endpoints
  let r1 = http.get('https://api.paper2code.com/api/problems');
  check(r1, { 'problems: status 200': (r) => r.status === 200 });
  
  let r2 = http.get('https://api.paper2code.com/api/papers');
  check(r2, { 'papers: status 200': (r) => r.status === 200 });
  
  sleep(1);
}
```

**`tests/load/stress.js`:**
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '10m', target: 100 },
    { duration: '15m', target: 100 },
    { duration: '5m', target: 0 },
  ],
  thresholds: {
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  // Heavier requests (dojo submit, tutor ask)
  let token = 'test-jwt-token'; // Use test user token
  
  let r = http.post('https://api.paper2code.com/api/dojo/submit', 
    JSON.stringify({ code: 'print("hello")', problem_id: 'prob-001' }),
    { headers: { 'Authorization': `Bearer ${token}` } }
  );
  check(r, { 'submit: status 2xx': (r) => r.status >= 200 && r.status < 300 });
  
  sleep(Math.random() * 3);
}
```

---

## Database Migration Strategy (Week 1, Late)

### Pre-Migration

**Backup:**
```bash
# Full backup of production database
pg_dump $DATABASE_URL > backup_$(date +%s).sql
gzip backup_*.sql
# Upload to S3 / Backup bucket
```

**Dry Run:**
```bash
# Test migration on a clone of production
createdb paper2code_test_clone
pg_restore --dbname=paper2code_test_clone backup_latest.sql
alembic upgrade head  # Test the upgrade
# Verify schema matches expected
```

### Migration Steps

**Week 1 (Friday evening, off-peak: 11 PM UTC):**

1. **Enable read-only mode** (prevents writes during migration)
   ```python
   # Update server.py: all POST/PATCH endpoints return 503
   # Or use database user with SELECT-only permissions
   ```

2. **Run migrations**
   ```bash
   alembic upgrade head
   # Wait for migrations to complete (should be < 2 minutes for H–K changes)
   ```

3. **Verify schema**
   ```bash
   psql $DATABASE_URL -c "SELECT * FROM information_schema.tables WHERE table_schema='public';"
   # Confirm new columns exist (email_digest, updated_at on AssessmentAttempt)
   ```

4. **Disable read-only mode** (allow writes again)
   ```python
   # Restore normal permissions
   ```

5. **Smoke tests**
   ```bash
   curl -X GET https://api.paper2code.com/api/auth/me -H "Authorization: Bearer $TEST_TOKEN"
   # Verify 200 OK
   ```

### Rollback Plan

If migration fails:
```bash
# Restore from backup
pg_restore --dbname=$DATABASE_URL backup_latest.sql
# Redeploy previous code version
git checkout main~1
./deploy.sh
```

---

## CI/CD Setup (Week 1, Parallel)

### GitHub Actions Workflow

**File: `.github/workflows/deploy-prod.yml`**

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:  # Manual trigger

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest tests/ -x -q
        env:
          DATABASE_URL: sqlite:///test.db
          REDIS_URL: redis://localhost:6379

  security-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run bandit (security linter)
        run: |
          pip install bandit
          bandit -r backend/ -ll
      - name: Check for secrets
        run: |
          pip install detect-secrets
          detect-secrets scan --all-files --force-use-all-plugins

  build:
    needs: [test, security-check]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t paper2code:${{ github.sha }} .
      - name: Push to registry
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push paper2code:${{ github.sha }}
          docker tag paper2code:${{ github.sha }} paper2code:latest
          docker push paper2code:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: production
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Kubernetes
        run: |
          curl -X POST https://your-deployment-webhook.com/deploy \
            -H "Authorization: Bearer ${{ secrets.DEPLOY_TOKEN }}" \
            -d '{"image": "paper2code:${{ github.sha }}"}'
      - name: Verify deployment
        run: |
          sleep 30  # Wait for pods to start
          curl -f https://api.paper2code.com/api/health || exit 1
      - name: Smoke tests
        run: |
          bash tests/smoke_tests.sh
      - name: Slack notification
        if: success()
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -H 'Content-Type: application/json' \
            -d '{"text": "✅ Production deployment successful"}'
```

---

## Deployment Strategy: Blue-Green

### Architecture

```
Load Balancer
    ├─ Blue (v1.0, current production)
    │   ├─ 3× API pods
    │   ├─ PostgreSQL
    │   └─ Redis
    │
    └─ Green (v1.1, new code)
        ├─ 3× API pods
        ├─ PostgreSQL (shared)
        └─ Redis (shared)

Traffic initially 100% → Blue
Deployment: Deploy Green, run smoke tests
If Green healthy: Traffic 100% → Green
If Green unhealthy: Keep 100% → Blue (rollback)
```

### Deployment Steps

**Week 2 (Monday morning):**

1. **Deploy Green environment**
   ```bash
   kubectl apply -f k8s/green-deployment.yaml
   # Waits for 3 pods to be ready
   ```

2. **Run smoke tests against Green**
   ```bash
   tests/smoke_tests.sh https://green.api.paper2code.com
   # Verify all critical endpoints working
   ```

3. **Switch traffic 10% → Green** (canary)
   ```bash
   kubectl patch service api -p '{"spec":{"selector":{"version":"green"}}}'
   # Monitor error rates for 5 minutes
   ```

4. **If errors < 0.1%, switch 100% → Green**
   ```bash
   kubectl patch service api -p '{"spec":{"selector":{"version":"green"}}}'
   ```

5. **Decommission Blue** (keep as rollback for 24 hours)
   ```bash
   kubectl scale deployment api-blue --replicas=0
   ```

### Rollback Procedure (if needed)

**Within 30 minutes of deployment:**

```bash
# Switch traffic back to Blue
kubectl patch service api -p '{"spec":{"selector":{"version":"blue"}}}'

# If Blue still running, traffic restored immediately
# If Blue down, scale it back up (2–3 minute recovery)
kubectl scale deployment api-blue --replicas=3

# Investigate Green logs
kubectl logs deployment/api-green --tail=1000 | grep ERROR
```

---

## Monitoring & Alerting Setup (Week 2, Early)

### Prometheus Scrape Config

**`prometheus.yml`:**
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'paper2code-api'
    static_configs:
      - targets: ['api.paper2code.com:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres.paper2code.com:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis.paper2code.com:6379']
```

### Key Metrics to Monitor

**Application Metrics:**
- `http_requests_total` (by status code, endpoint)
- `http_request_duration_seconds` (p50, p95, p99)
- `http_requests_failed_total` (5xx errors)

**Database:**
- `postgresql_connections_active`
- `postgresql_query_duration_seconds`
- `postgresql_database_size_bytes`

**Celery:**
- `celery_tasks_total` (by status: success, failure)
- `celery_task_duration_seconds` (by task type)
- `celery_queue_length` (by queue name)

**Redis:**
- `redis_memory_used_bytes`
- `redis_connected_clients`
- `redis_keyspace_hits_total` / `redis_keyspace_misses_total`

### Alert Rules

**`alerts.yml`:**
```yaml
groups:
  - name: paper2code
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_failed_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Error rate > 5% for 5 minutes"
          severity: critical

      - alert: HighLatency
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 1
        for: 10m
        annotations:
          summary: "p95 latency > 1s for 10 minutes"
          severity: warning

      - alert: CeleryBacklog
        expr: celery_queue_length > 1000
        for: 5m
        annotations:
          summary: "Celery queue depth > 1000"
          severity: warning

      - alert: DatabaseConnections
        expr: postgresql_connections_active > 80
        for: 5m
        annotations:
          summary: "DB connections > 80/100"
          severity: warning

      - alert: RedisMemory
        expr: redis_memory_used_bytes / 1073741824 > 4  # 4 GB
        for: 5m
        annotations:
          summary: "Redis memory > 4 GB"
          severity: warning
```

### Grafana Dashboards

**Create these dashboards:**

1. **Overview (red flags)**
   - Error rate (5-minute)
   - p95 latency (5-minute)
   - Active users (last hour)
   - Celery queue depth

2. **API Performance**
   - Requests per second (by endpoint)
   - Latency distribution (p50, p95, p99)
   - Error rate by endpoint
   - Top slow endpoints

3. **Infrastructure**
   - CPU, memory, disk usage
   - Network I/O
   - Database connections
   - Redis memory usage

4. **Business Metrics**
   - Code submissions per hour
   - Papers uploaded per day
   - Tutor queries per hour
   - Unique active users

---

## Documentation (Week 2, Early)

### API Documentation

**Endpoint:** `https://api.paper2code.com/docs` (Swagger UI)

**Verify:**
- [ ] All endpoints documented (use FastAPI docstrings)
- [ ] Request/response schemas shown
- [ ] Authentication method documented (JWT)
- [ ] Rate limits documented (100 req/min)
- [ ] Error codes documented (401, 403, 404, 500)

**Update:**
```python
# In backend/server.py
app = FastAPI(
    title="Paper2Code API",
    version="1.0.0",
    description="Convert research papers to executable PyTorch code",
    contact={
        "name": "Paper2Code Support",
        "email": "support@paper2code.com",
    },
)
```

### Runbook

**Create `RUNBOOK.md`:**

```markdown
# Paper2Code Production Runbook

## Incident Response

### Error Rate Spike
1. Check `/metrics` endpoint for spike source
2. If API errors: check logs, restart pods if OOM
3. If Celery errors: check queue depth, scale workers
4. If Database errors: check connections, run VACUUM

### High Latency
1. Check slow query log: `SELECT * FROM pg_stat_statements`
2. Add index if needed
3. Check Redis memory usage
4. Check Celery queue depth

### Celery Queue Backlog
1. Scale workers: `kubectl scale deployment celery-worker --replicas=8`
2. Check for stuck tasks: `celery -A backend.celery_app inspect active`
3. If stuck, revoke: `celery -A backend.celery_app revoke <task_id>`

### Database Connection Pool Exhausted
1. Check active connections: `SELECT count(*) FROM pg_stat_activity`
2. Kill idle connections: `SELECT pg_terminate_backend(pid) FROM ...`
3. Restart API pods (clears connection pool)
4. Increase pool size in `DATABASE_URL` if recurring

## Rollback Procedure

```bash
# If deployed in last 30 minutes
git checkout main~1
./deploy.sh

# Monitor for errors
curl https://api.paper2code.com/api/health
```

## Useful Commands

### Check API Status
```bash
curl https://api.paper2code.com/api/health
# Should return 200 OK
```

### View Recent Errors
```bash
# From Sentry dashboard or:
curl https://sentry.io/api/0/projects/org/project/events/ \
  -H "Authorization: Bearer $SENTRY_TOKEN"
```

### Scale Workers
```bash
kubectl scale deployment celery-worker --replicas=N
```

### Run Database Maintenance
```bash
# Connect to prod database
psql $DATABASE_URL

# Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# Vacuum (compress)
VACUUM ANALYZE;

# Check slow queries
SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
```
```

### Deployment Checklist

**Create `DEPLOYMENT_CHECKLIST.md`:**

```markdown
# Deployment Checklist

## Pre-Deployment (2 hours before)

- [ ] All tests passing: `pytest tests/ -x -q`
- [ ] Code review approved
- [ ] Database backup created
- [ ] Dry-run migration tested on staging
- [ ] Monitoring dashboards ready
- [ ] Runbook reviewed
- [ ] Team notified in Slack

## Deployment (execution)

- [ ] Green environment deployed
- [ ] Smoke tests pass against Green
- [ ] Canary traffic switched (10%)
- [ ] Monitor errors for 5 minutes
- [ ] If no errors, full traffic switch (100%)
- [ ] Verify metrics normal
- [ ] Update status page (if have one)

## Post-Deployment (1 hour after)

- [ ] All endpoints responding 200
- [ ] Error rate < 0.1%
- [ ] p95 latency < 1 second
- [ ] No critical alerts firing
- [ ] Database connections normal
- [ ] Celery queue depth normal
- [ ] Monitor for 30 minutes
- [ ] Post success message to Slack

## Rollback Triggers

If any of these occur within 30 minutes:

- [ ] Error rate > 1%
- [ ] p95 latency > 2 seconds
- [ ] API pod restarts > 2
- [ ] Database connection pool exhausted
- [ ] Critical alert firing

**Rollback:** See Runbook section

---

## Week 2 Timeline

### Monday
- [ ] 8 AM: Kick-off meeting (confirm all infra ready)
- [ ] 10 AM: Final smoke tests on staging
- [ ] 12 PM: Deploy Green environment
- [ ] 1 PM: Run smoke tests against Green
- [ ] 2 PM: Canary switch (10% traffic)
- [ ] 2:30 PM: Monitor & assess errors
- [ ] 3 PM: Full switch (100% traffic) if healthy
- [ ] 4–5 PM: Monitor closely

### Tuesday–Friday
- [ ] Monitor metrics daily (check overnight logs)
- [ ] Fix any bugs found by users
- [ ] Iterate on performance (index missing queries, etc.)
- [ ] Decommission Blue on Friday (keep 1 week for safety)

---

## Success Criteria

By end of Week 2, you should have:

✅ Zero unhandled 500 errors in first 48 hours  
✅ p95 latency < 500ms across all endpoints  
✅ Uptime > 99.5% (no more than 3 minutes downtime)  
✅ All agents working (code review, tutor, learning path)  
✅ Monitoring dashboards populated with real data  
✅ Team comfortable with runbook procedures  
✅ First paying users signed up (optional, but ideal)  

---

## Go/No-Go Criteria

**Go** if:
- All tests pass
- Load tests show system handles 100 concurrent users
- Security audit found no CRITICAL issues
- Database migration tested successfully
- Rollback plan documented and tested

**No-Go** if:
- Load tests fail (p95 > 2s or errors > 0.5%)
- Security audit finds CRITICAL issue
- Database migration takes > 5 minutes
- No monitoring in place

---

## Post-Launch (Week 3+)

- Monitor for 1 week before declaring "stable"
- Collect user feedback on new features
- Plan Phase 2 (Frontend Integration)
- Begin planning Phase 3 (More Architectures)
