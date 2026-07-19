# Production Deployment Execution Log

**Start Date:** 2026-06-28  
**Target Go-Live:** 2026-07-12 (Week 2, Monday)  
**Status:** IN PROGRESS

---

## WEEK 1: Pre-Deployment (June 28 – July 4)

### ✅ PHASE 1A: Infrastructure Setup (TODAY – June 28)

**Priority: CRITICAL** – Blocks everything else

- [ ] **Database (PostgreSQL 14+)**
  - [ ] Verify instance is running
  - [ ] Enable SSL connections
  - [ ] Create automated backups (daily, 30-day retention)
  - [ ] Set max connections = 100
  - [ ] Create connection pool (PgBouncer or Supabase)
  - [ ] Test connection: `psql $DATABASE_URL -c "SELECT 1"`
  - **Owner:** You | **Deadline:** Today 5 PM
  - **Time Est:** 1–2 hours

- [ ] **Redis (7+)**
  - [ ] Verify instance is running
  - [ ] Set password authentication
  - [ ] Enable persistence (RDB snapshots)
  - [ ] Set max memory policy to `allkeys-lru`
  - [ ] Test connection: `redis-cli -u $REDIS_URL PING`
  - **Owner:** You | **Deadline:** Today 5 PM
  - **Time Est:** 1 hour

- [ ] **Domain + SSL**
  - [ ] Verify domain resolves: `nslookup api.paper2code.com`
  - [ ] Issue SSL certificate (Let's Encrypt or paid CA)
  - [ ] Test HTTPS: `curl -I https://api.paper2code.com`
  - [ ] Set HSTS header (max-age=31536000)
  - **Owner:** You | **Deadline:** Today 5 PM
  - **Time Est:** 1–2 hours

- [ ] **Kubernetes Cluster (if using K8s)**
  - [ ] Cluster running (3–5 nodes)
  - [ ] `kubectl get nodes` shows all healthy
  - [ ] Create namespace: `kubectl create namespace production`
  - [ ] Create secrets: `kubectl create secret generic paper2code-secrets ...`
  - **Owner:** You | **Deadline:** Today 6 PM
  - **Time Est:** 2–3 hours

---

### ✅ PHASE 1B: Code & Config (June 28–29, Morning)

- [ ] **Environment Variables**
  - [ ] List all required vars in `.env.production.example`
  - [ ] Verify none are hardcoded in code
  - [ ] Required: DATABASE_URL, REDIS_URL, ANTHROPIC_API_KEY, SECRET_KEY, R2_BUCKET, CORS_ORIGINS
  - [ ] Test locally: `python -c "import os; print(os.getenv('ANTHROPIC_API_KEY'))"`
  - **Owner:** You | **Deadline:** June 29, 10 AM
  - **Time Est:** 1 hour

- [ ] **Generate Production Secrets**
  - [ ] New SECRET_KEY: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
  - [ ] Rotate ANTHROPIC_API_KEY if needed
  - [ ] Rotate OPENID_PROVIDER_CLIENT_SECRET if needed
  - [ ] Store in secure vault (AWS Secrets Manager, 1Password, Vault)
  - **Owner:** You | **Deadline:** June 29, 11 AM
  - **Time Est:** 1 hour

- [ ] **Security Headers**
  - [ ] Verify `SecurityHeadersMiddleware` is enabled in `backend/server.py`
  - [ ] Test headers: `curl -I https://api.paper2code.com | grep -i "X-Content-Type"`
  - [ ] Expected: X-Content-Type-Options, X-Frame-Options, Referrer-Policy, CSP
  - **Owner:** You | **Deadline:** June 29, 12 PM
  - **Time Est:** 30 min

- [ ] **Sentry Setup**
  - [ ] Create Sentry project
  - [ ] Get DSN: `https://...@sentry.io/...`
  - [ ] Set in production config: `SENTRY_DSN=<dsn>`
  - [ ] Test: `python -c "from sentry_sdk import init; init('<dsn>')"`
  - **Owner:** You | **Deadline:** June 29, 1 PM
  - **Time Est:** 30 min

---

### ✅ PHASE 1C: Database Backup & Migration Test (June 29–30)

- [ ] **Create Full Backup**
  ```bash
  pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
  gzip backup_*.sql
  # Upload to S3 / backup bucket
  # Verify restore works on test instance
  ```
  - **Owner:** You | **Deadline:** June 29, 3 PM
  - **Time Est:** 1 hour

- [ ] **Test Migration Dry-Run**
  ```bash
  # On staging/clone of production
  createdb paper2code_test_clone
  pg_restore --dbname=paper2code_test_clone backup_latest.sql
  alembic upgrade head
  # Verify: new columns exist, no errors
  ```
  - **Owner:** You | **Deadline:** June 30, 10 AM
  - **Time Est:** 1–2 hours

- [ ] **Create Rollback Plan**
  ```bash
  # Document rollback steps
  # Verify backup restores in < 5 minutes
  # Test on staging: pg_restore backup_latest.sql
  ```
  - **Owner:** You | **Deadline:** June 30, 11 AM
  - **Time Est:** 1 hour

---

### ✅ PHASE 1D: Tests & Load Testing (July 1–2)

- [ ] **Unit Tests Still Passing**
  ```bash
  cd /papper2code
  .venv\Scripts\python -m pytest tests\ -x -q
  # Expected: 1358 passing
  ```
  - **Owner:** You | **Deadline:** July 1, 9 AM
  - **Time Est:** 5 min (or 2 hours if failures)

- [ ] **Build Docker Image**
  ```bash
  docker build -t paper2code:latest .
  docker run -p 8000:8000 paper2code:latest &
  sleep 5
  curl http://localhost:8000/health
  # Expected: 200 OK
  ```
  - **Owner:** You | **Deadline:** July 1, 10 AM
  - **Time Est:** 30 min

- [ ] **Baseline Load Test (10 VUs, 15 min)**
  ```bash
  k6 run --vus 10 --duration 15m tests/load/baseline.js
  # Monitor: p95 < 100ms, errors < 1%
  ```
  - **Owner:** You | **Deadline:** July 1, 2 PM
  - **Time Est:** 20 min (+ analysis 30 min)
  - **Expected Result:** ✅ Pass (p95 < 100ms)

- [ ] **Stress Test (ramp to 100 VUs, 30 min)**
  ```bash
  k6 run --stage 10m:100 --stage 15m:100 --stage 5m:0 tests/load/stress.js
  # Monitor: no 5xx errors, recovery < 30s
  ```
  - **Owner:** You | **Deadline:** July 2, 10 AM
  - **Time Est:** 40 min (+ analysis 30 min)
  - **Expected Result:** ✅ Pass (no 5xx, p99 < 1s)

- [ ] **Spike Test (spike to 200 VUs, 10 min)**
  ```bash
  k6 run tests/load/spike.js
  # Monitor: queue drains, recovery < 30s
  ```
  - **Owner:** You | **Deadline:** July 2, 2 PM
  - **Time Est:** 15 min (+ analysis 20 min)
  - **Expected Result:** ✅ Pass (recovers < 30s)

---

### ✅ PHASE 1E: Monitoring Setup (July 3)

- [ ] **Prometheus Scrape Config**
  - [ ] Create `prometheus.yml`
  - [ ] Add targets: API, PostgreSQL, Redis
  - [ ] Test: `curl http://prometheus:9090/api/v1/targets`
  - **Owner:** You | **Deadline:** July 3, 9 AM
  - **Time Est:** 30 min

- [ ] **Alert Rules**
  - [ ] Create `alerts.yml` with 8 rules (see PRODUCTION_DEPLOYMENT_PLAN.md)
  - [ ] Test each rule: `amtool alert`
  - **Owner:** You | **Deadline:** July 3, 10 AM
  - **Time Est:** 1 hour

- [ ] **Grafana Dashboards**
  - [ ] Create 4 dashboards: Overview, API, Infrastructure, Business
  - [ ] Import from templates or build from scratch
  - [ ] Verify data flowing in
  - **Owner:** You | **Deadline:** July 3, 12 PM
  - **Time Est:** 2 hours

- [ ] **On-Call Runbook**
  - [ ] Review `RUNBOOK.md`
  - [ ] Customize team names, Slack channels, escalation
  - [ ] Share with team
  - **Owner:** You | **Deadline:** July 3, 3 PM
  - **Time Est:** 1 hour

---

### ✅ PHASE 1F: Pre-Flight Check (July 4)

- [ ] **Final Checklist**
  - [ ] All tests passing ✅
  - [ ] Load tests passed ✅
  - [ ] Database backup working ✅
  - [ ] SSL certificate valid ✅
  - [ ] Monitoring dashboards ready ✅
  - [ ] Team briefed ✅
  - [ ] Rollback plan documented ✅

- [ ] **Team Sync**
  - [ ] Meeting with deployment team
  - [ ] Review runbook together
  - [ ] Assign on-call rotation (Week 1–2)
  - [ ] Confirm deployment time (Monday 8 AM UTC)
  - **Owner:** You | **Deadline:** July 4, 3 PM
  - **Time Est:** 1 hour

---

## WEEK 2: Deployment (July 5–12)

### 🚀 MONDAY, JULY 8: GO LIVE

**Timeline (UTC+0):**

| Time | Task | Owner | Status |
|---|---|---|---|
| 8:00 AM | Kick-off meeting | You | ⏳ |
| 8:15 AM | Final staging smoke tests | You | ⏳ |
| 8:30 AM | Brief team on runbook | You | ⏳ |
| 9:00 AM | Deploy Green environment | You | ⏳ |
| 9:10 AM | Wait for 3 pods ready | You | ⏳ |
| 9:15 AM | Smoke tests on Green | You | ⏳ |
| 9:30 AM | Canary switch (10% traffic) | You | ⏳ |
| 10:00 AM | Monitor error rate (30 min) | You | ⏳ |
| 10:30 AM | If healthy: 100% switch | You | ⏳ |
| 11:00 AM | Continuous monitoring (1 hour) | You | ⏳ |
| 12:00 PM | Team huddle (check in) | You | ⏳ |
| 3:00 PM | Success announcement | You | ⏳ |

**Deployment Commands:**
```bash
# 9:00 AM: Deploy Green
kubectl apply -f k8s_deployment_green.yaml

# 9:15 AM: Smoke tests
bash tests/smoke_tests.sh https://green.api.paper2code.com

# 9:30 AM: Canary (10%)
kubectl patch service api -p '{"spec":{"selector":{"version":"green"}}}'

# 10:00 AM: Monitor
kubectl logs deployment/api-green -f
curl https://api.paper2code.com/api/health

# 10:30 AM: Full switch (100%)
kubectl patch service api -p '{"spec":{"selector":{"version":"green"}}}'

# 11:00 AM: Watch metrics
open https://grafana.paper2code.com/dashboard/overview
```

---

### 📊 TUESDAY–FRIDAY: STABILIZATION

- [ ] **Daily 9 AM Check-In**
  - [ ] Review error logs
  - [ ] Check p95 latency
  - [ ] Verify uptime > 99.5%
  - [ ] Note bugs for fixing
  - **Owner:** You | **Daily 9 AM**

- [ ] **Bug Fixes (as needed)**
  - [ ] Prioritize by severity
  - [ ] Deploy fixes (same blue-green pattern)
  - [ ] Re-test after each fix
  - **Owner:** You | **As needed**

- [ ] **Optimization (if needed)**
  - [ ] Slow query log analysis
  - [ ] Add missing indices
  - [ ] Re-run load tests if added indices
  - **Owner:** You | **If p95 > 500ms**

- [ ] **Friday: Decommission Blue**
  - [ ] Keep backup for 1 week (in case of emergency)
  - [ ] Scale Blue down: `kubectl scale deployment api-blue --replicas=0`
  - [ ] Announce final success to team
  - **Owner:** You | **Friday 5 PM**

---

### ✅ SUCCESS CRITERIA (End of Week 2)

Check these by Friday 5 PM:

- [ ] **Zero unhandled 500 errors** (first 48 hours)
- [ ] **p95 latency < 500ms** (all endpoints)
- [ ] **Uptime > 99.5%** (max 3 min downtime)
- [ ] **All agents working:**
  - [ ] Code review agent (async task)
  - [ ] Agentic tutor (responds to queries)
  - [ ] Learning path (generates curriculum)
  - [ ] Paper ingestion (converts papers)
  - [ ] Research RAG (answers about papers)
- [ ] **First users signed up and using system**
- [ ] **Team comfortable with runbook**
- [ ] **Monitoring dashboards populated**

---

## EMERGENCY PROCEDURES

### 🚨 If Deployment Fails

**Within 30 minutes of deployment:**

```bash
# IMMEDIATE: Switch back to Blue
kubectl patch service api -p '{"spec":{"selector":{"version":"blue"}}}'

# Check Blue is still running
kubectl get pods -l version=blue

# If Blue is down, scale it up
kubectl scale deployment api-blue --replicas=3

# Monitor recovery
kubectl logs deployment/api-blue -f

# Expected recovery time: 2–3 minutes
```

**After rollback:**
1. Investigate what went wrong (check Green logs)
2. Fix the issue
3. Re-test on staging
4. Deploy again (next day, not immediately)

---

## NOTES & LESSONS LEARNED

(To be filled in during deployment)

### What Went Well ✅
- 

### What Could Be Better 🔧
- 

### Bugs Fixed 🐛
- 

### Optimizations Applied ⚡
- 

---

## Sign-Off

- [ ] **Deployment Complete** (by July 12, 5 PM)
- [ ] **All success criteria met**
- [ ] **Team trained on runbook**
- [ ] **On-call rotation assigned**
- [ ] **Next phase planned** (Frontend Integration)

**Signed:** ________________  
**Date:** ________________
