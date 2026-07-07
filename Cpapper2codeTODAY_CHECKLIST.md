# 🚀 TODAY'S ACTION ITEMS (June 28)

## CRITICAL: Complete by 5 PM UTC

These 4 items BLOCK everything else. If any are not ready, **Week 2 deployment will be delayed**.

---

## 1️⃣ DATABASE (PostgreSQL 14+) — 1–2 hours

### Pre-Check
```bash
# Verify PostgreSQL is running
psql $DATABASE_URL -c "SELECT version();"

# Expected output: PostgreSQL 14.x or higher
```

### Setup Tasks

**[ ] SSL Connections**
```bash
# Verify SSL mode
psql $DATABASE_URL -c "SELECT current_setting('ssl');"
# Expected: on

# If off, update connection string to: ?sslmode=require
```

**[ ] Automated Backups**
```bash
# Create backup now
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql
gzip backup_*.sql

# Upload to S3/backup bucket (keep for 30 days)
aws s3 cp backup_*.sql.gz s3://your-backup-bucket/

# Verify restore works
createdb test_restore_check
pg_restore --dbname=test_restore_check backup_latest.sql
psql test_restore_check -c "SELECT count(*) FROM users;"
dropdb test_restore_check
```

**[ ] Connection Pooling**
- If using Supabase: Built-in (check dashboard)
- If using raw PostgreSQL: Install PgBouncer
  ```bash
  apt-get install pgbouncer
  # Configure /etc/pgbouncer/pgbouncer.ini
  # pool_size = 25, max_client_conn = 100
  ```

**[ ] Verify Connection Works**
```bash
psql $DATABASE_URL -c "SELECT 1" && echo "✅ PostgreSQL ready"
```

---

## 2️⃣ REDIS (7+) — 1 hour

### Pre-Check
```bash
# Verify Redis is running
redis-cli -u $REDIS_URL PING
# Expected: PONG

# Check version
redis-cli -u $REDIS_URL --version
# Expected: redis-cli 7.x or higher
```

### Setup Tasks

**[ ] Password Authentication**
```bash
# Verify password is set
redis-cli -u $REDIS_URL CONFIG GET requirepass
# Expected: requirepass <password>

# If not set:
redis-cli -u $REDIS_URL CONFIG SET requirepass <strong-password>
redis-cli -u $REDIS_URL CONFIG REWRITE  # Persist to disk
```

**[ ] Persistence**
```bash
# Verify RDB snapshots enabled
redis-cli -u $REDIS_URL CONFIG GET save
# Expected: save 900 1 (snapshot every 15 min if 1+ key changed)

# If not set:
redis-cli -u $REDIS_URL CONFIG SET save "900 1"
redis-cli -u $REDIS_URL CONFIG REWRITE
```

**[ ] Max Memory Policy**
```bash
# Verify eviction policy
redis-cli -u $REDIS_URL CONFIG GET maxmemory-policy
# Expected: allkeys-lru

# If not set:
redis-cli -u $REDIS_URL CONFIG SET maxmemory-policy allkeys-lru
redis-cli -u $REDIS_URL CONFIG REWRITE
```

**[ ] Verify Connection Works**
```bash
redis-cli -u $REDIS_URL SET test-key "hello" && redis-cli -u $REDIS_URL GET test-key && echo "✅ Redis ready"
```

---

## 3️⃣ DOMAIN + SSL — 1–2 hours

### Pre-Check
```bash
# Verify domain resolves
nslookup api.paper2code.com
# Expected: points to your API IP

# Verify HTTPS works
curl -I https://api.paper2code.com
# Expected: 200 OK (or 503 if not deployed yet, but SSL should work)
```

### Setup Tasks

**[ ] SSL Certificate**
- [ ] If using Let's Encrypt:
  ```bash
  certbot certonly --standalone -d api.paper2code.com
  # Gets cert at /etc/letsencrypt/live/api.paper2code.com/
  ```

- [ ] If using paid CA: 
  - [ ] Issue certificate
  - [ ] Download .crt and .key files
  - [ ] Store in secret manager (AWS Secrets Manager, Vault, etc.)

**[ ] HSTS Header**
```bash
# In backend/server.py or nginx config:
app.add_middleware(
    "Strict-Transport-Security: max-age=31536000; includeSubDomains"
)
# This tells browsers: "Always use HTTPS for this domain, for 1 year"

# Verify:
curl -I https://api.paper2code.com | grep -i "strict-transport"
```

**[ ] Force HTTPS Redirect**
```bash
# In nginx or FastAPI middleware:
# All http:// requests → https://
# Test:
curl -I http://api.paper2code.com
# Expected: 301 Moved Permanently to https://
```

---

## 4️⃣ KUBERNETES CLUSTER (if using K8s) — 2–3 hours

### Pre-Check
```bash
# Verify cluster running
kubectl get nodes
# Expected: 3–5 nodes in Ready state

# Verify kubectl can access cluster
kubectl get pods --all-namespaces | head
```

### Setup Tasks

**[ ] Create Production Namespace**
```bash
kubectl create namespace production
kubectl config set-context --current --namespace=production
```

**[ ] Create Secrets**
```bash
kubectl create secret generic paper2code-secrets \
  --from-literal=database-url=$DATABASE_URL \
  --from-literal=redis-url=$REDIS_URL \
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY \
  --from-literal=secret-key=$(python -c "import secrets; print(secrets.token_urlsafe(32))") \
  --from-literal=r2-access-key=$R2_ACCESS_KEY \
  --from-literal=r2-secret-key=$R2_SECRET_KEY \
  -n production

# Verify:
kubectl get secrets -n production
```

**[ ] Create ConfigMap (non-secrets)**
```bash
kubectl create configmap paper2code-config \
  --from-literal=environment=production \
  --from-literal=cors-origins="https://api.paper2code.com,https://paper2code.com" \
  --from-literal=log-level=INFO \
  -n production

# Verify:
kubectl get configmap -n production
```

---

## ✅ VERIFICATION CHECKLIST

Run this script at 4 PM to verify everything is ready:

```bash
#!/bin/bash

echo "🔍 Verifying production setup..."

# 1. Database
echo "1️⃣  Database..."
psql $DATABASE_URL -c "SELECT 1" && echo "   ✅ PostgreSQL ready" || echo "   ❌ PostgreSQL failed"

# 2. Redis
echo "2️⃣  Redis..."
redis-cli -u $REDIS_URL PING | grep PONG && echo "   ✅ Redis ready" || echo "   ❌ Redis failed"

# 3. SSL
echo "3️⃣  SSL/HTTPS..."
curl -s -I https://api.paper2code.com | grep -q "200\|301\|307\|503" && echo "   ✅ HTTPS ready" || echo "   ❌ HTTPS failed"

# 4. Kubernetes
echo "4️⃣  Kubernetes..."
kubectl get nodes 2>/dev/null | grep -q "Ready" && echo "   ✅ K8s ready" || echo "   ❌ K8s failed"

# 5. Secrets
echo "5️⃣  Secrets..."
kubectl get secrets -n production 2>/dev/null | grep -q "paper2code-secrets" && echo "   ✅ Secrets ready" || echo "   ❌ Secrets failed"

echo ""
echo "✅ All systems go for deployment!" || echo "❌ Fix failures above before proceeding"
```

---

## 📋 IF YOU GET STUCK

**Database issues?**
- Check connection string: `echo $DATABASE_URL`
- Test manually: `psql $DATABASE_URL -c "SELECT version();"`
- Common: wrong password, IP whitelist not set

**Redis issues?**
- Check connection string: `echo $REDIS_URL`
- Test manually: `redis-cli -u $REDIS_URL PING`
- Common: wrong password, port closed

**SSL issues?**
- Check cert validity: `openssl x509 -enddate -in /path/to/cert.pem`
- Check domain matches: `openssl x509 -subject -in /path/to/cert.pem`
- Common: wildcard cert doesn't match subdomain, expired

**Kubernetes issues?**
- Check node status: `kubectl get nodes`
- Check available resources: `kubectl top nodes`
- Common: not enough CPU/memory, node not ready

---

## 🎯 SUCCESS: You're Done When

- ✅ PostgreSQL: `psql $DATABASE_URL -c "SELECT 1"` returns `1`
- ✅ Redis: `redis-cli -u $REDIS_URL PING` returns `PONG`
- ✅ HTTPS: `curl -I https://api.paper2code.com` returns `200` or `503` (not SSL error)
- ✅ Kubernetes: `kubectl get nodes` shows 3+ `Ready` nodes
- ✅ Secrets: `kubectl get secrets -n production` lists `paper2code-secrets`

**If all 5 are green:** You're ready for Phase 1B tomorrow morning! ✅

If any are red: Fix it now (takes < 1 hour usually), then move on.

---

## 📞 NEED HELP?

- PostgreSQL docs: https://www.postgresql.org/docs/14/
- Redis docs: https://redis.io/docs/
- Kubernetes docs: https://kubernetes.io/docs/
- Your team Slack/Discord for cloud infra help

**Good luck! You've got this.** 🚀
