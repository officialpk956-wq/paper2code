import httpx, sys, json, time

BASE = "http://127.0.0.1:8000"
PASS = []
FAIL = []

def check(label, resp, expected_status=200, body_contains=None):
    ok = resp.status_code == expected_status
    if body_contains and ok:
        ok = body_contains in resp.text
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}: {resp.status_code}")
    if not ok:
        print(f"         body: {resp.text[:200]}")
        FAIL.append(label)
    else:
        PASS.append(label)
    return ok

print("\n=== HEALTH CHECKS ===")
with httpx.Client(base_url=BASE, timeout=10) as c:
    check("GET /api/health", c.get("/api/health"), 200, "ok")
    check("GET /api/health/db", c.get("/api/health/db"), 200)
    check("GET /api/health/redis", c.get("/api/health/redis"))  # 200 or 503 (no redis in dev)
    check("GET /api/health/piston", c.get("/api/health/piston"))  # 200 or 503 (optional)

print("\n=== AUTH FLOW ===")
TEST_EMAIL = f"verify_test_{int(time.time())}@example.com"
TEST_PASS  = "VerifyPass1!"
TOKEN = None

with httpx.Client(base_url=BASE, timeout=15) as c:
    # register
    r = c.post("/api/auth/register", json={
        "email": TEST_EMAIL, "password": TEST_PASS, "name": "Verify Bot"
    })
    check("POST /api/auth/register", r, 201)

    # login (form-urlencoded)
    r = c.post("/api/auth/login", data={"username": TEST_EMAIL, "password": TEST_PASS})
    if check("POST /api/auth/login", r, 200, "access_token"):
        TOKEN = r.json()["access_token"]

    if TOKEN:
        headers = {"Authorization": f"Bearer {TOKEN}"}

        # /me
        check("GET /api/auth/me", c.get("/api/auth/me", headers=headers), 200, "email")

        print("\n=== PAPERS ===")
        check("GET /api/papers", c.get("/api/papers", headers=headers), 200)
        check("GET /api/papers (no auth -> 401)", c.get("/api/papers"), 401)

        print("\n=== DOJO ===")
        check("GET /api/dojo/problems", c.get("/api/dojo/problems", headers=headers), 200)

        print("\n=== LEARNING ===")
        check("GET /api/learning/domains", c.get("/api/learning/domains", headers=headers))
        check("GET /api/recommendations", c.get("/api/recommendations", headers=headers))

        print("\n=== SEARCH ===")
        check("GET /api/search?q=attention", c.get("/api/search", params={"q": "attention"}, headers=headers))

        print("\n=== TUTOR ===")
        r = c.post("/api/tutor/ask", json={
            "query": "What is a transformer?",
            "context_type": "general",
            "context_data": {}
        }, headers=headers, timeout=30)
        check("POST /api/tutor/ask", r, 200, "answer")

        print("\n=== SSE STREAMING ===")
        # Task stream: should 401 without token
        r_noauth = c.get("/api/tasks/fake-id/stream")
        check("GET /api/tasks/{id}/stream (no auth -> 401)", r_noauth, 401)

        # Tutor stream: check auth + content-type header only (don't consume full stream)
        with c.stream("POST", "/api/tutor/stream",
                      json={"query": "explain relu", "context_type": "general", "context_data": {}},
                      headers=headers, timeout=15) as sr:
            ct = sr.headers.get("content-type", "")
            ok_stream = sr.status_code in (200, 429) and (
                sr.status_code == 429 or "text/event-stream" in ct
            )
            mark = "PASS" if ok_stream else "FAIL"
            print(f"  [{mark}] POST /api/tutor/stream: {sr.status_code} {ct}")
            if ok_stream:
                PASS.append("POST /api/tutor/stream")
            else:
                FAIL.append("POST /api/tutor/stream")

        print("\n=== NOTIFICATIONS ===")
        check("GET /api/notifications", c.get("/api/notifications", headers=headers))

        print("\n=== ACHIEVEMENTS ===")
        check("GET /api/achievements", c.get("/api/achievements", headers=headers))

        print("\n=== USER PROFILE ===")
        check("GET /api/me/profile", c.get("/api/me/profile", headers=headers))
        check("GET /api/me/stats",   c.get("/api/me/stats",   headers=headers))

print("\n" + "="*50)
print(f"RESULT: {len(PASS)} PASS / {len(FAIL)} FAIL")
if FAIL:
    print("FAILED:")
    for f in FAIL: print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
