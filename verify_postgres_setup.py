"""
verify_postgres_setup.py

Integration tests for PostgreSQL validation.
"""

import os
import sys

# Force PostgreSQL URL before any backend imports
os.environ["DATABASE_URL"] = "postgresql+psycopg2://postgres@localhost:5432/postgres"

from fastapi.testclient import TestClient
from backend.server import app
from backend.database import SessionLocal, engine, ping_db
from backend.services.user_service import UserService
from backend.services.problem_service import ProblemService
from backend.models import Difficulty, User, Problem
import threading
import time


def run_tests():
    print("--- 1. Database Connection ---")
    status = ping_db()
    if status["ok"]:
        print(f"[PASS] Connected successfully. Dialect: {status['dialect']}, URL: {status['url']}")
        if status['dialect'] != 'postgresql':
            print("[FAIL] Not using PostgreSQL!")
            sys.exit(1)
    else:
        print(f"[FAIL] Connection failed: {status['error']}")
        sys.exit(1)

    print("\n--- 2. CRUD and Relationship Loading ---")
    db = SessionLocal()
    try:
        user_service = UserService(db)
        problem_service = ProblemService(db)

        # Create user
        test_email = "pg_test@example.com"
        user = user_service.get_or_create_user(email=test_email, name="PG User")
        print(f"[PASS] Created/Fetched User: {user}")

        # Create problem
        test_title = "Postgres Verification Problem"
        problem = problem_service.repo.get_by_title(test_title)
        if not problem:
            problem = problem_service.create_problem(
                title=test_title,
                description="Test description.",
                category="Testing",
                difficulty=Difficulty.Medium,
                starter_code="def test(): pass",
                test_cases=[{"expression": "True", "expected": "True"}]
            )
        print(f"[PASS] Created/Fetched Problem: {problem}")
        
        # Verify relationships (should be empty initially)
        print(f"[PASS] User submissions loaded successfully: {len(user.submissions)}")
        print(f"[PASS] Problem submissions loaded successfully: {len(problem.submissions)}")

    finally:
        db.close()

    print("\n--- 3. Transaction Rollback Behavior ---")
    db = SessionLocal()
    try:
        # Intentionally cause a rollback by creating a user with the same email
        try:
            from backend.models import User
            dup_user = User(email="pg_test@example.com", name="Duplicate")
            db.add(dup_user)
            db.flush()
            print("[FAIL] Duplicate email did not raise an error!")
        except Exception as e:
            db.rollback()
            print(f"[PASS] Transaction rolled back on constraint violation: {type(e).__name__}")
            
        # Verify we can still use the session after rollback
        user_count = db.query(User).count()
        print(f"[PASS] Session recovered successfully, user count: {user_count}")
    finally:
        db.close()
        
    print("\n--- 4. Concurrent Session Behavior ---")
    
    def worker(worker_id):
        sess = SessionLocal()
        try:
            u = User(email=f"worker_{worker_id}@example.com", name=f"Worker {worker_id}")
            sess.add(u)
            sess.commit()
        except Exception as e:
            print(f"[FAIL] Worker {worker_id} failed: {e}")
        finally:
            sess.close()

    threads = []
    for i in range(5):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    db = SessionLocal()
    try:
        concurrent_users = db.query(User).filter(User.email.like("worker_%")).count()
        if concurrent_users == 5:
            print(f"[PASS] 5 concurrent sessions executed successfully.")
        else:
            print(f"[FAIL] Expected 5 concurrent users, found {concurrent_users}")
    finally:
        db.close()

    print("\n--- 5. API Health Endpoint ---")
    client = TestClient(app)
    response = client.get("/api/health/db")
    if response.status_code == 200:
        data = response.json()
        print(f"[PASS] Health endpoint ok. Response: {data}")
    else:
        print(f"[FAIL] Health endpoint returned {response.status_code}: {response.text}")

if __name__ == "__main__":
    run_tests()
