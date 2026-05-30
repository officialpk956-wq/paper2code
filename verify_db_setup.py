"""
verify_db_setup.py

Integration tests to verify the database backend is correctly set up.
"""

import sys
from fastapi.testclient import TestClient
from backend.server import app
from backend.database import SessionLocal, engine, ping_db
from backend.services.user_service import UserService
from backend.services.problem_service import ProblemService
from backend.models import Difficulty

def run_tests():
    print("--- 1. Database Connection ---")
    status = ping_db()
    if status["ok"]:
        print(f"[PASS] Connected successfully. Dialect: {status['dialect']}, URL: {status['url']}")
    else:
        print(f"[FAIL] Connection failed: {status['error']}")
        sys.exit(1)

    print("\n--- 2. CRUD Example via Services ---")
    db = SessionLocal()
    try:
        user_service = UserService(db)
        problem_service = ProblemService(db)

        # Create a user
        test_email = "test@example.com"
        user = user_service.get_or_create_user(email=test_email, name="Test User")
        print(f"[PASS] Created/Fetched User: {user}")

        # Create a problem
        test_title = "Verify DB Integration Problem"
        problem = problem_service.repo.get_by_title(test_title)
        if not problem:
            problem = problem_service.create_problem(
                title=test_title,
                description="Test description.",
                category="Testing",
                difficulty=Difficulty.Easy,
                starter_code="def test(): pass",
                test_cases=[{"expression": "True", "expected": "True"}]
            )
        print(f"[PASS] Created/Fetched Problem: {problem}")

        # Award points
        updated_user = user_service.award_points(user.id, 50)
        print(f"[PASS] User Points updated: {updated_user.points}")

    finally:
        db.close()

    print("\n--- 3. API Health Endpoint ---")
    client = TestClient(app)
    response = client.get("/api/health/db")
    if response.status_code == 200:
        data = response.json()
        print(f"[PASS] Health endpoint ok. Response: {data}")
    else:
        print(f"[FAIL] Health endpoint returned {response.status_code}: {response.text}")

if __name__ == "__main__":
    run_tests()
