import pytest
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from backend.server import app
from backend.database import get_db
from backend.models import Base, Problem  # ensure problems exist for tests
from backend.modules.jobs.models import BackgroundJob


@pytest.fixture(scope="session", autouse=True)
def configure_celery():
    from backend.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True

@pytest.fixture(scope="function", autouse=True)
def reset_rate_limits(monkeypatch):
    from backend.modules.auth.middleware import rate_limit
    rate_limit._in_memory_store.clear()
    monkeypatch.setattr(rate_limit, "_redis_client", None)
    # Clear in-process learning path cache — it keys on user.id, which resets to 1
    # for every test that uses a fresh in-memory DB.
    from backend.routers.user import get_learning_path
    if hasattr(get_learning_path, "_cache"):
        get_learning_path._cache.clear()


@pytest.fixture(scope="function")
def db_engine():
    # Each test gets its own isolated in-memory SQLite database.
    # StaticPool ensures all connections within the engine share one in-memory DB.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()

@pytest.fixture(scope="function")
def db_session(db_engine):
    TestingSession = sessionmaker(bind=db_engine)
    session = TestingSession()
    yield session
    session.close()

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        yield db_session
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def auth_client(db_session):
    from backend.modules.auth.dependencies import get_current_user as auth_get_current_user
    from backend.dependencies import get_current_user as core_get_current_user
    from backend.models import User
    def override_get_db():
        yield db_session
    def override_get_current_user():
        return User(id=1, is_admin=False)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[auth_get_current_user] = override_get_current_user
    app.dependency_overrides[core_get_current_user] = override_get_current_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def admin_client(db_session):
    from backend.modules.auth.dependencies import get_current_user as auth_get_current_user
    from backend.dependencies import get_current_user as core_get_current_user
    from backend.models import User
    def override_get_db():
        yield db_session
    def override_get_current_user():
        return User(id=1, is_admin=True)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[auth_get_current_user] = override_get_current_user
    app.dependency_overrides[core_get_current_user] = override_get_current_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def seeded_db(db_session):
    """Seed minimal test data: 1 problem for dojo tests."""
    from backend.models import Problem
    # clean up previous if exists
    existing = db_session.query(Problem).filter_by(id="test-problem-id").first()
    if not existing:
        p = Problem(
            id="test-problem-id",
            slug="test-problem",
            title="Test Problem",
            difficulty="Easy",
            category="Testing",
            description="Test",
            python_template="pass",
            test_cases=[],
            hints=[],
            explanation=[]
        )
        db_session.add(p)
        db_session.commit()
    return db_session

@pytest.fixture(scope="function")
def regular_user_headers(db_session, client):
    from backend.models import User
    from backend.modules.auth.security.hashing import hash_password
    import uuid
    email = f"user_{uuid.uuid4()}@example.com"
    pwd = "TestPass1!"
    u = User(email=email, name="reg", hashed_password=hash_password(pwd), is_verified=True, is_admin=False)
    db_session.add(u)
    db_session.commit()
    resp = client.post("/api/auth/login", data={"username": email, "password": pwd})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture(scope="function")
def user_a_headers(db_session, client):
    from backend.models import User
    from backend.modules.auth.security.hashing import hash_password
    import uuid
    email = f"userA_{uuid.uuid4()}@example.com"
    pwd = "TestPassA!"
    u = User(email=email, name="userA", hashed_password=hash_password(pwd), is_verified=True, is_admin=False)
    db_session.add(u)
    db_session.commit()
    resp = client.post("/api/auth/login", data={"username": email, "password": pwd})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture(scope="function")
def user_b_submission_id(db_session):
    from backend.models import User, DojoSubmission
    from backend.modules.auth.security.hashing import hash_password
    import uuid
    email = f"userB_{uuid.uuid4()}@example.com"
    u = User(email=email, name="userB", hashed_password=hash_password("pwd"), is_verified=True)
    db_session.add(u)
    db_session.commit()
    db_session.refresh(u)
    sub = DojoSubmission(user_id=u.id, problem_id="test-problem-id", code="pass", passed=True)
    db_session.add(sub)
    db_session.commit()
    db_session.refresh(sub)
    return sub.id

def generate_expired_token():
    import jwt
    from datetime import datetime, timedelta
    from backend.modules.auth.config import SECRET_KEY, ALGORITHM
    expire = datetime.utcnow() - timedelta(hours=1)
    to_encode = {"sub": "1", "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
