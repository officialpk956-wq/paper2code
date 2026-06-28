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
