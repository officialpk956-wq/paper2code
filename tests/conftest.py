import pytest
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from backend.server import app
from backend.database import get_db
from backend.models import Base, Problem  # ensure problems exist for tests
from backend.modules.jobs.models import BackgroundJob


TEST_DB_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session", autouse=True)
def configure_celery():
    from backend.celery_app import celery_app
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(db_engine):
    TestingSession = sessionmaker(bind=db_engine)
    session = TestingSession()
    yield session
    session.rollback()
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
