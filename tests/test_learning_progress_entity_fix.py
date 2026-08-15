"""
tests/test_learning_progress_entity_fix.py

Tests verifying that LearnerProgress polymorphic entity_id resolution
works properly across /api/learning/dashboard, learning paths, adaptive tutor,
and assessment status without triggering AttributeError.
"""

import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import User, Paper, PaperModule, LearnerProgress
from backend.routers.learning import _fetch_adaptive_data as learning_fetch
from backend.routers.tutor import _fetch_adaptive_data as tutor_fetch
from backend.routers.assessment import _fetch_adaptive_data as assessment_fetch


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_learning_fetch_adaptive_data_polymorphic(db_session):
    # Setup test data
    user = User(id=1, email="test@test.com", name="Learner 1")
    paper = Paper(id=1, title="Attention Is All You Need")
    module = PaperModule(id=42, paper_id=1, layer_name="MultiHeadAttention", module_type="attention", order_index=1)
    db_session.add_all([user, paper, module])
    db_session.commit()

    # Create polymorphic progress record with entity_type and entity_id
    progress = LearnerProgress(
        learner_id="1",
        entity_type="paper_module",
        entity_id="42",
        status="in_progress",
        started_at=datetime.datetime.utcnow(),
    )
    db_session.add(progress)
    db_session.commit()

    # Execute _fetch_adaptive_data from learning router
    attempts, progress_data, tutor_data, modules_data = learning_fetch(db_session, "1")
    assert len(progress_data) == 1
    assert progress_data[0]["module_id"] == 42
    assert progress_data[0]["status"] == "in_progress"


def test_tutor_fetch_adaptive_data_polymorphic(db_session):
    progress = LearnerProgress(
        learner_id="2",
        entity_type="paper_module",
        entity_id="99",
        status="completed",
        started_at=datetime.datetime.utcnow(),
    )
    db_session.add(progress)
    db_session.commit()

    attempts, progress_data, tutor_data, modules_data = tutor_fetch(db_session, "2")
    assert len(progress_data) == 1
    assert progress_data[0]["module_id"] == 99
    assert progress_data[0]["status"] == "completed"


def test_assessment_fetch_adaptive_data_polymorphic(db_session):
    progress = LearnerProgress(
        learner_id="3",
        entity_type="paper_module",
        entity_id="101",
        status="not_started",
    )
    db_session.add(progress)
    db_session.commit()

    attempts, progress_data, tutor_data, modules_data = assessment_fetch(db_session, "3")
    assert len(progress_data) == 1
    assert progress_data[0]["module_id"] == 101
    assert progress_data[0]["status"] == "not_started"


def test_fetch_adaptive_data_ignores_non_module_entities(db_session):
    # Non-module progress (e.g. topic, problem) should not crash and be filtered
    progress_problem = LearnerProgress(
        learner_id="4",
        entity_type="problem",
        entity_id="prob_transformer_1",
        status="completed",
    )
    db_session.add(progress_problem)
    db_session.commit()

    attempts, progress_data, tutor_data, modules_data = learning_fetch(db_session, "4")
    assert len(progress_data) == 0


def test_fetch_adaptive_data_handles_invalid_entity_id_gracefully(db_session):
    # Non-integer entity_id for paper_module shouldn't crash with ValueError
    progress_corrupt = LearnerProgress(
        learner_id="5",
        entity_type="paper_module",
        entity_id="not_an_int",
        status="in_progress",
    )
    db_session.add(progress_corrupt)
    db_session.commit()

    attempts, progress_data, tutor_data, modules_data = learning_fetch(db_session, "5")
    assert len(progress_data) == 0
