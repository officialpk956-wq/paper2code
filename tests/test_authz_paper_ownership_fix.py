"""
tests/test_authz_paper_ownership_fix.py

Tests verifying that the authorization engine correctly identifies paper ownership
via the 'uploaded_by' column on the 'papers' table, preventing IDOR vulnerabilities
and false access denials.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import User, Paper
import backend.modules.authz.models  # Registers authz tables on Base metadata
from backend.modules.authz.engine import authorize


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


def test_owner_can_access_own_paper(db_session):
    user = User(id=1, email="owner@test.com", name="Owner", is_admin=False)
    paper = Paper(id=100, title="Transformers", uploaded_by=1)
    db_session.add_all([user, paper])
    db_session.commit()

    has_perm = authorize(db_session, user, action="delete", resource_type="paper", resource_id=100)
    assert has_perm is True


def test_non_owner_cannot_delete_paper(db_session):
    owner = User(id=1, email="owner@test.com", name="Owner", is_admin=False)
    attacker = User(id=2, email="attacker@test.com", name="Attacker", is_admin=False)
    paper = Paper(id=101, title="BERT", uploaded_by=1)
    db_session.add_all([owner, attacker, paper])
    db_session.commit()

    has_perm = authorize(db_session, attacker, action="delete", resource_type="paper", resource_id=101)
    assert has_perm is False


def test_admin_can_delete_any_paper(db_session):
    admin = User(id=3, email="admin@test.com", name="Admin", is_admin=True)
    paper = Paper(id=102, title="GPT-2", uploaded_by=99)
    db_session.add_all([admin, paper])
    db_session.commit()

    has_perm = authorize(db_session, admin, action="delete", resource_type="paper", resource_id=102)
    assert has_perm is True


def test_non_existent_paper_returns_false(db_session):
    user = User(id=4, email="user@test.com", name="User", is_admin=False)
    db_session.add(user)
    db_session.commit()

    has_perm = authorize(db_session, user, action="delete", resource_type="paper", resource_id=99999)
    assert has_perm is False
