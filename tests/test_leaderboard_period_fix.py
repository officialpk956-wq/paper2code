"""
tests/test_leaderboard_period_fix.py

Tests verifying that leaderboard queries rank by User.weekly_points for weekly periods
and User.points for alltime periods, deterministic tie-breaking is enforced,
and inactive/0-point users are appropriately filtered.
"""

import datetime
import pytest
from unittest.mock import patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import User
from backend.routers.leaderboard import get_leaderboard


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


def test_weekly_leaderboard_ranks_by_weekly_points(db_session):
    with patch("backend.redis_config.cache_redis", None):
        now = datetime.datetime.utcnow()
        # User A: High lifetime points, 0 weekly points, active today
        user_a = User(id=1, email="ua@test.com", name="User A", points=10000, weekly_points=0, last_active=now, streak=1)
        # User B: 500 lifetime, 500 weekly points, active today
        user_b = User(id=2, email="ub@test.com", name="User B", points=500, weekly_points=500, last_active=now, streak=5)
        # User C: 1000 lifetime, 300 weekly points, active today
        user_c = User(id=3, email="uc@test.com", name="User C", points=1000, weekly_points=300, last_active=now, streak=3)

        db_session.add_all([user_a, user_b, user_c])
        db_session.commit()

        # Query weekly leaderboard
        res = get_leaderboard(period="weekly", limit=10, db=db_session)
        leaders = res["leaders"]

        # User B (500 weekly points) should be #1, User C (300 weekly points) #2
        assert len(leaders) == 2
        assert leaders[0]["user_id"] == 2
        assert leaders[0]["points"] == 500
        assert leaders[1]["user_id"] == 3
        assert leaders[1]["points"] == 300

        # User A (0 weekly points) should NOT appear
        assert all(l["user_id"] != 1 for l in leaders)


def test_alltime_leaderboard_ranks_by_lifetime_points(db_session):
    with patch("backend.redis_config.cache_redis", None):
        now = datetime.datetime.utcnow()
        user_a = User(id=1, email="ua@test.com", name="User A", points=10000, weekly_points=0, last_active=now, streak=1)
        user_b = User(id=2, email="ub@test.com", name="User B", points=500, weekly_points=500, last_active=now, streak=5)
        db_session.add_all([user_a, user_b])
        db_session.commit()

        res = get_leaderboard(period="all", limit=10, db=db_session)
        leaders = res["leaders"]

        assert len(leaders) == 2
        assert leaders[0]["user_id"] == 1
        assert leaders[0]["points"] == 10000
        assert leaders[1]["user_id"] == 2
        assert leaders[1]["points"] == 500


def test_deterministic_tie_breaking(db_session):
    with patch("backend.redis_config.cache_redis", None):
        now = datetime.datetime.utcnow()
        # Tied users with same score
        user_1 = User(id=1, email="u1@test.com", name="Alpha", points=100, weekly_points=100, last_active=now, streak=1)
        user_2 = User(id=2, email="u2@test.com", name="Beta", points=100, weekly_points=100, last_active=now, streak=1)
        db_session.add_all([user_2, user_1])
        db_session.commit()

        res1 = get_leaderboard(period="weekly", limit=10, db=db_session)
        res2 = get_leaderboard(period="weekly", limit=10, db=db_session)

        # Tie-breaking by User.id asc means user 1 is always ahead of user 2
        assert res1["leaders"][0]["user_id"] == 1
        assert res1["leaders"][1]["user_id"] == 2
        assert res2["leaders"][0]["user_id"] == 1
        assert res2["leaders"][1]["user_id"] == 2
