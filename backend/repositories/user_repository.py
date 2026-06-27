"""
backend/repositories/user_repository.py

Data-access layer for the User model.

All methods receive an active SQLAlchemy Session and return ORM objects
or None.  No business logic lives here — only raw CRUD.
"""

from __future__ import annotations

import datetime
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import User


class UserRepository:
    """CRUD operations for User rows."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Return User by primary key, or None."""
        return self._db.get(User, user_id)

    def get_by_email(self, email: str) -> Optional[User]:
        """Return User by unique email address, or None."""
        stmt = select(User).where(User.email == email)
        return self._db.execute(stmt).scalar_one_or_none()

    def list_all(self, *, limit: int = 100, offset: int = 0) -> Sequence[User]:
        """Return a paginated slice of all users ordered by points descending."""
        stmt = (
            select(User)
            .order_by(User.points.desc())
            .limit(limit)
            .offset(offset)
        )
        return self._db.execute(stmt).scalars().all()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        email: str,
        name: str,
        hashed_password: Optional[str] = None,
        avatar_url: Optional[str] = None,
    ) -> User:
        """
        Insert a new User row.

        Raises sqlalchemy.exc.IntegrityError if email is not unique
        (let the caller handle / rollback).
        """
        user = User(
            email=email,
            name=name,
            hashed_password=hashed_password,
            avatar_url=avatar_url,
            points=0,
            streak=0,
        )
        self._db.add(user)
        self._db.flush()   # assign id without committing
        return user

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def add_points(self, user_id: int, points: int) -> Optional[User]:
        """Increment a user's point total.  Returns updated User or None."""
        user = self.get_by_id(user_id)
        if user is None:
            return None
        user.points = (user.points or 0) + points
        self._db.flush()
        return user

    def update_streak(self, user_id: int, streak: int) -> Optional[User]:
        """Set a user's streak counter.  Returns updated User or None."""
        user = self.get_by_id(user_id)
        if user is None:
            return None
        user.streak = streak
        user.last_active = datetime.datetime.utcnow()
        self._db.flush()
        return user

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, user_id: int) -> bool:
        """Delete a user row.  Returns True if the row existed."""
        user = self.get_by_id(user_id)
        if user is None:
            return False
        self._db.delete(user)
        self._db.flush()
        return True
