"""
backend/services/user_service.py

Business logic and orchestration for User operations.
"""

from typing import Optional, Sequence
from sqlalchemy.orm import Session

from backend.models import User
from backend.repositories.user_repository import UserRepository


class UserService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = UserRepository(db)

    def get_user(self, user_id: int) -> Optional[User]:
        return self.repo.get_by_id(user_id)

    def get_or_create_user(self, email: str, name: str, avatar_url: Optional[str] = None) -> User:
        user = self.repo.get_by_email(email)
        if user:
            # Optionally update name/avatar if they changed
            return user
        
        user = self.repo.create(email=email, name=name, avatar_url=avatar_url)
        self.db.commit()
        return user

    def get_leaderboard(self, limit: int = 10) -> Sequence[User]:
        return self.repo.list_all(limit=limit)

    def award_points(self, user_id: int, points: int) -> Optional[User]:
        user = self.repo.add_points(user_id, points)
        if user:
            self.db.commit()
        return user
