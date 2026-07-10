"""
Usage: python -m backend.scripts.make_admin user@example.com
Makes the given user an admin. Requires DATABASE_URL to be set.
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from backend.database import SessionLocal
from backend.models import User


def make_admin(email: str) -> None:
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email.lower().strip()).first()
        if not user:
            print(f"No user found with email: {email}")
            sys.exit(1)
        user.is_admin = True
        db.commit()
        print(f"✓ {email} is now admin (id={user.id})")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m backend.scripts.make_admin <email>")
        sys.exit(1)
    make_admin(sys.argv[1])
