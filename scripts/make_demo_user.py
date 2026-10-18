"""Create (or reset) a pre-verified local demo user, skipping the email-verification step.

Usage: .venv\\Scripts\\python.exe scripts\\make_demo_user.py you@example.com "your password"
Writes to the same DB the API uses (DATABASE_URL from .env). Local development only.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
load_dotenv(root / ".env")

from backend.database import SessionLocal  # noqa: E402
from backend.models import User  # noqa: E402
from backend.modules.auth.security.hashing import hash_password  # noqa: E402

email, password = sys.argv[1].lower(), sys.argv[2]
with SessionLocal() as db:
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(email=email, name=email.split("@")[0], is_admin=False)
        db.add(user)
    user.hashed_password = hash_password(password)
    user.is_verified = True
    db.commit()
    print(f"demo user ready: {email}")
