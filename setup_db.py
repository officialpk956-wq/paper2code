from backend.database import engine
from backend.models import Base
from backend.services.achievement_service import seed_achievements
from backend.database import SessionLocal
import subprocess

print("Creating all tables via SQLAlchemy...")
Base.metadata.create_all(bind=engine)

print("Seeding achievements...")
db = SessionLocal()
seed_achievements(db)
db.close()

print("Stamping alembic head...")
subprocess.run(["alembic", "stamp", "head"])
print("Done.")
