import json
import os
import sys

# Add the backend folder to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models import Problem

def seed_problems():
    db: Session = SessionLocal()
    with open('seeds/problems.json', 'r') as f:
        problems_data = json.load(f)
    
    for pd in problems_data:
        existing = db.query(Problem).filter(Problem.id == pd['slug']).first()
        if not existing:
            new_problem = Problem(
                id=pd['slug'],
                slug=pd['slug'],
                title=pd['title'],
                difficulty=pd['difficulty'],
                python_template=pd['python_template'],
                test_cases=pd['test_cases'],
                hints=pd['hints']
            )
            db.add(new_problem)
    
    db.commit()
    db.close()
    print("Seeded 10 problems.")

if __name__ == '__main__':
    seed_problems()
