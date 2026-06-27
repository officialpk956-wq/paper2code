import os
import json
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.database import SessionLocal, engine
from backend.models import Base, Problem, InterviewQuestion, Roadmap

Base.metadata.create_all(bind=engine)

def seed_data():
    db = SessionLocal()
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'json_dump')
        
        # Seed Problems
        problems_file = os.path.join(data_dir, 'problems.json')
        if os.path.exists(problems_file):
            with open(problems_file, 'r', encoding='utf-8') as f:
                problems = json.load(f)
                for p in problems:
                    if not db.query(Problem).filter_by(id=p['id']).first():
                        prob = Problem(
                            id=p['id'],
                            slug=p['slug'],
                            title=p['title'],
                            category=p['category'],
                            difficulty=p['difficulty'],
                            estimated_time=p['estimatedTime'],
                            description=p['description'],
                            tags=p.get('tags', []),
                            related_architectures=p.get('relatedArchitectures', []),
                            related_papers=p.get('relatedPapers', []),
                            related_math=p.get('relatedMath', []),
                            learning_points=p.get('learningPoints', []),
                            visualization_url=p.get('visualizationUrl'),
                            python_template=p.get('pythonTemplate', ''),
                            test_cases=p.get('testCases', []),
                            hints=p.get('hints', {}),
                            explanation=p.get('explanation', {})
                        )
                        db.add(prob)
            print("Problems seeded.")

        # Seed Interview Questions
        interviews_file = os.path.join(data_dir, 'interview_questions.json')
        if os.path.exists(interviews_file):
            with open(interviews_file, 'r', encoding='utf-8') as f:
                interviews = json.load(f)
                for i in interviews:
                    if not db.query(InterviewQuestion).filter_by(id=i['id']).first():
                        interview = InterviewQuestion(
                            id=i['id'],
                            question=i.get('question', ''),
                            difficulty=i.get('difficulty', ''),
                            category=i.get('category', ''),
                            companies=i.get('companies', []),
                            tags=i.get('tags', []),
                            key_points=i.get('keyPoints', []),
                            expected_answer=i.get('expectedAnswer', ''),
                            hints=i.get('hints', [])
                        )
                        db.add(interview)
            print("Interviews seeded.")

        # Seed Roadmaps
        roadmaps_file = os.path.join(data_dir, 'roadmaps.json')
        if os.path.exists(roadmaps_file):
            with open(roadmaps_file, 'r', encoding='utf-8') as f:
                # Assuming roadmap file exports a dictionary or list
                roadmaps = json.load(f)
                # the file src/data/roadmaps.ts usually exports ROADMAP_NODES which is a flat list
                # We will just insert it as a single roadmap or based on the file structure.
                if isinstance(roadmaps, list):
                    # just pack them into one default roadmap for now
                    if not db.query(Roadmap).filter_by(id="default").first():
                        rm = Roadmap(id="default", title="Default Roadmap", description="Core AI Engineering", nodes=roadmaps)
                        db.add(rm)
                elif isinstance(roadmaps, dict):
                    for k, v in roadmaps.items():
                        if not db.query(Roadmap).filter_by(id=k).first():
                            rm = Roadmap(id=k, title=k, description="", nodes=v)
                            db.add(rm)
            print("Roadmaps seeded.")

        db.commit()
        print("Database seeding completed.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
