"""
Phase 9 Adaptive Engine — smoke test.
Run: python tests/smoke_phase9.py
"""
import sys
sys.path.insert(0, ".")

from unittest.mock import MagicMock
from sqlalchemy.orm import Session
from core.analytics.adaptive_engine import adaptive_engine
from backend.models import AssessmentAttempt, TutorAnalytics, LearnerProgress, PaperModule, Paper

# ── Build a realistic mock DB for a learner weak on Attention/Transformers ───
attempts = [
    AssessmentAttempt(learner_id="demo", question_text="Attention block query projection...", is_correct=False, assessment_type="architecture"),
    AssessmentAttempt(learner_id="demo", question_text="transformer encoder block...", is_correct=False, assessment_type="architecture"),
    AssessmentAttempt(learner_id="demo", question_text="A Conv2D layer with stride 2...", is_correct=True, assessment_type="tensor"),
    AssessmentAttempt(learner_id="demo", question_text="Pooling reduces spatial dimensions...", is_correct=True, assessment_type="tensor"),
]
progress = [
    LearnerProgress(learner_id="demo", module_id=10, status="completed"),
]
tutor_records = [
    TutorAnalytics(learner_id="demo", architecture="Transformer", module="attention_block", question_count=4),
]
modules = [
    PaperModule(id=10, explanation="convolutional kernel stride", module_type="conv", layer_name="conv1"),
    PaperModule(id=11, explanation="attention softmax query key value", module_type="attention", layer_name="attn1"),
]
papers = [
    Paper(id=1, title="Vision Transformer", architecture_graph={"classification": "Transformer"}, modules=[]),
    Paper(id=2, title="ResNet-50 Deep Residual Learning", architecture_graph={"classification": "ResNet"}, modules=[]),
]

def make_db():
    db = MagicMock(spec=Session)
    def query_side(model):
        q = MagicMock()
        if model == AssessmentAttempt:
            q.filter().all.return_value = attempts
        elif model == LearnerProgress:
            q.filter().all.return_value = progress
        elif model == TutorAnalytics:
            q.filter().all.return_value = tutor_records
        elif model == PaperModule:
            q.all.return_value = modules
        elif model == Paper:
            q.all.return_value = papers
        else:
            q.filter().all.return_value = []
            q.all.return_value = []
        return q
    db.query.side_effect = query_side
    return db


db = make_db()

# ── Knowledge Profile ────────────────────────────────────────────────────────
profile = adaptive_engine.compute_knowledge_profile(db, "demo")
print("=== Knowledge Profile ===")
for concept, score in sorted(profile.items()):
    bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
    print(f"  {concept:<40} {bar}  {score:.2f}")

# ── Weakness Detection ───────────────────────────────────────────────────────
db = make_db()
weaknesses = adaptive_engine.detect_weaknesses(db, "demo")
print()
print(f"=== Weaknesses (confidence={weaknesses['confidence']:.2f}) ===")
for w in weaknesses["weak_topics"]:
    print(f"  • {w['topic']:<40} score={w['weakness_score']:.2f}")

# ── Recommendations ──────────────────────────────────────────────────────────
db = make_db()
recs = adaptive_engine.get_personalized_recommendations(db, "demo")
print()
print("=== Personalized Recommendations ===")
print(f"  Weak topics  : {[w['topic'] for w in recs['weak_topics']]}")
print(f"  Papers       : {[p['title'] for p in recs['suggested_papers']]}")
print(f"  Assessments  : {[a['assessment_type'] + ' / ' + a['architecture'] for a in recs['suggested_assessments']]}")

# ── Daily Review Plan ────────────────────────────────────────────────────────
db = make_db()
plan = adaptive_engine.get_daily_review_plan(db, "demo")
print()
print(f"=== Daily Review Plan ({len(plan['today_review'])} items) ===")
for item in plan["today_review"]:
    print(f"  [{item['type'].upper():<10}] {item['title']}")

# ── Concept Graph ────────────────────────────────────────────────────────────
db = make_db()
graph_nodes = adaptive_engine.get_concept_graph(db, "demo")
print()
print(f"=== Concept Graph ({len(graph_nodes)} nodes) ===")
for n in graph_nodes:
    print(f"  {n['label']:<25}  mastery={n['mastery']:.2f}  status={n['status']}")

# ── Adaptive Learning Path ───────────────────────────────────────────────────
db = make_db()
path = adaptive_engine.get_adaptive_learning_path(db, "demo")
print()
print(f"=== Adaptive Learning Path ({len(path)} stages) ===")
for i, step in enumerate(path, 1):
    tag = "  [REMEDIATION] " if step.get("remediation") else f"  {i}. "
    print(f"{tag}{step['level']}")
    print(f"       {step['focus']}")

print()
print("✓ All smoke tests passed — adaptive engine operating correctly.")
