"""
tests/test_phase9_adaptive.py

Unit tests for Phase 9 Adaptive Learning & Personalization features.
"""

import pytest

from core.analytics.adaptive_engine import adaptive_engine
from core.agents.tutor_agent import tutor_manager


def test_learner_knowledge_profiles_distinct():
    """Success Criteria: Two learners with different strengths must receive different profiles."""

    # Learner A: Strong in Convolutions (100% correct), Weak in Attention (0% correct)
    attempts_a = [
        {"learner_id": "learner_a", "question_text": "A Conv2D layer...", "is_correct": True, "assessment_type": "tensor", "architecture": "CNN"},
        {"learner_id": "learner_a", "question_text": "Attention block...", "is_correct": False, "assessment_type": "architecture", "architecture": "Transformer"}
    ]
    progress_a = [
        {"learner_id": "learner_a", "status": "completed", "entity_type": "paper_module", "entity_id": "101", "module_id": 101}
    ]
    tutor_a = []

    # Learner B: Weak in Convolutions (0% correct), Strong in Attention (100% correct)
    attempts_b = [
        {"learner_id": "learner_b", "question_text": "A Conv2D layer...", "is_correct": False, "assessment_type": "tensor", "architecture": "CNN"},
        {"learner_id": "learner_b", "question_text": "Attention block...", "is_correct": True, "assessment_type": "architecture", "architecture": "Transformer"}
    ]
    progress_b = [
        {"learner_id": "learner_b", "status": "completed", "entity_type": "paper_module", "entity_id": "102", "module_id": 102}
    ]
    tutor_b = []

    all_modules = [
        {"id": 101, "explanation": "convolutional kernel", "module_type": "conv", "layer_name": "conv1", "architecture": "CNN", "paper_id": 1},
        {"id": 102, "explanation": "attention sequence softmax", "module_type": "attention", "layer_name": "attn1", "architecture": "Transformer", "paper_id": 2}
    ]

    profile_a = adaptive_engine.compute_knowledge_profile(attempts_a, progress_a, tutor_a, all_modules)
    profile_b = adaptive_engine.compute_knowledge_profile(attempts_b, progress_b, tutor_b, all_modules)

    # Assert distinct profiles reflecting different strengths
    assert profile_a.get("Convolutions", 0.0) > profile_b.get("Convolutions", 0.0)
    assert profile_b.get("Attention", 0.0) > profile_a.get("Attention", 0.0)


def test_weakness_detection():
    """Verify weakness scores are correctly compiled and sorted."""
    attempts = [
        {"learner_id": "test_user", "question_text": "ResNet identity skip connection...", "is_correct": False, "assessment_type": "architecture", "architecture": "ResNet"},
        {"learner_id": "test_user", "question_text": "skip connection...", "is_correct": False, "assessment_type": "architecture", "architecture": "ResNet"}
    ]
    tutor = [
        {"learner_id": "test_user", "architecture": "ResNet", "module": "residual_block", "question_count": 4}
    ]

    weakness_info = adaptive_engine.detect_weaknesses(attempts, [], tutor, [])

    assert len(weakness_info["weak_topics"]) > 0
    top_weakness = weakness_info["weak_topics"][0]
    assert top_weakness["topic"] == "Residual Connections"
    assert top_weakness["weakness_score"] > 0.5


def test_personalized_recommendations_and_review_plan():
    """Verify recommended actions align with weakness profile and daily plan compiles exactly 3 items."""
    attempts = [
        {"learner_id": "test_user", "question_text": "transformer block weights...", "is_correct": False, "assessment_type": "architecture", "architecture": "Transformer"}
    ]
    tutor = []
    progress = []
    all_papers = [
        {"id": 1, "title": "Vision Transformer Paper", "architecture_graph": {"classification": "Transformer"}, "modules": []}
    ]

    recs = adaptive_engine.get_personalized_recommendations(attempts, progress, tutor, [], all_papers)
    assert len(recs["suggested_papers"]) > 0
    assert recs["suggested_papers"][0]["classification"] == "Transformer"

    plan = adaptive_engine.get_daily_review_plan(attempts, progress, tutor, [], all_papers)
    assert len(plan["today_review"]) == 3


def test_adaptive_quizzes():
    """Verify quiz generator prioritizes questions matching weakness profile."""
    weak_topics = ["FLOPs Reasoning"]
    module_data = {"layer_name": "Conv2d_1", "module_type": "conv", "explanation": "Some explanation", "paper_title": "ResNet18"}

    questions = tutor_manager.generate_quiz(module_data, weak_topics=weak_topics)
    assert len(questions) > 0
    assert "FLOP complexity" in questions[0]["question"]


def test_adaptive_learning_path():
    """Verify remediation stages are dynamically inserted for struggling prerequisites."""
    attempts = [
        {"learner_id": "test_user", "question_text": "resnet skip connection...", "is_correct": False, "assessment_type": "architecture", "architecture": "ResNet"}
    ]

    path = adaptive_engine.get_adaptive_learning_path(attempts, [], [], [])
    assert any(step.get("remediation") is True and step.get("concept") == "Residual Connections" for step in path)
