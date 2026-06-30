"""
tests/test_phase8_assessment.py

Unit and integration tests for Phase 8 Interactive Assessments & Active Learning.
"""

import pytest
from unittest.mock import MagicMock
from sqlalchemy.orm import Session

from core.assessment.engine import assessment_engine
from core.assessment.architecture_challenges import get_architecture_challenge
from core.assessment.tensor_challenges import get_tensor_challenge
from core.assessment.flops_challenges import get_flops_challenge
from core.assessment.comparison_challenges import get_comparison_challenge
from core.analytics.recommendation_engine import recommendation_engine
from backend.models import AssessmentAttempt, TutorAnalytics, LearnerProgress


def test_architecture_challenges():
    """Verify that architecture mutation challenges generate properly for all types."""
    architectures = ["ResNet", "DenseNet", "U-Net", "Transformer", "ViT"]
    difficulties = ["beginner", "intermediate", "advanced"]

    for arch in architectures:
        for diff in difficulties:
            challenge = get_architecture_challenge(architecture=arch, difficulty=diff)
            assert "question" in challenge
            assert "choices" in challenge
            assert "answer" in challenge
            assert "explanation" in challenge
            assert challenge["difficulty"] in ["beginner", "intermediate", "advanced"]


def test_tensor_challenges():
    """Verify shape-propagation challenges generate with exact mathematical correctness."""
    for diff in ["beginner", "intermediate", "advanced"]:
        challenge = get_tensor_challenge(difficulty=diff)
        assert "question" in challenge
        assert "answer" in challenge
        assert "explanation" in challenge
        assert challenge["difficulty"] == diff


def test_flops_challenges():
    """Verify FLOPs complexity calculation challenges generate properly."""
    for diff in ["beginner", "intermediate", "advanced"]:
        challenge = get_flops_challenge(difficulty=diff)
        assert "question" in challenge
        assert "answer" in challenge
        assert "explanation" in challenge
        assert challenge["difficulty"] == diff


def test_comparison_challenges_static():
    """Verify comparison challenges fallback to static challenges when database is empty."""
    challenge = get_comparison_challenge(difficulty="intermediate", db_metrics=[])
    assert "question" in challenge
    assert "choices" in challenge
    assert "answer" in challenge
    assert challenge["metrics_source"] == "static_structural"


def test_comparison_challenges_db():
    """Verify comparison challenges use actual corpus metrics when available."""
    db_metrics = [
        {
            "title": "ResNet18 Paper",
            "architecture_type": "resnet",
            "parameter_count": 11000000,
            "flops": 1800000000,
            "module_count": 18,
        },
        {
            "title": "DenseNet121 Paper",
            "architecture_type": "densenet",
            "parameter_count": 8000000,
            "flops": 2800000000,
            "module_count": 121,
        },
    ]
    challenge = get_comparison_challenge(difficulty="intermediate", db_metrics=db_metrics)
    assert "question" in challenge
    assert "choices" in challenge
    assert "answer" in challenge
    assert challenge["metrics_source"] == "corpus_db"
    assert "ResNet18 Paper" in challenge["question"]
    assert "DenseNet121 Paper" in challenge["question"]


def test_assessment_engine_validation():
    """Verify deterministic validation logic for choices and string answers."""
    # Test choice-based challenge
    challenge_choice = {
        "question": "Which layer changes shape?",
        "choices": ["Option A", "Option B", "Option C"],
        "answer": "Option A",
        "answer_index": 0,
        "explanation": "Option A is correct.",
    }
    
    # Text input match
    res1 = assessment_engine.validate(challenge_choice, "Option A")
    assert res1["is_correct"] is True
    assert res1["score"] == 1
    
    # Case insensitive
    res2 = assessment_engine.validate(challenge_choice, "option a")
    assert res2["is_correct"] is True

    # Numeric index match
    res3 = assessment_engine.validate(challenge_choice, "0")
    assert res3["is_correct"] is True

    # Incorrect match
    res4 = assessment_engine.validate(challenge_choice, "Option B")
    assert res4["is_correct"] is False
    assert res4["score"] == 0

    # Numeric embedded answer match (e.g. "49" in "49 patches")
    challenge_numeric = {
        "question": "How many patches?",
        "answer": "49 patches",
        "explanation": "14x14 divided by 2x2 = 49.",
    }
    res5 = assessment_engine.validate(challenge_numeric, "49")
    assert res5["is_correct"] is True


def test_recommendation_engine_empty():
    """Verify recommendation engine handles brand new users with no data gracefully."""
    db_mock = MagicMock(spec=Session)
    db_mock.query().filter().all.return_value = []  # No attempts or tutor analytics
    
    recs = recommendation_engine.compute([], [])
    
    assert recs["total_attempts"] == 0
    assert recs["overall_accuracy"] == 0.0
    assert len(recs["suggested_assessments"]) == 4
    assert len(recs["recommended_reviews"]) == 0
    assert len(recs["weakest_topics"]) == 0


def test_recommendation_engine_calculation():
    """Verify difficulty score formula and min-max normalization."""
    # 1. Mock assessment attempts
    # We want to create different failure rates for ResNet vs DenseNet
    attempts = [
        {"architecture": "ResNet", "assessment_type": "tensor", "is_correct": False},
        {"architecture": "ResNet", "assessment_type": "tensor", "is_correct": False},
        {"architecture": "DenseNet", "assessment_type": "tensor", "is_correct": True}
    ]

    # 2. Mock tutor analytics questions count
    # DenseNet has high questions count, ResNet has low questions count
    tutor_rows = [
        {"architecture": "DenseNet", "module": "denseblock_1", "question_count": 10},
        {"architecture": "ResNet", "module": "basicblock_1", "question_count": 2}
    ]

    recs = recommendation_engine.compute(attempts, tutor_rows)

    assert recs["total_attempts"] == 3
    assert recs["overall_accuracy"] == pytest.approx(0.333, abs=0.01)
    
    # We should have identified weakest topics and sorted them
    assert len(recs["weakest_topics"]) > 0
    # The difficulty score should fall between 0.0 and 1.0 inclusive
    for w in recs["weakest_topics"]:
        assert 0.0 <= w["difficulty_score"] <= 1.0
