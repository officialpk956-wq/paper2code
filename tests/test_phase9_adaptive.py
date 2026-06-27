"""
tests/test_phase9_adaptive.py

Unit tests for Phase 9 Adaptive Learning & Personalization features.
"""

import pytest
from unittest.mock import MagicMock
from sqlalchemy.orm import Session

from core.analytics.adaptive_engine import adaptive_engine
from core.agents.tutor_agent import tutor_manager
from backend.models import AssessmentAttempt, TutorAnalytics, LearnerProgress, PaperModule, Paper


def test_learner_knowledge_profiles_distinct():
    """Success Criteria: Two learners with different strengths must receive different profiles."""
    db_mock = MagicMock(spec=Session)

    # Learner A: Strong in Convolutions (100% correct), Weak in Attention (0% correct)
    attempts_a = [
        AssessmentAttempt(learner_id="learner_a", question_text="A Conv2D layer...", is_correct=True, assessment_type="tensor"),
        AssessmentAttempt(learner_id="learner_a", question_text="Attention block...", is_correct=False, assessment_type="architecture")
    ]
    p_a = LearnerProgress(learner_id="learner_a", status="completed", entity_type="paper_module", entity_id="101")
    p_a.module_id = 101
    progress_a = [p_a]
    tutor_a = []

    # Learner B: Weak in Convolutions (0% correct), Strong in Attention (100% correct)
    attempts_b = [
        AssessmentAttempt(learner_id="learner_b", question_text="A Conv2D layer...", is_correct=False, assessment_type="tensor"),
        AssessmentAttempt(learner_id="learner_b", question_text="Attention block...", is_correct=True, assessment_type="architecture")
    ]
    p_b = LearnerProgress(learner_id="learner_b", status="completed", entity_type="paper_module", entity_id="102")
    p_b.module_id = 102
    progress_b = [p_b]
    tutor_b = []

    # Mock DB modules matching concepts
    module_conv = PaperModule(id=101, explanation="convolutional kernel", module_type="conv", layer_name="conv1")
    module_attn = PaperModule(id=102, explanation="attention sequence softmax", module_type="attention", layer_name="attn1")
    all_modules = [module_conv, module_attn]

    # Configure query routing for Learner A
    def query_routing_a(model):
        mock_query = MagicMock()
        if model == AssessmentAttempt:
            mock_query.filter().all.return_value = attempts_a
        elif model == LearnerProgress:
            mock_query.filter().all.return_value = progress_a
        elif model == TutorAnalytics:
            mock_query.filter().all.return_value = tutor_a
        elif model == PaperModule:
            mock_query.all.return_value = all_modules
        else:
            mock_query.filter().all.return_value = []
        return mock_query

    db_mock.query.side_effect = query_routing_a
    profile_a = adaptive_engine.compute_knowledge_profile(db_mock, "learner_a")

    # Configure query routing for Learner B
    def query_routing_b(model):
        mock_query = MagicMock()
        if model == AssessmentAttempt:
            mock_query.filter().all.return_value = attempts_b
        elif model == LearnerProgress:
            mock_query.filter().all.return_value = progress_b
        elif model == TutorAnalytics:
            mock_query.filter().all.return_value = tutor_b
        elif model == PaperModule:
            mock_query.all.return_value = all_modules
        else:
            mock_query.filter().all.return_value = []
        return mock_query

    db_mock.query.side_effect = query_routing_b
    profile_b = adaptive_engine.compute_knowledge_profile(db_mock, "learner_b")

    # Assert distinct profiles reflecting different strengths
    assert profile_a["Convolutions"] > profile_b["Convolutions"]
    assert profile_b["Attention"] > profile_a["Attention"]


def test_weakness_detection():
    """Verify weakness scores are correctly compiled and sorted."""
    db_mock = MagicMock(spec=Session)

    # Learner failed Residual Connection assessments repeatedly
    attempts = [
        AssessmentAttempt(learner_id="test_user", question_text="ResNet identity skip connection...", is_correct=False, assessment_type="architecture"),
        AssessmentAttempt(learner_id="test_user", question_text="skip connection...", is_correct=False, assessment_type="architecture")
    ]
    progress = []
    tutor = [
        TutorAnalytics(learner_id="test_user", architecture="ResNet", module="residual_block", question_count=4)
    ]

    def query_routing(model):
        mock_query = MagicMock()
        if model == AssessmentAttempt:
            mock_query.filter().all.return_value = attempts
        elif model == LearnerProgress:
            mock_query.filter().all.return_value = progress
        elif model == TutorAnalytics:
            mock_query.filter().all.return_value = tutor
        elif model == PaperModule:
            mock_query.all.return_value = []
        return mock_query

    db_mock.query.side_effect = query_routing
    weakness_info = adaptive_engine.detect_weaknesses(db_mock, "test_user")

    assert len(weakness_info["weak_topics"]) > 0
    top_weakness = weakness_info["weak_topics"][0]
    assert top_weakness["topic"] == "Residual Connections"
    assert top_weakness["weakness_score"] > 0.5


def test_personalized_recommendations_and_review_plan():
    """Verify recommended actions align with weakness profile and daily plan compiles exactly 3 items."""
    db_mock = MagicMock(spec=Session)

    attempts = [
        AssessmentAttempt(learner_id="test_user", question_text="transformer block weights...", is_correct=False, assessment_type="architecture")
    ]
    progress = []
    tutor = []
    
    # Mock papers matching classification
    paper1 = Paper(id=1, title="Vision Transformer Paper", architecture_graph={"classification": "Transformer"}, modules=[])
    all_papers = [paper1]

    def query_routing(model):
        mock_query = MagicMock()
        if model == AssessmentAttempt:
            mock_query.filter().all.return_value = attempts
        elif model == LearnerProgress:
            mock_query.filter().all.return_value = progress
        elif model == TutorAnalytics:
            mock_query.filter().all.return_value = tutor
        elif model == Paper:
            mock_query.all.return_value = all_papers
        elif model == PaperModule:
            mock_query.all.return_value = []
        return mock_query

    db_mock.query.side_effect = query_routing

    recs = adaptive_engine.get_personalized_recommendations(db_mock, "test_user")
    assert len(recs["suggested_papers"]) > 0
    assert recs["suggested_papers"][0]["classification"] == "Transformer"

    plan = adaptive_engine.get_daily_review_plan(db_mock, "test_user")
    assert len(plan["today_review"]) == 3


def test_adaptive_quizzes():
    """Verify quiz generator prioritizes questions matching weakness profile."""
    # Struggling with FLOPs Reasoning
    weak_topics = ["FLOPs Reasoning"]
    module_data = {"layer_name": "Conv2d_1", "module_type": "conv", "explanation": "Some explanation", "paper_title": "ResNet18"}

    questions = tutor_manager.generate_quiz(module_data, weak_topics=weak_topics)
    assert len(questions) > 0
    
    # First question should be the adapted FLOPs query
    assert "FLOP complexity" in questions[0]["question"]


def test_adaptive_learning_path():
    """Verify remediation stages are dynamically inserted for struggling prerequisites."""
    db_mock = MagicMock(spec=Session)

    # Struggling with Residual connections (0% correct)
    attempts = [
        AssessmentAttempt(learner_id="test_user", question_text="resnet skip connection...", is_correct=False, assessment_type="architecture")
    ]

    def query_routing(model):
        mock_query = MagicMock()
        if model == AssessmentAttempt:
            mock_query.filter().all.return_value = attempts
        elif model == LearnerProgress:
            mock_query.filter().all.return_value = []
        elif model == TutorAnalytics:
            mock_query.filter().all.return_value = []
        elif model == PaperModule:
            mock_query.all.return_value = []
        return mock_query

    db_mock.query.side_effect = query_routing

    path = adaptive_engine.get_adaptive_learning_path(db_mock, "test_user")
    
    # We should have injected a remediation node right after beginner
    assert any(step.get("remediation") is True and step.get("concept") == "Residual Connections" for step in path)
