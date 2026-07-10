"""
core/assessment/engine.py

AssessmentEngine — central router for all assessment challenge types.

Routes:
  type=architecture → architecture_challenges.get_architecture_challenge()
  type=tensor       → tensor_challenges.get_tensor_challenge()
  type=flops        → flops_challenges.get_flops_challenge()
  type=comparison   → comparison_challenges.get_comparison_challenge()

All answers are validated deterministically by comparing user_answer
to the challenge's 'answer' field (case-insensitive, normalized).
"""

from __future__ import annotations

import re
import time
from typing import Any

from core.assessment.architecture_challenges import get_architecture_challenge
from core.assessment.comparison_challenges import get_comparison_challenge
from core.assessment.flops_challenges import get_flops_challenge
from core.assessment.tensor_challenges import get_tensor_challenge

# ---------------------------------------------------------------------------
# Answer Validation
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Normalize answer for comparison: lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_answer(challenge: dict[str, Any], user_answer: str) -> dict[str, Any]:
    """
    Deterministically validate a user's answer against the challenge ground truth.

    Returns:
        {
            "is_correct": bool,
            "correct_answer": str,
            "user_answer": str,
            "explanation": str,
            "score": int   # 0 or 1
        }
    """
    correct = challenge.get("answer", "")
    explanation = challenge.get("explanation", "")

    # Try exact normalized match first
    is_correct = _normalize(user_answer) == _normalize(correct)

    # For multiple-choice, also accept the answer_index as a string ("0", "1", etc.)
    if not is_correct and "answer_index" in challenge:
        try:
            idx = int(user_answer.strip())
            is_correct = idx == challenge["answer_index"]
        except (ValueError, TypeError):
            pass

    # For numeric answers embedded in choice strings (e.g. "49" in "49 patches")
    if not is_correct:
        # Extract numbers from both and compare
        user_nums = re.findall(r"\d+\.?\d*", user_answer)
        correct_nums = re.findall(r"\d+\.?\d*", correct)
        if user_nums and correct_nums and set(user_nums) & set(correct_nums):
            # All numbers in user answer appear in correct answer
            if all(n in correct_nums for n in user_nums):
                is_correct = True

    return {
        "is_correct": is_correct,
        "correct_answer": correct,
        "user_answer": user_answer,
        "explanation": explanation,
        "score": 1 if is_correct else 0,
    }


# ---------------------------------------------------------------------------
# AssessmentEngine
# ---------------------------------------------------------------------------


class AssessmentEngine:
    """
    Central engine for generating and validating assessment challenges.

    All challenge generation is deterministic — no LLM calls.
    Answers are validated by comparing against backend-computed ground truth.
    """

    SUPPORTED_TYPES = ("architecture", "tensor", "flops", "comparison")
    SUPPORTED_DIFFICULTIES = ("beginner", "intermediate", "advanced")
    SUPPORTED_ARCHITECTURES = ("ResNet", "DenseNet", "U-Net", "Transformer", "ViT")

    def generate(
        self,
        challenge_type: str = "tensor",
        architecture: str = "ResNet",
        difficulty: str = "beginner",
        seed: int | None = None,
        db_metrics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a challenge.

        Args:
            challenge_type: architecture | tensor | flops | comparison
            architecture: ResNet | DenseNet | U-Net | Transformer | ViT
            difficulty: beginner | intermediate | advanced
            seed: Optional int seed for reproducibility
            db_metrics: Optional list of paper metrics from corpus (for comparison)

        Returns:
            Challenge dict with keys:
              challenge_id, question, choices, answer, answer_index, explanation,
              difficulty, assessment_type, architecture
        """
        ct = challenge_type.lower()
        diff = difficulty.lower()

        if ct not in self.SUPPORTED_TYPES:
            ct = "tensor"
        if diff not in self.SUPPORTED_DIFFICULTIES:
            diff = "beginner"

        challenge: dict[str, Any] = {}

        if ct == "architecture":
            challenge = get_architecture_challenge(
                architecture=architecture, difficulty=diff, seed=seed
            )
        elif ct == "tensor":
            challenge = get_tensor_challenge(difficulty=diff, seed=seed)
        elif ct == "flops":
            challenge = get_flops_challenge(difficulty=diff, seed=seed)
        elif ct == "comparison":
            challenge = get_comparison_challenge(difficulty=diff, seed=seed, db_metrics=db_metrics)

        # Normalize: always attach metadata
        challenge["assessment_type"] = ct
        challenge.setdefault("architecture", architecture)
        challenge.setdefault("difficulty", diff)
        challenge["generated_at"] = int(time.time())

        return challenge

    def validate(
        self,
        challenge: dict[str, Any],
        user_answer: str,
    ) -> dict[str, Any]:
        """
        Validate a user's answer against a previously generated challenge.

        Args:
            challenge: The full challenge dict returned by generate()
            user_answer: The user's submitted answer

        Returns:
            Validation result dict (see validate_answer)
        """
        return validate_answer(challenge, user_answer)


# Singleton instance
assessment_engine = AssessmentEngine()
