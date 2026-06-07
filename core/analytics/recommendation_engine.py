"""
core/analytics/recommendation_engine.py

Recommendation Engine — Phase 8 Decision 7 & 8.

Logic:
  difficulty_score = (normalized_failure_rate × 0.6) + (normalized_question_frequency × 0.4)

Normalization (min-max):
  normalized_value = (value - min) / (max - min)
  Edge case: if max == min → return 0 (no divide-by-zero)

Reads from SQLite tables:
  - assessment_attempts  (for failure_rate per architecture/type)
  - tutor_analytics      (for question_frequency per module/architecture)
  - learner_progress     (for completion status)
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from collections import defaultdict


# ---------------------------------------------------------------------------
# Min-Max Normalization
# ---------------------------------------------------------------------------

def _minmax_normalize(values: List[float]) -> List[float]:
    """
    Apply min-max normalization.
    Returns 0 for all values if max == min (edge case guard).
    """
    if not values:
        return []
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _normalize_single(value: float, mn: float, mx: float) -> float:
    """Normalize a single value given pre-computed min and max."""
    if mx == mn:
        return 0.0
    return (value - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# RecommendationEngine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    Computes personalized learning recommendations from SQLite analytics.

    All logic is deterministic — no LLM involvement.
    """

    def compute(
        self,
        db: Session,
        learner_id: str,
    ) -> Dict[str, Any]:
        """
        Compute recommendations for a learner.

        Returns:
          {
            "recommended_reviews": list[dict],
            "weakest_topics": list[dict],
            "suggested_assessments": list[dict]
          }
        """
        from backend.models import AssessmentAttempt, TutorAnalytics, LearnerProgress

        # ── 1. Load all attempts for this learner ──────────────────────────
        attempts = (
            db.query(AssessmentAttempt)
            .filter(AssessmentAttempt.learner_id == learner_id)
            .all()
        )

        # ── 2. Compute failure rate per (architecture, assessment_type) ────
        bucket_attempts: Dict[str, int] = defaultdict(int)
        bucket_failures: Dict[str, int] = defaultdict(int)

        for a in attempts:
            key = f"{a.architecture or 'General'}::{a.assessment_type or 'general'}"
            bucket_attempts[key] += 1
            if not a.is_correct:
                bucket_failures[key] += 1

        failure_rates: Dict[str, float] = {}
        for key, total in bucket_attempts.items():
            failure_rates[key] = bucket_failures[key] / total if total > 0 else 0.0

        # ── 3. Load tutor question frequency per (architecture, module) ────
        tutor_rows = (
            db.query(TutorAnalytics)
            .filter(TutorAnalytics.learner_id == learner_id)
            .all()
        )

        question_freq: Dict[str, int] = defaultdict(int)
        for t in tutor_rows:
            key = f"{t.architecture or 'General'}::{t.module or 'general'}"
            question_freq[key] += t.question_count

        # ── 4. Compute difficulty scores ───────────────────────────────────
        # Combine keys from both sources
        all_keys = set(failure_rates.keys()) | set(question_freq.keys())

        if not all_keys:
            return _empty_recommendations()

        fr_values = [failure_rates.get(k, 0.0) for k in all_keys]
        qf_values = [float(question_freq.get(k, 0)) for k in all_keys]

        fr_min, fr_max = min(fr_values), max(fr_values)
        qf_min, qf_max = min(qf_values), max(qf_values)

        scored: List[Dict[str, Any]] = []
        for key in all_keys:
            arch, topic = key.split("::", 1)
            fr = failure_rates.get(key, 0.0)
            qf = float(question_freq.get(key, 0))

            norm_fr = _normalize_single(fr, fr_min, fr_max)
            norm_qf = _normalize_single(qf, qf_min, qf_max)

            difficulty_score = round(norm_fr * 0.6 + norm_qf * 0.4, 4)

            scored.append({
                "key": key,
                "architecture": arch,
                "topic": topic,
                "failure_rate": round(fr, 3),
                "question_frequency": int(qf),
                "norm_failure_rate": round(norm_fr, 4),
                "norm_question_frequency": round(norm_qf, 4),
                "difficulty_score": difficulty_score,
            })

        # Sort by difficulty_score descending (hardest topics first)
        scored.sort(key=lambda x: x["difficulty_score"], reverse=True)

        # ── 5. Build recommendations ───────────────────────────────────────
        weakest_topics = scored[:5]

        # Suggested reviews: top failing architectures
        recommended_reviews = []
        seen_arch = set()
        for s in scored:
            if s["failure_rate"] > 0.3 and s["architecture"] not in seen_arch:
                recommended_reviews.append({
                    "architecture": s["architecture"],
                    "topic": s["topic"],
                    "failure_rate": s["failure_rate"],
                    "reason": f"You answered {int(s['failure_rate']*100)}% of {s['architecture']} questions incorrectly.",
                })
                seen_arch.add(s["architecture"])
            if len(recommended_reviews) >= 3:
                break

        # Suggested assessments: challenge types with no attempts
        all_types = {"architecture", "tensor", "flops", "comparison"}
        attempted_types = {a.assessment_type for a in attempts if a.assessment_type}
        untried_types = all_types - attempted_types

        suggested_assessments = []
        for t in sorted(untried_types):
            suggested_assessments.append({
                "assessment_type": t,
                "reason": f"You haven't tried {t} challenges yet.",
                "suggested_difficulty": "beginner",
            })

        # Also suggest higher difficulty for consistently correct areas
        correct_by_type: Dict[str, List[bool]] = defaultdict(list)
        for a in attempts:
            if a.assessment_type:
                correct_by_type[a.assessment_type].append(a.is_correct)

        for atype, results in correct_by_type.items():
            if len(results) >= 3 and sum(results) / len(results) >= 0.8:
                suggested_assessments.append({
                    "assessment_type": atype,
                    "reason": f"You're performing well ({int(sum(results)/len(results)*100)}% correct) — try advanced {atype} challenges.",
                    "suggested_difficulty": "advanced",
                })

        return {
            "recommended_reviews": recommended_reviews,
            "weakest_topics": [
                {k: v for k, v in s.items() if k != "key"}
                for s in weakest_topics
            ],
            "suggested_assessments": suggested_assessments[:5],
            "total_attempts": len(attempts),
            "overall_accuracy": (
                round(sum(1 for a in attempts if a.is_correct) / len(attempts), 3)
                if attempts else 0.0
            ),
        }


def _empty_recommendations() -> Dict[str, Any]:
    """Return empty recommendations when no data exists."""
    return {
        "recommended_reviews": [],
        "weakest_topics": [],
        "suggested_assessments": [
            {"assessment_type": t, "reason": "No assessment data yet — start here!", "suggested_difficulty": "beginner"}
            for t in ["architecture", "tensor", "flops", "comparison"]
        ],
        "total_attempts": 0,
        "overall_accuracy": 0.0,
    }


# Singleton
recommendation_engine = RecommendationEngine()
