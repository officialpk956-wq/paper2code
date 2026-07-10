"""
core/lab/hypothesis_engine.py — Hypothesis & Experiment Tracking (Phase 11)

Responsibilities:
  - Define the Hypothesis and ExperimentResult data structures
  - Provide a HypothesisEngine that can score/evaluate user predictions vs actual diffs
  - Provide helpers for the frontend to generate prediction challenge data

Note: Persistence is handled by the frontend using localStorage.
      This module provides pure-Python logic for scoring predictions.
"""

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A user-submitted hypothesis about the effect of a mutation."""

    id: str
    mutation_type: str
    architecture: str
    prediction: str  # User's free-text prediction
    predicted_param_direction: str  # "increase", "decrease", "no_change"
    predicted_flops_direction: str  # "increase", "decrease", "no_change"
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """The result of running a mutation experiment, with scoring."""

    id: str
    hypothesis_id: str | None
    mutation_type: str
    architecture: str
    mutations_applied: list[dict[str, Any]]
    diff_summary: str
    param_delta_pct: float
    flops_delta: int
    depth_delta: int
    prediction_score: float | None  # 0.0–1.0
    prediction_feedback: str | None
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Hypothesis Engine
# ---------------------------------------------------------------------------


class HypothesisEngine:
    """
    Scores user predictions against actual diff results.

    Scoring rubric (max 1.0):
      - param_direction correct  : +0.35
      - flops_direction correct  : +0.35
      - free-text quality bonus  : +0.30 (based on word count + keyword matching)
    """

    # Mutation-specific keywords that indicate domain understanding
    MUTATION_KEYWORDS = {
        "increase_depth": [
            "capacity",
            "expressivity",
            "overfitting",
            "gradient",
            "vanishing",
            "depth",
        ],
        "decrease_depth": ["faster", "inference", "pruning", "efficiency", "smaller"],
        "increase_width": ["channels", "width", "representation", "parameters", "larger"],
        "decrease_width": ["efficient", "compressed", "mobile", "smaller", "lightweight"],
        "add_residual": ["gradient flow", "shortcut", "residual", "skip", "highway", "identity"],
        "remove_residual": ["degradation", "vanishing gradient", "accuracy drop", "deeper"],
        "add_attention": [
            "global context",
            "receptive field",
            "self-attention",
            "tokens",
            "quadratic",
        ],
        "change_patch_size": ["resolution", "tokens", "granularity", "patches", "sequence length"],
        "change_hidden_dim": ["embedding", "capacity", "d_model", "hidden", "projection"],
    }

    def score_prediction(
        self,
        hypothesis: dict[str, Any],
        actual_diff: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score a user's prediction against the actual diff.

        Args:
            hypothesis: dict with keys: predicted_param_direction, predicted_flops_direction,
                        prediction (free text), mutation_type
            actual_diff: dict returned by diff_engine.compute_diff()

        Returns:
            dict with: score (0–1), max_score (1.0), feedback, param_correct, flops_correct
        """
        score = 0.0
        feedback_parts = []

        # --- Param direction ---
        actual_param_pct = actual_diff.get("param_delta", {}).get("pct", 0)
        predicted_param = hypothesis.get("predicted_param_direction", "no_change")
        actual_param_dir = _direction(actual_param_pct)

        param_correct = predicted_param == actual_param_dir
        if param_correct:
            score += 0.35
            feedback_parts.append("✓ Parameter prediction correct.")
        else:
            feedback_parts.append(
                f"✗ Parameter prediction: you said '{predicted_param}', actual was '{actual_param_dir}' "
                f"({actual_param_pct:+.1f}%)."
            )

        # --- FLOPs direction ---
        actual_flops_abs = actual_diff.get("flops_delta", {}).get("absolute", 0)
        predicted_flops = hypothesis.get("predicted_flops_direction", "no_change")
        actual_flops_dir = _direction(actual_flops_abs)

        flops_correct = predicted_flops == actual_flops_dir
        if flops_correct:
            score += 0.35
            feedback_parts.append("✓ FLOPs prediction correct.")
        else:
            feedback_parts.append(
                f"✗ FLOPs prediction: you said '{predicted_flops}', actual was '{actual_flops_dir}' "
                f"({actual_flops_abs:+d})."
            )

        # --- Free-text quality bonus ---
        prediction_text = (hypothesis.get("prediction") or "").lower()
        mutation_type = hypothesis.get("mutation_type", "")
        keywords = self.MUTATION_KEYWORDS.get(mutation_type, [])

        words_matched = sum(1 for kw in keywords if kw in prediction_text)
        word_count = len(prediction_text.split())
        text_bonus = 0.0
        if word_count >= 10 and words_matched >= 1:
            text_bonus = 0.15
        if word_count >= 20 and words_matched >= 2:
            text_bonus = 0.30
        score += text_bonus

        if text_bonus > 0:
            feedback_parts.append(
                f"✓ Quality bonus (+{int(text_bonus * 100)}%): good use of domain terminology."
            )
        else:
            if keywords:
                feedback_parts.append(f"Tip: Explain using terms like: {', '.join(keywords[:4])}.")

        score = min(1.0, round(score, 2))

        return {
            "score": score,
            "max_score": 1.0,
            "param_correct": param_correct,
            "flops_correct": flops_correct,
            "text_bonus": round(text_bonus, 2),
            "feedback": " ".join(feedback_parts),
            "actual_param_direction": actual_param_dir,
            "actual_flops_direction": actual_flops_dir,
        }

    def generate_prediction_prompt(self, mutation_type: str, architecture: str) -> dict[str, Any]:
        """
        Return a structured prediction prompt for the 'Predict Before Reveal' challenge.
        """
        prompts = {
            "increase_depth": {
                "question": f"If you increase the depth of this {architecture} by adding more blocks, what do you predict will happen?",
                "hints": [
                    "Think about parameter count",
                    "Consider gradient flow",
                    "What about training time?",
                ],
                "expected_effects": {"param_direction": "increase", "flops_direction": "increase"},
            },
            "decrease_depth": {
                "question": f"If you reduce the depth of this {architecture}, what changes do you expect?",
                "hints": ["Consider model capacity", "Inference speed?", "Risk of underfitting?"],
                "expected_effects": {"param_direction": "decrease", "flops_direction": "decrease"},
            },
            "increase_width": {
                "question": f"If you increase the channel width (number of filters) of this {architecture}, what happens?",
                "hints": [
                    "More parameters means?",
                    "How does memory scale?",
                    "Effect on expressivity?",
                ],
                "expected_effects": {"param_direction": "increase", "flops_direction": "increase"},
            },
            "decrease_width": {
                "question": f"If you halve the channel widths of this {architecture}, what do you predict?",
                "hints": [
                    "Efficiency vs accuracy tradeoff",
                    "Mobile architectures do this",
                    "Think about MobileNet",
                ],
                "expected_effects": {"param_direction": "decrease", "flops_direction": "decrease"},
            },
            "add_residual": {
                "question": f"If you add residual (skip) connections to this {architecture}, what changes?",
                "hints": [
                    "ResNet's key innovation",
                    "Gradient flow in deep networks",
                    "Identity shortcut effect",
                ],
                "expected_effects": {
                    "param_direction": "no_change",
                    "flops_direction": "no_change",
                },
            },
            "remove_residual": {
                "question": f"If you remove all skip connections from this {architecture}, what do you predict?",
                "hints": [
                    "Effect on training stability",
                    "Gradient vanishing in depth",
                    "Performance on deep networks",
                ],
                "expected_effects": {
                    "param_direction": "no_change",
                    "flops_direction": "no_change",
                },
            },
            "add_attention": {
                "question": f"If you add a Multi-Head Attention block to this {architecture}, what effects do you predict?",
                "hints": [
                    "Global receptive field",
                    "Quadratic complexity (O(N²))",
                    "Memory usage?",
                ],
                "expected_effects": {"param_direction": "increase", "flops_direction": "increase"},
            },
            "change_patch_size": {
                "question": f"If you change the patch size in this {architecture} from 16 to 8, what happens?",
                "hints": [
                    "Number of tokens = (224/patch_size)²",
                    "Quadratic attention scaling",
                    "Resolution vs compute",
                ],
                "expected_effects": {"param_direction": "no_change", "flops_direction": "increase"},
            },
            "change_hidden_dim": {
                "question": f"If you increase the hidden dimension (d_model) in this {architecture}, what do you expect?",
                "hints": [
                    "Model capacity scales with d²",
                    "Transformer expressivity",
                    "Memory footprint?",
                ],
                "expected_effects": {"param_direction": "increase", "flops_direction": "increase"},
            },
        }

        prompt = prompts.get(
            mutation_type,
            {
                "question": f"Predict the effect of '{mutation_type}' on this {architecture} architecture.",
                "hints": ["Consider parameters, FLOPs, and memory"],
                "expected_effects": {},
            },
        )

        return {
            "mutation_type": mutation_type,
            "architecture": architecture,
            **prompt,
        }

    def build_experiment_result(
        self,
        hypothesis: dict[str, Any] | None,
        mutations_applied: list[dict[str, Any]],
        architecture: str,
        actual_diff: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a full experiment result dict, optionally with prediction scoring.
        """
        result_id = str(uuid.uuid4())[:8]
        hypothesis_id = hypothesis.get("id") if hypothesis else None

        # Score if hypothesis provided
        score_data = None
        if hypothesis:
            score_data = self.score_prediction(hypothesis, actual_diff)

        mutation_summary = " + ".join(m.get("type", "?") for m in mutations_applied)

        return {
            "id": result_id,
            "hypothesis_id": hypothesis_id,
            "mutation_summary": mutation_summary,
            "architecture": architecture,
            "mutations_applied": mutations_applied,
            "diff_summary": actual_diff.get("summary_text", ""),
            "param_delta_pct": actual_diff.get("param_delta", {}).get("pct", 0),
            "flops_delta": actual_diff.get("flops_delta", {}).get("absolute", 0),
            "depth_delta": actual_diff.get("depth_delta", {}).get("absolute", 0),
            "nodes_added": actual_diff.get("nodes_added", []),
            "nodes_removed": actual_diff.get("nodes_removed", []),
            "skip_delta": actual_diff.get("skip_delta", {}).get("absolute", 0),
            "prediction_score": score_data["score"] if score_data else None,
            "prediction_feedback": score_data["feedback"] if score_data else None,
            "created_at": datetime.datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _direction(value: float, threshold: float = 0.5) -> str:
    """Convert a numeric change value to 'increase', 'decrease', or 'no_change'."""
    if value > threshold:
        return "increase"
    if value < -threshold:
        return "decrease"
    return "no_change"


# Singleton
hypothesis_engine = HypothesisEngine()
