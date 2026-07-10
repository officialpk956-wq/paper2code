"""
core/dojo/ — "Implement It Yourself" Code Dojo (Phase 12, M1)

A guided coding environment where learners implement core deep-learning
primitives from scratch (activations, losses, optimizers, layers, metrics)
and get instant, deterministic pass/fail feedback.

Design notes:
  - All exercise content is deterministic (no LLM).
  - Reference solutions live server-side. The API ships only
    {test_inputs, expected_outputs, tolerance} so solutions stay hidden
    until the learner explicitly reveals them.
  - Learner code executes client-side (Pyodide) — this module never runs
    untrusted code. validator.py only ever runs OUR OWN reference solutions
    to precompute expected outputs.
"""

from core.dojo.exercises import (
    EXERCISES,
    get_exercise_list,
    get_public_exercise,
    get_solution,
)

__all__ = [
    "get_exercise_list",
    "get_public_exercise",
    "get_solution",
    "EXERCISES",
]
