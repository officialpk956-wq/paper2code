"""
core/implementation/__init__.py

Phase 10: Research Engineer Mode — Implementation Engine
"""

from core.implementation.code_mapper import (
    get_architecture_implementation,
    get_module_implementation,
)
from core.implementation.cost_estimator import estimate_training_cost
from core.implementation.reproduction_cards import get_reproduction_card
from core.implementation.training_config import get_hyperparameter_explanations, get_training_config

__all__ = [
    "get_module_implementation",
    "get_architecture_implementation",
    "get_training_config",
    "get_hyperparameter_explanations",
    "estimate_training_cost",
    "get_reproduction_card",
]
