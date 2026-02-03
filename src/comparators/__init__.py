"""Comparators module for architecture analysis and comparison."""

from src.comparators.architecture_comparator import (
    summarize_compute,
    summarize_spatial_behavior,
    summarize_scaling_behavior,
    compare_graphs,
)
from src.comparators.comparison_explainer import (
    explain_compute_difference,
    explain_spatial_difference,
    explain_scaling_difference,
    explain_architecture_comparison,
)

__all__ = [
    "summarize_compute",
    "summarize_spatial_behavior",
    "summarize_scaling_behavior",
    "compare_graphs",
    "explain_compute_difference",
    "explain_spatial_difference",
    "explain_scaling_difference",
    "explain_architecture_comparison",
]
