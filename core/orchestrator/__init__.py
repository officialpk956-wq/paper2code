"""
Orchestration layer for Paper2Code pipeline.

Connects agents in clean orchestration flows:
- run_single: single architecture → graph + visualization + explanation
- run_comparison: two architectures → comparison with visuals and reasoning
"""

from core.orchestrator.pipeline import Paper2CodePipeline

__all__ = ["Paper2CodePipeline"]
