"""
Agent system type definitions.

This module defines the data structures that agents accept and return.
No behavior, no logic, no defaults.
"""

from typing import TypedDict, Literal, Union, Optional, Dict, List, Set, Any, Tuple


# ============================================================================
# PARSING AGENT INPUT TYPES
# ============================================================================

class ConfigDict(TypedDict, total=False):
    """Architecture configuration dictionary."""
    name: str
    description: str
    layers: List[Dict[str, Any]]
    connections: List[Tuple[str, str]]
    metadata: Dict[str, Any]


class PaperExcerpt(TypedDict, total=False):
    """Raw text extracted from academic paper."""
    type: Literal["markdown", "text", "pdf_extract"]
    content: str
    source: str  # Paper ID, section name, etc.
    metadata: Dict[str, Any]


class SymbolicDesc(TypedDict, total=False):
    """Symbolic architecture description."""
    type: Literal["symbolic"]
    spec: str  # e.g., "Conv2D(64,3,1)→ReLU→MaxPool→ResBlock(64)×3→..."
    notation: str  # Optional: "paper2code", "custom"
    metadata: Dict[str, Any]


ParsingSource = Union[ConfigDict, PaperExcerpt, SymbolicDesc, Dict[str, Any]]
"""Union of all supported parsing input formats."""


# ============================================================================
# VISUALIZATION AGENT INPUT TYPES
# ============================================================================

class ComparisonContext(TypedDict, total=False):
    """
    Context for comparison-mode visualization.
    
    Used when rendering two architectures side-by-side.
    All fields are optional (None values are valid).
    """
    mode: Literal["compare"]
    current_arch: Literal["A", "B"]
    dominant_compute: Optional[Literal["A", "B"]]
    dominant_spatial: Optional[Literal["A", "B"]]
    scaling_issue: Optional[Literal["A", "B"]]
    bottleneck_node_id: Optional[str]


class VisualizationOptions(TypedDict, total=False):
    """Configuration options for visualization."""
    expand_composite: bool
    include_params: bool
    theme: Literal["light", "dark"]
    rankdir: Literal["TB", "LR"]


VisualizationMode = Literal["single", "compare"]
"""Visualization mode: single architecture or side-by-side comparison."""


# ============================================================================
# VISUALIZATION AGENT OUTPUT TYPES
# ============================================================================

class NodeVisuals(TypedDict, total=False):
    """Visual attributes for a single node."""
    color: str
    penwidth: float
    fillcolor: str
    style: str
    label_suffix: str


class VisualRepresentation(TypedDict, total=False):
    """Complete visual representation of a graph."""
    graphviz_dot: str
    node_annotations: Dict[str, NodeVisuals]
    visual_cues: List[str]
    comparison_mode: bool


# ============================================================================
# EXPLANATION AGENT INPUT TYPES (from existing comparison utilities)
# ============================================================================

class ComputeSummary(TypedDict, total=False):
    """Summary of computational cost analysis."""
    total_high_flops: int
    primary_bottleneck: Optional[str]
    reason: str


class SpatialSummary(TypedDict, total=False):
    """Summary of spatial preservation analysis."""
    spatial_preservation: Literal["high", "medium", "low"]
    reason: str


class ScalingSummary(TypedDict, total=False):
    """Summary of scaling behavior analysis."""
    scaling: Literal["good", "acceptable", "moderate", "poor"]
    reason: str


class ComparisonResult(TypedDict, total=False):
    """Complete comparison result from summarize_* functions."""
    compute_summary_a: ComputeSummary
    compute_summary_b: ComputeSummary
    spatial_summary_a: SpatialSummary
    spatial_summary_b: SpatialSummary
    scaling_summary_a: ScalingSummary
    scaling_summary_b: ScalingSummary


class VisualMetadata(TypedDict, total=False):
    """
    Metadata about visual highlighting from visualization stage.
    
    Allows explanation agent to reference visual cues without recomputing them.
    """
    bottleneck_a: Optional[str]
    bottleneck_b: Optional[str]
    highlighted_nodes_a: Set[str]
    highlighted_nodes_b: Set[str]
    visual_cues_applied: List[str]


# ============================================================================
# EXPLANATION AGENT TEMPLATE TYPES
# ============================================================================

class ExplanationTemplate(TypedDict, total=False):
    """
    A single explanation template mapping semantic fact to text.
    
    Used by ExplanationAgent to ensure deterministic, template-based output.
    """
    semantic_fact: str  # e.g., ("flops", "high"), ("attention", "quadratic")
    template: str       # e.g., "{node} is computationally expensive"


class TemplateLibrary(TypedDict, total=False):
    """
    Collection of explanation templates.
    
    Organized by category (compute, spatial, scaling, etc.)
    """
    semantic_templates: Dict[str, str]
    visual_references: Dict[str, str]
    comparison_templates: Dict[str, str]
    tie_templates: Dict[str, str]
