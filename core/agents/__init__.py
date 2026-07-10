"""
Agent system interfaces and types.

This package defines the contract for three thin, deterministic agents:
- ParsingAgent: Raw specs → ArchitectureGraph
- VisualizationAgent: Graph → Visual representation
- ExplanationAgent: Facts → Natural language

No implementation, no logic, no behavior.
Pure interfaces and type definitions.
"""

from core.agents.config_parser import ConfigParsingAgent
from core.agents.explanation_agent import ExplanationAgent
from core.agents.explanation_agent_impl import DefaultExplanationAgent
from core.agents.parsing_agent import ParsingAgent
from core.agents.parsing_agent_impl import ParsingAgentImpl
from core.agents.types import (
    # Visualization types
    ComparisonContext,
    ComparisonResult,
    # Explanation types
    ComputeSummary,
    # Parsing inputs
    ConfigDict,
    ExplanationTemplate,
    NodeVisuals,
    PaperExcerpt,
    ParsingSource,
    ScalingSummary,
    SpatialSummary,
    SymbolicDesc,
    TemplateLibrary,
    VisualizationMode,
    VisualizationOptions,
    VisualMetadata,
    VisualRepresentation,
)
from core.agents.visualization_agent import VisualizationAgent
from core.agents.visualization_agent_impl import DefaultVisualizationAgent

__all__ = [
    # Types
    "ConfigDict",
    "PaperExcerpt",
    "SymbolicDesc",
    "ParsingSource",
    "ComparisonContext",
    "VisualizationOptions",
    "VisualizationMode",
    "NodeVisuals",
    "VisualRepresentation",
    "ComputeSummary",
    "SpatialSummary",
    "ScalingSummary",
    "ComparisonResult",
    "VisualMetadata",
    "ExplanationTemplate",
    "TemplateLibrary",
    # Agents
    "ParsingAgent",
    "VisualizationAgent",
    "ExplanationAgent",
    "ConfigParsingAgent",
    "ParsingAgentImpl",
    "DefaultVisualizationAgent",
    "DefaultExplanationAgent",
]
