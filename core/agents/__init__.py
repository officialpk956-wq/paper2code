"""
Agent system interfaces and types.

This package defines the contract for three thin, deterministic agents:
- ParsingAgent: Raw specs → ArchitectureGraph
- VisualizationAgent: Graph → Visual representation
- ExplanationAgent: Facts → Natural language

No implementation, no logic, no behavior.
Pure interfaces and type definitions.
"""

from core.agents.types import (
    # Parsing inputs
    ConfigDict,
    PaperExcerpt,
    SymbolicDesc,
    ParsingSource,
    # Visualization types
    ComparisonContext,
    VisualizationOptions,
    VisualizationMode,
    NodeVisuals,
    VisualRepresentation,
    # Explanation types
    ComputeSummary,
    SpatialSummary,
    ScalingSummary,
    ComparisonResult,
    VisualMetadata,
    ExplanationTemplate,
    TemplateLibrary,
)

from core.agents.parsing_agent import ParsingAgent
from core.agents.visualization_agent import VisualizationAgent
from core.agents.explanation_agent import ExplanationAgent
from core.agents.config_parser import ConfigParsingAgent
from core.agents.visualization_agent_impl import DefaultVisualizationAgent
from core.agents.explanation_agent_impl import DefaultExplanationAgent


from core.agents.parsing_agent_impl import ParsingAgentImpl

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
