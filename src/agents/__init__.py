"""
Agent system interfaces and types.

This package defines the contract for three thin, deterministic agents:
- ParsingAgent: Raw specs → ArchitectureGraph
- VisualizationAgent: Graph → Visual representation
- ExplanationAgent: Facts → Natural language

No implementation, no logic, no behavior.
Pure interfaces and type definitions.
"""

from src.agents.types import (
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

from src.agents.parsing_agent import ParsingAgent
from src.agents.visualization_agent import VisualizationAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.config_parser import ConfigParsingAgent
from src.agents.visualization_agent_impl import DefaultVisualizationAgent
from src.agents.explanation_agent_impl import DefaultExplanationAgent


from src.agents.parsing_agent_impl import ParsingAgentImpl

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
