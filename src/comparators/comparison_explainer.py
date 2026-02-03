"""Human-readable comparative explanations for architecture analysis.

This module generates natural language explanations comparing neural network
architectures based on their computational, spatial, and scaling characteristics.

All explanations are deterministic and rule-based.
"""

from typing import Dict, Any
from src.architecture_graph import ArchitectureGraph
from src.comparators.architecture_comparator import compare_graphs


def explain_compute_difference(
    summary_a: Dict[str, Any],
    summary_b: Dict[str, Any],
    name_a: str,
    name_b: str
) -> str:
    """
    Generate human-readable explanation of computational differences.
    
    Args:
        summary_a: Compute summary from summarize_compute() for first architecture
        summary_b: Compute summary from summarize_compute() for second architecture
        name_a: Name of first architecture
        name_b: Name of second architecture
        
    Returns:
        Multi-line explanation of computational differences
    """
    lines = []
    
    count_a = summary_a["total_high_flops"]
    count_b = summary_b["total_high_flops"]
    bottleneck_a = summary_a["primary_bottleneck"]
    bottleneck_b = summary_b["primary_bottleneck"]
    
    # Overall comparison
    if count_a > count_b:
        if count_b == 0:
            lines.append(f"**{name_a}** is significantly more compute-intensive, with {count_a} high-cost operation(s), while **{name_b}** has no major computational bottlenecks.")
        else:
            lines.append(f"**{name_a}** requires more computation ({count_a} vs {count_b} high-FLOPs nodes).")
    elif count_a < count_b:
        if count_a == 0:
            lines.append(f"**{name_b}** is significantly more compute-intensive, with {count_b} high-cost operation(s), while **{name_a}** has no major computational bottlenecks.")
        else:
            lines.append(f"**{name_b}** requires more computation ({count_b} vs {count_a} high-FLOPs nodes).")
    else:
        if count_a == 0:
            lines.append(f"Both architectures have minimal computational overhead — no major bottlenecks detected.")
        else:
            lines.append(f"Both architectures have similar compute intensity ({count_a} high-FLOPs nodes each).")
    
    # Bottleneck details
    if bottleneck_a:
        lines.append(f"- **{name_a}** primary bottleneck: `{bottleneck_a}`")
    
    if bottleneck_b:
        lines.append(f"- **{name_b}** primary bottleneck: `{bottleneck_b}`")
    
    # Context-specific insights
    if count_a > 1 or count_b > 1:
        lines.append("\n*Multiple high-cost blocks suggest deep or attention-based architecture.*")
    
    return "\n".join(lines)


def explain_spatial_difference(
    summary_a: Dict[str, str],
    summary_b: Dict[str, str],
    name_a: str,
    name_b: str
) -> str:
    """
    Generate human-readable explanation of spatial behavior differences.
    
    Args:
        summary_a: Spatial summary from summarize_spatial_behavior() for first architecture
        summary_b: Spatial summary from summarize_spatial_behavior() for second architecture
        name_a: Name of first architecture
        name_b: Name of second architecture
        
    Returns:
        Multi-line explanation of spatial preservation differences
    """
    lines = []
    
    spatial_a = summary_a["spatial_preservation"]
    spatial_b = summary_b["spatial_preservation"]
    reason_a = summary_a["reason"]
    reason_b = summary_b["reason"]
    
    # Map levels to numeric values for comparison
    spatial_levels = {"high": 3, "medium": 2, "low": 1}
    level_a = spatial_levels.get(spatial_a, 2)
    level_b = spatial_levels.get(spatial_b, 2)
    
    # Overall comparison
    if level_a > level_b:
        lines.append(f"**{name_a}** preserves spatial structure better than **{name_b}**.")
        lines.append(f"- **{name_a}**: {reason_a}")
        lines.append(f"- **{name_b}**: {reason_b}")
    elif level_a < level_b:
        lines.append(f"**{name_b}** preserves spatial structure better than **{name_a}**.")
        lines.append(f"- **{name_b}**: {reason_b}")
        lines.append(f"- **{name_a}**: {reason_a}")
    else:
        lines.append(f"Both architectures have similar spatial preservation ({spatial_a}).")
        lines.append(f"- **{name_a}**: {reason_a}")
        lines.append(f"- **{name_b}**: {reason_b}")
    
    # Add context based on patterns
    if "skip connection" in reason_a.lower() or "skip connection" in reason_b.lower():
        lines.append("\n*Skip connections help recover fine-grained spatial details.*")
    
    if "token" in reason_a.lower() or "token" in reason_b.lower():
        lines.append("\n*Token-based processing sacrifices spatial locality for global context.*")
    
    return "\n".join(lines)


def explain_scaling_difference(
    summary_a: Dict[str, str],
    summary_b: Dict[str, str],
    name_a: str,
    name_b: str
) -> str:
    """
    Generate human-readable explanation of scaling behavior differences.
    
    Args:
        summary_a: Scaling summary from summarize_scaling_behavior() for first architecture
        summary_b: Scaling summary from summarize_scaling_behavior() for second architecture
        name_a: Name of first architecture
        name_b: Name of second architecture
        
    Returns:
        Multi-line explanation of input size scaling differences
    """
    lines = []
    
    scaling_a = summary_a["scaling"]
    scaling_b = summary_b["scaling"]
    reason_a = summary_a["reason"]
    reason_b = summary_b["reason"]
    
    # Map scaling quality to numeric values
    scaling_levels = {"good": 3, "moderate": 2, "poor": 1}
    level_a = scaling_levels.get(scaling_a, 2)
    level_b = scaling_levels.get(scaling_b, 2)
    
    # Overall comparison
    if level_a > level_b:
        lines.append(f"**{name_a}** scales better with increasing input size.")
        lines.append(f"- **{name_a}**: {reason_a}")
        lines.append(f"- **{name_b}**: {reason_b}")
    elif level_a < level_b:
        lines.append(f"**{name_b}** scales better with increasing input size.")
        lines.append(f"- **{name_b}**: {reason_b}")
        lines.append(f"- **{name_a}**: {reason_a}")
    else:
        lines.append(f"Both architectures have similar scaling behavior ({scaling_a}).")
        lines.append(f"- **{name_a}**: {reason_a}")
        lines.append(f"- **{name_b}**: {reason_b}")
    
    # Add context for attention mechanisms
    if "quadratic" in reason_a.lower() or "quadratic" in reason_b.lower():
        lines.append("\n*⚠️ Quadratic attention (O(n²)) becomes prohibitively expensive for large inputs (e.g., high-resolution images).*")
    
    if "linear" in reason_a.lower() or "linear" in reason_b.lower():
        lines.append("\n*Convolutional architectures scale linearly with spatial dimensions, making them efficient for large images.*")
    
    return "\n".join(lines)


def explain_architecture_comparison(
    graph_a: ArchitectureGraph,
    graph_b: ArchitectureGraph
) -> str:
    """
    Generate comprehensive human-readable comparison of two architectures.
    
    This is the main entry point for generating comparative explanations.
    It calls compare_graphs() internally and formats the results into
    readable sections.
    
    Args:
        graph_a: First architecture to compare
        graph_b: Second architecture to compare
        
    Returns:
        Formatted multi-section explanation with markdown headers
    """
    # Get structured comparison
    comparison = compare_graphs(graph_a, graph_b)
    
    name_a = comparison["graph_a"]["name"]
    name_b = comparison["graph_b"]["name"]
    
    # Generate section explanations
    compute_explanation = explain_compute_difference(
        comparison["graph_a"]["compute"],
        comparison["graph_b"]["compute"],
        name_a,
        name_b
    )
    
    spatial_explanation = explain_spatial_difference(
        comparison["graph_a"]["spatial"],
        comparison["graph_b"]["spatial"],
        name_a,
        name_b
    )
    
    scaling_explanation = explain_scaling_difference(
        comparison["graph_a"]["scaling"],
        comparison["graph_b"]["scaling"],
        name_a,
        name_b
    )
    
    # Format as multi-section document
    sections = [
        f"# Architecture Comparison: {name_a} vs {name_b}",
        "",
        "## Computational Cost",
        "",
        compute_explanation,
        "",
        "## Spatial Structure",
        "",
        spatial_explanation,
        "",
        "## Scaling Behavior",
        "",
        scaling_explanation,
        "",
        "---",
        "",
        "### Quick Summary",
        "",
    ]
    
    # Add quick summary bullets from comparison
    for insight in comparison["summary"]:
        sections.append(f"- {insight}")
    
    return "\n".join(sections)
