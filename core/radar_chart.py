"""
Multi-architecture radar chart visualization.

F8: Uses Altair to create a radar/spider chart comparing 2-5 architectures
on 4 dimensions: compute efficiency, spatial preservation, scaling, depth.
"""

import math
import altair as alt
import pandas as pd
from core.architecture_graph import ArchitectureGraph
from core.comparators import (
    summarize_compute,
    summarize_spatial_behavior,
    summarize_scaling_behavior,
)


AXES = ["Compute Efficiency", "Spatial Preservation", "Scaling", "Depth"]


def compute_radar_scores(graph: ArchitectureGraph) -> dict:
    """
    Compute normalized scores (0.0-1.0) for a graph on 4 axes.

    Axes:
    - Compute Efficiency: 1 - (high_flops_count / 10), clamped to [0, 1]
    - Spatial Preservation: from summarize_spatial_behavior
    - Scaling: from summarize_scaling_behavior
    - Depth: min(1.0, node_count / 20)

    Args:
        graph: ArchitectureGraph to analyze

    Returns:
        Dict with keys: "name" + 4 axis scores
    """
    compute = summarize_compute(graph)
    spatial = summarize_spatial_behavior(graph)
    scaling = summarize_scaling_behavior(graph)

    # Compute efficiency: lower high_flops count = higher score
    compute_score = max(0.0, 1.0 - compute["total_high_flops"] / 10.0)

    # Spatial preservation: map categorical to numeric
    spatial_map = {"high": 1.0, "medium": 0.5, "low": 0.2}
    spatial_score = spatial_map.get(spatial["spatial_preservation"], 0.5)

    # Scaling: map categorical to numeric
    scaling_map = {"good": 1.0, "moderate": 0.6, "poor": 0.1}
    scaling_score = scaling_map.get(scaling["scaling"], 0.5)

    # Depth: normalize by 20 (typical architecture depth)
    depth_score = min(1.0, len(graph.nodes) / 20.0)

    return {
        "name": graph.name,
        "Compute Efficiency": round(compute_score, 2),
        "Spatial Preservation": round(spatial_score, 2),
        "Scaling": round(scaling_score, 2),
        "Depth": round(depth_score, 2),
    }


def build_radar_chart(records: list) -> alt.Chart:
    """
    Build an Altair radar/spider chart from architecture scores.

    Converts (axis, score, architecture) tuples to (x, y) coordinates
    using polar projection, then renders as a line chart with markers.

    Args:
        records: List of dicts from compute_radar_scores()

    Returns:
        Altair Chart object (radar chart)
    """
    n_axes = len(AXES)
    rows = []

    # Convert each architecture's scores to (x, y) coordinates
    for rec in records:
        name = rec["name"]
        for i, axis in enumerate(AXES):
            # Angle for this axis
            angle = (2 * math.pi * i / n_axes) - math.pi / 2

            # Normalized radius (0.0-1.0)
            r = rec.get(axis, 0.0)

            # Convert polar to Cartesian
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            rows.append({
                "name": name,
                "axis": axis,
                "score": r,
                "x": x,
                "y": y,
                "order": i,
            })

    df = pd.DataFrame(rows)

    # Close the polygon by repeating the first point at the end
    closing = df[df["order"] == 0].copy()
    closing["order"] = n_axes
    df = pd.concat([df, closing], ignore_index=True)

    # Main chart: lines + points
    chart = alt.Chart(df).mark_line(point=True, tooltip=True).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-1.2, 1.2])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.2, 1.2])),
        color=alt.Color("name:N", legend=alt.Legend(title="Architecture")),
        order=alt.Order("order:Q"),
        tooltip=["name:N", "axis:N", "score:Q"],
    ).properties(
        width=400,
        height=400,
        title="Architecture Radar Chart",
    )

    # Axis labels at r=1.25
    label_rows = []
    for i, axis in enumerate(AXES):
        angle = (2 * math.pi * i / n_axes) - math.pi / 2
        label_rows.append({
            "axis": axis,
            "x": 1.25 * math.cos(angle),
            "y": 1.25 * math.sin(angle),
        })

    label_df = pd.DataFrame(label_rows)
    labels = alt.Chart(label_df).mark_text(fontSize=11, align="center").encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        text=alt.Text("axis:N"),
    )

    return (chart + labels).configure_view(strokeWidth=0)
