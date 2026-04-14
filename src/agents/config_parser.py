"""
Concrete ParsingAgent implementation for ConfigDict input.

Converts ConfigDict → ArchitectureGraph with semantic_params attached.
Deterministic, type-based rule application, no LLM or inference.
"""

from typing import Dict, Any
from src.architecture_graph import ArchitectureGraph, GraphNode
from src.agents.parsing_agent import ParsingAgent
from src.agents.types import ParsingSource


class ParsingError(Exception):
    """Raised when ConfigDict parsing fails."""
    pass


class ConfigParsingAgent(ParsingAgent):
    """
    Concrete ParsingAgent for ConfigDict input.

    Converts:
    - layers → GraphNode (with complete semantic_params)
    - connections → GraphEdge (edge_type from explicit config only)

    Semantic parameter rules (type-based, deterministic):
    - Conv2D/Conv1D → flops="high", compute_role="feature_extraction"
    - Linear → flops="high", compute_role="projection"
    - MultiHeadAttention/Attention → flops="very high", attention="quadratic", compute_role="attention"
    - MaxPool/AvgPool → feature_map="downsampling", compute_role="pooling"
    - UpSample → feature_map="upsampling", compute_role="reconstruction"
    - BatchNorm/LayerNorm → compute_role="normalization"
    - ReLU/GELU/Sigmoid → compute_role="activation"
    - Residual/ResidualBlock → skip_connection="yes", compute_role="residual"
    - Dropout → compute_role="regularization"
    """

    # Type → semantic_params mapping (deterministic rules, lowercase keys)
    _SEMANTIC_RULES = {
        "conv2d": {"flops": "high", "compute_role": "feature_extraction"},
        "conv1d": {"flops": "high", "compute_role": "feature_extraction"},
        "linear": {"flops": "high", "compute_role": "projection"},
        "multiheadattention": {"flops": "very high", "attention": "quadratic", "compute_role": "attention"},
        "attention": {"flops": "very high", "attention": "quadratic", "compute_role": "attention"},
        "maxpool": {"feature_map": "downsampling", "compute_role": "pooling"},
        "maxpool2d": {"feature_map": "downsampling", "compute_role": "pooling"},
        "avgpool": {"feature_map": "downsampling", "compute_role": "pooling"},
        "avgpool2d": {"feature_map": "downsampling", "compute_role": "pooling"},
        "upsample": {"feature_map": "upsampling", "compute_role": "reconstruction"},
        "batchnorm": {"compute_role": "normalization"},
        "batchnorm2d": {"compute_role": "normalization"},
        "layernorm": {"compute_role": "normalization"},
        "relu": {"compute_role": "activation"},
        "gelu": {"compute_role": "activation"},
        "sigmoid": {"compute_role": "activation"},
        "residual": {"skip_connection": "yes", "compute_role": "residual"},
        "residualblock": {"skip_connection": "yes", "compute_role": "residual"},
        "dropout": {"compute_role": "regularization"},
    }

    # Default semantic_params for all nodes
    _DEFAULT_SEMANTIC_PARAMS = {
        "flops": "low",
        "compute_role": "unknown",
    }

    def parse(
        self,
        source: ParsingSource,
        format_hint: str = "auto"
    ) -> ArchitectureGraph:
        """
        Parse ConfigDict into ArchitectureGraph.

        Args:
            source: Must be ConfigDict (or dict-like with required keys)
            format_hint: Ignored for ConfigDict (always "auto" or explicit "config")

        Returns:
            ArchitectureGraph with nodes (complete semantic_params) and edges

        Raises:
            ParsingError: If source is not a valid ConfigDict
        """
        # Validate input is ConfigDict-like
        if not isinstance(source, dict):
            raise ParsingError(f"Expected dict, got {type(source).__name__}")

        # Extract required fields (do not mutate source)
        name = source.get("name")
        if not name:
            raise ParsingError("ConfigDict must have 'name' field")

        layers = source.get("layers", [])
        if not isinstance(layers, list):
            raise ParsingError("'layers' must be a list")

        connections = source.get("connections", [])
        if not isinstance(connections, list):
            raise ParsingError("'connections' must be a list")

        # Create graph
        graph = ArchitectureGraph(name=name)

        # Parse layers → nodes
        layer_ids = set()
        for layer_idx, layer in enumerate(layers):
            try:
                node = self._layer_to_node(layer, layer_idx)
                graph.add_node(node)
                layer_ids.add(node.id)
            except Exception as e:
                raise ParsingError(f"Failed to parse layer {layer_idx}: {e}")

        # Parse connections → edges
        for conn_idx, connection in enumerate(connections):
            try:
                source_id, target_id, edge_type = self._parse_connection(connection)

                # Validate nodes exist
                if source_id not in layer_ids:
                    raise ParsingError(
                        f"Connection {conn_idx}: source '{source_id}' not found in layers"
                    )
                if target_id not in layer_ids:
                    raise ParsingError(
                        f"Connection {conn_idx}: target '{target_id}' not found in layers"
                    )

                graph.add_edge(source_id, target_id, edge_type=edge_type)
            except Exception as e:
                raise ParsingError(f"Failed to parse connection {conn_idx}: {e}")

        return graph

    def _layer_to_node(self, layer: Dict[str, Any], idx: int) -> GraphNode:
        """
        Convert layer dict to GraphNode with complete semantic_params.

        Args:
            layer: Layer configuration dict (not mutated)
            idx: Layer index (for default ID generation)

        Returns:
            GraphNode with semantic_params attached
        """
        # Extract node properties (do not mutate layer)
        node_id = layer.get("id") or f"layer_{idx}"
        node_type = layer.get("type")
        if not node_type:
            raise ParsingError("Each layer must have a 'type' field")

        node_label = layer.get("label") or node_type
        description = layer.get("description", "")
        params = layer.get("params", {})

        # Apply semantic_params rules: default → type-based → config overrides
        semantic_params = self._compute_semantic_params(node_type, layer)

        # Create node
        node = GraphNode(
            id=node_id,
            type=node_type,
            label=node_label,
            params=params,
            description=description,
            semantic_params=semantic_params,
        )

        return node

    def _compute_semantic_params(
        self,
        layer_type: str,
        layer: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute complete semantic_params for a node.

        Strategy:
        1. Start with defaults
        2. Override with type-based rules
        3. Transfer repeat metadata from params (if present)
        4. Override with explicit config values (if any)

        Args:
            layer_type: Layer type string (e.g., "Conv2D")
            layer: Full layer config dict

        Returns:
            Complete semantic_params dict
        """
        # Step 1: Start with defaults
        result = self._DEFAULT_SEMANTIC_PARAMS.copy()

        # Step 2: Override with type-based rules
        type_lower = layer_type.lower()
        if type_lower in self._SEMANTIC_RULES:
            result.update(self._SEMANTIC_RULES[type_lower])

        # Step 3: Transfer repeat metadata from params to semantic_params
        # Repeat metadata (_repeat_*) lives in params during extraction but
        # should be transferred to semantic_params for explanation/visualization
        params = layer.get("params", {})
        if isinstance(params, dict):
            for key in ["_repeat_group", "_repeat_index", "_repeat_total", "_repeat_truncated"]:
                if key in params:
                    result[key] = params[key]

        # Step 4: Override with explicit config values (if present)
        if "semantic_params" in layer and isinstance(layer["semantic_params"], dict):
            result.update(layer["semantic_params"])

        return result

    def _parse_connection(self, connection: Any) -> tuple:
        """
        Parse connection specification into (source_id, target_id, edge_type).

        Edge type is ONLY assigned if explicitly provided in connection config.
        Otherwise defaults to "flow".

        Args:
            connection: Tuple (source, target) or dict with 'source'/'target'/'edge_type' keys

        Returns:
            Tuple (source_id, target_id, edge_type)
        """
        source_id = None
        target_id = None
        edge_type = "flow"  # Default

        # Parse tuple/list format
        if isinstance(connection, (list, tuple)) and len(connection) >= 2:
            source_id = str(connection[0])
            target_id = str(connection[1])
            if len(connection) >= 3:
                edge_type = str(connection[2])

        # Parse dict format
        elif isinstance(connection, dict):
            source_id = connection.get("source")
            target_id = connection.get("target")
            if "edge_type" in connection:
                edge_type = connection.get("edge_type")

            if source_id:
                source_id = str(source_id)
            if target_id:
                target_id = str(target_id)

        # Validate we have both source and target
        if not source_id or not target_id:
            raise ParsingError(f"Invalid connection format: {connection}")

        return (source_id, target_id, edge_type)
