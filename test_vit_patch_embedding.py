import pytest

def test_vit_patch_embedding_extraction():
    from src.agents.parsing_agent_impl import ParsingAgentImpl
    parsing_agent = ParsingAgentImpl()
    
    vit_config = {
        "name": "ViT-Test",
        "layers": [
            {
                "type": "patchembedding",
                "params": {
                    "patch_size": 16,
                    "embed_dim": 768,
                    "num_patches": 196
                }
            },
            {
                "type": "transformerblock",
                "params": {"hidden_size": 768}
            }
        ],
        "connections": [["layer_0", "layer_1"]]
    }
    
    graph = parsing_agent.parse(vit_config, format_hint="config")
    
    # Check node 0
    patch_node = graph.nodes[0]
    assert patch_node.type == "patchembedding"
    assert patch_node.semantic_params["semantic_role"] == "patch_embedding"
    assert patch_node.semantic_params["patch_size"] == 16
    assert patch_node.semantic_params["num_patches"] == 196
    
    # Check helper method
    info = patch_node.get_patch_info()
    assert info["patch_size"] == 16
    assert info["num_patches"] == 196
    assert info["embed_dim"] == 768

def test_vit_tensor_flow():
    from src.rag.tensor_tracker import TensorTracker, TensorMismatchError
    from src.architecture_graph import ArchitectureGraph, GraphNode
    
    graph = ArchitectureGraph(name="ViT-Flow-Full")
    nodes = [
        GraphNode(id="p", type="patchembedding", label="P", params={"patch_size": 16, "embed_dim": 768}),
        GraphNode(id="c", type="clstoken", label="C", params={}),
        GraphNode(id="pos", type="positionalembedding", label="Pos", params={"embed_dim": 768}),
        GraphNode(id="t", type="transformerblock", label="T", params={"hidden_size": 768}),
    ]
    for n in nodes: graph.add_node(n)
    graph.add_edge("p", "c")
    graph.add_edge("c", "pos")
    graph.add_edge("pos", "t")
    
    tracker = TensorTracker()
    tracker.propagate_shapes(graph, initial_shape=(1, 3, 224, 224))
    
    assert nodes[0].output_shape == (1, 196, 768)  # Patchify
    assert nodes[1].output_shape == (1, 197, 768)  # CLS Insert
    assert nodes[2].output_shape == (1, 197, 768)  # Pos Embed
    assert nodes[3].output_shape == (1, 197, 768)  # Transformer
    
    # Check trace
    assert len(graph.metadata["tensor_trace"]) == 4

def test_invalid_cls_token():
    from src.rag.tensor_tracker import TensorTracker, TensorMismatchError
    from src.architecture_graph import ArchitectureGraph, GraphNode
    
    graph = ArchitectureGraph(name="Invalid-CLS")
    graph.add_node(GraphNode(id="img", type="conv2d", label="I", params={}))
    graph.add_node(GraphNode(id="cls", type="clstoken", label="C", params={}))
    graph.add_edge("img", "cls")
    
    tracker = TensorTracker()
    with pytest.raises(TensorMismatchError, match="CLS Token Error"):
        tracker.propagate_shapes(graph, initial_shape=(1, 3, 224, 224))

def test_invalid_pos_embedding_dim():
    from src.rag.tensor_tracker import TensorTracker, TensorMismatchError
    from src.architecture_graph import ArchitectureGraph, GraphNode
    
    graph = ArchitectureGraph(name="Invalid-Pos")
    graph.add_node(GraphNode(id="p", type="patchembedding", label="P", params={"embed_dim": 768}))
    graph.add_node(GraphNode(id="pos", type="positionalembedding", label="Pos", params={"embed_dim": 512}))
    graph.add_edge("p", "pos")
    
    tracker = TensorTracker()
    with pytest.raises(TensorMismatchError, match="Positional Embedding Error.*Dimension mismatch"):
        tracker.propagate_shapes(graph, initial_shape=(1, 3, 224, 224))

def test_decomposed_transformer_flow():
    from src.rag.tensor_tracker import TensorTracker, TensorMismatchError
    from src.architecture_graph import ArchitectureGraph, GraphNode
    
    graph = ArchitectureGraph(name="Decomposed-Transformer")
    nodes = [
        GraphNode(id="in", type="linear", label="In", params={"hidden_size": 768}),
        GraphNode(id="norm1", type="layernorm", label="Norm", params={}),
        GraphNode(id="q", type="query_projection", label="Q", params={"hidden_size": 768}),
        GraphNode(id="k", type="key_projection", label="K", params={"hidden_size": 768}),
        GraphNode(id="v", type="value_projection", label="V", params={"hidden_size": 768}),
        GraphNode(id="mhsa", type="mhsa", label="MHSA", params={}),
        GraphNode(id="merge", type="attention_merge", label="Merge", params={"hidden_size": 768}),
        GraphNode(id="add1", type="residual_add", label="Add", params={}),
    ]
    for n in nodes: graph.add_node(n)
    
    # Connectivity
    graph.add_edge("in", "norm1")
    graph.add_edge("norm1", "q")
    graph.add_edge("norm1", "k")
    graph.add_edge("norm1", "v")
    graph.add_edge("q", "mhsa")
    graph.add_edge("k", "mhsa")
    graph.add_edge("v", "mhsa")
    graph.add_edge("mhsa", "merge")
    graph.add_edge("merge", "add1")
    graph.add_edge("in", "add1", edge_type="skip")  # Residual skip
    
    tracker = TensorTracker()
    tracker.propagate_shapes(graph, initial_shape=(1, 197, 768))
    
    # Assertions
    assert nodes[5].input_shape == (1, 197, 768)  # MHSA receives 768
    assert nodes[7].input_shape == (1, 197, 768)  # Add receives 768 from both paths
    assert nodes[7].output_shape == (1, 197, 768)

def test_invalid_decomposed_dim():
    from src.rag.tensor_tracker import TensorTracker, TensorMismatchError
    from src.architecture_graph import ArchitectureGraph, GraphNode
    
    graph = ArchitectureGraph(name="Invalid-Decomposed")
    graph.add_node(GraphNode(id="in", type="linear", label="In", params={"hidden_size": 768}))
    graph.add_node(GraphNode(id="q", type="query_projection", label="Q", params={"hidden_size": 512}))
    graph.add_node(GraphNode(id="k", type="key_projection", label="K", params={"hidden_size": 768}))
    graph.add_node(GraphNode(id="mhsa", type="mhsa", label="MHSA", params={}))

    
    graph.add_edge("in", "q")
    graph.add_edge("in", "k")
    graph.add_edge("q", "mhsa")
    graph.add_edge("k", "mhsa")
    
    tracker = TensorTracker()
    # This should fail because MHSA receives mismatched dimensions from Q (512) and K (768)
    with pytest.raises(TensorMismatchError, match="Cannot merge tensors"):
        tracker.propagate_shapes(graph, initial_shape=(1, 197, 768))

