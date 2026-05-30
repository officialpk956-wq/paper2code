from core.rag.tensor_tracker import TensorTracker, TensorMismatchError
from core.architecture_graph import ArchitectureGraph, GraphNode
import pytest

def test_transformer_operations():
    tracker = TensorTracker()
    graph = ArchitectureGraph(name="Generic-Transformer")
    
    # 1. Reshape with wildcard
    n_reshape = GraphNode(id="r1", type="reshape", label="Reshape", params={"shape": ["B", 196, 768]})
    out = tracker._compute_output_shape(n_reshape, ("B", 3, 224, 224)) # Dummy input, logic uses params
    # Wait, _compute_output_shape uses in_shape. (B, 3, 224, 224) -> 3*224*224 = 150528
    # (B, 196, 768) -> 196*768 = 150528. Correct.
    assert out == ("B", 196, 768)
    
    # 2. Split Heads
    n_split = GraphNode(id="s1", type="split_heads", label="Split", params={"num_heads": 12})
    out = tracker._compute_output_shape(n_split, ("B", 196, 768))
    # 768 / 12 = 64
    assert out == ("B", 12, 196, 64)
    
    # 3. Transpose
    n_trans = GraphNode(id="t1", type="transpose", label="Trans", params={"dims": [1, 2]})
    out = tracker._compute_output_shape(n_trans, ("B", 12, 196, 64))
    assert out == ("B", 196, 12, 64)
    
    # 4. Merge Heads
    n_merge = GraphNode(id="m1", type="merge_heads", label="Merge", params={})
    # Input is (B, 12, 196, 64)
    out = tracker._compute_output_shape(n_merge, ("B", 12, 196, 64))
    assert out == ("B", 196, 768)

def test_transformer_errors():
    tracker = TensorTracker()
    
    # 1. Head divisibility error
    n_split = GraphNode(id="s1", type="split_heads", label="Split", params={"num_heads": 10})
    with pytest.raises(TensorMismatchError, match="not divisible by 10"):
        tracker._compute_output_shape(n_split, ("B", 196, 768))
        
    # 2. Reshape mismatch
    n_reshape = GraphNode(id="r1", type="reshape", label="Reshape", params={"shape": ["B", 100, 100]})
    with pytest.raises(TensorMismatchError, match="Total elements mismatch"):
        tracker._compute_output_shape(n_reshape, ("B", 3, 224, 224))

if __name__ == "__main__":
    test_transformer_operations()
    test_transformer_errors()
    print("Transformer operation tests passed!")
