import sys
import os

sys.path.append(os.getcwd())

from core.transformer_builder import TransformerBuilder
from core.rag.tensor_tracker import TensorTracker
from core.architecture_graph import ArchitectureGraph

def test_encoder_only_transformer():
    print("=== Testing Encoder-Only Transformer (BERT-like) ===")
    builder = TransformerBuilder("BERT-Like")
    
    # 1. Embeddings
    input_node = builder.add_token_embedding(vocab_size=30522, embed_dim=768)
    builder.add_positional_embedding(max_seq_len=512, embed_dim=768)
    
    # 2. Encoder Blocks
    builder.add_encoder_block(embed_dim=768, num_heads=12, ffn_dim=3072, pre_norm=False)
    builder.add_encoder_block(embed_dim=768, num_heads=12, ffn_dim=3072, pre_norm=False)
    
    # 3. Head
    builder.add_sequence_pooling("mean")
    builder.add_classifier_head(num_classes=2)
    
    graph = builder.get_graph()
    
    print(f"Graph nodes: {len(graph.nodes)}")
    assert len(graph.nodes) > 10, "Graph should have at least 10 nodes for embeddings, 2 encoders, pooling, head"
    
    print("Checking Tensor Propagation...")
    tracker = TensorTracker()
    try:
        # Input shape: (Batch, SeqLen, EmbedDim) since token embedding assumes lookup was done and output is (B, N, D)
        # Actually standard input shape could be B, N for tokens, but token embedding outputs B, N, D
        # So we start tensor tracker propagation after token embedding
        tracker.propagate_shapes(graph, initial_shape=("B", 128, 768))
        print("[PASS] Tensor Propagation successful")
        
        last_node = graph.nodes[-1]
        assert last_node.output_shape == ("B", 2)
        print("[PASS] Classifier head output shape correct")
        
    except Exception as e:
        print(f"[FAIL] Tensor Propagation failed: {e}")

def test_decoder_only_transformer():
    print("\n=== Testing Decoder-Only Transformer (GPT/Llama-like) ===")
    builder = TransformerBuilder("GPT-Like")
    
    input_node = builder.add_token_embedding(vocab_size=50257, embed_dim=1024)
    builder.add_positional_embedding(max_seq_len=2048, embed_dim=1024)
    
    # Add 2 causal decoder blocks
    builder.add_decoder_block(embed_dim=1024, num_heads=16, ffn_dim=4096)
    builder.add_decoder_block(embed_dim=1024, num_heads=16, ffn_dim=4096)
    
    # Next token prediction head
    builder.add_classifier_head(num_classes=50257)
    
    graph = builder.get_graph()
    assert graph.metadata["attention_type"] == "causal"
    print("[PASS] Causal attention metadata correctly tagged.")
    
    tracker = TensorTracker()
    try:
        tracker.propagate_shapes(graph, initial_shape=("B", 512, 1024))
        print("[PASS] Decoder Tensor Propagation successful")
        assert graph.nodes[-1].output_shape == ("B", 512, 50257) # Since sequence pooling is omitted, output is (B, N, Vocab)
        print("[PASS] Next-token head output shape correct")
    except Exception as e:
        print(f"[FAIL] Decoder Tensor Propagation failed: {e}")

def main():
    test_encoder_only_transformer()
    test_decoder_only_transformer()

if __name__ == "__main__":
    main()
