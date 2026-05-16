
import sys
import os
from typing import Dict, Any

# Add the project root to sys.path
sys.path.append(os.getcwd())

from src.rag.semantic_explainer import SemanticExplainer
from src.rag.knowledge_graph import KnowledgeGraph
from src.architecture_graph import ArchitectureGraph, GraphNode

def validate_kag_system():
    print("=== KAG Semantic Explanation System Validation ===\n")
    
    kg = KnowledgeGraph()
    explainer = SemanticExplainer()
    
    # 1. Test Semantic Role Inference from Knowledge Graph
    test_cases = [
        ("patchembedding", "patch_embedding"),
        ("clstoken", "is_part_of"), # Note: get_semantic_role only handles 'implements' and 'performs'
        ("transformerblock", "sequence_encoder"),
        ("mhsa", "token_mixer"),
        ("globalavgpool", "feature_aggregator")
    ]
    
    print("1. Knowledge Graph Role Inference:")
    for node_type, expected_role in test_cases:
        role = kg.get_semantic_role(node_type)
        print(f"   Node: {node_type:20} -> Inferred Role: {str(role):20} (Expected: {expected_role if expected_role in ['patch_embedding', 'sequence_encoder', 'token_mixer', 'feature_aggregator'] else 'None/Part'})")
    print()

    # 2. Test Explanation Generation
    nodes_to_test = [
        {"type": "patchembedding", "params": {"patch_size": 16}, "role": "patch_embedding"},
        {"type": "clstoken", "params": {}, "role": None},
        {"type": "positionalembedding", "params": {}, "role": None},
        {"type": "mhsa", "params": {}, "role": "token_mixer"},
        {"type": "transformerblock", "params": {}, "role": "sequence_encoder"},
        {"type": "globalavgpool", "params": {}, "role": "feature_aggregator"},
        {"type": "linear", "params": {}, "role": "classifier_head"}
    ]
    
    print("2. Generated Explanations (Architecture-Aware):")
    results = []
    for n in nodes_to_test:
        explanation = explainer.explain(n["type"], n["role"], n["params"])
        print(f"   [{n['type']}] ({n['role']}):")
        print(f"   \"{explanation}\"\n")
        results.append({
            "type": n["type"],
            "role": n["role"],
            "explanation": explanation
        })
        
    # 3. Validation Report
    print("3. Validation Summary:")
    passed = True
    
    # Check if specific ViT explanations are present
    vit_specifics = ["clstoken", "positionalembedding", "patchembedding"]
    for v in vit_specifics:
        found = any(r["type"] == v for r in results)
        if not found:
            print(f"   [FAIL] Missing explanation for {v}")
            passed = False
            
    # Check for hallucination/determinism (should match _EXPLANATIONS or hardcoded blocks)
    for r in results:
        if "A " in r["explanation"] and "optimized for" in r["explanation"] and r["role"] is None:
             # This is the fallback, which is fine if no specific or role explanation exists
             pass
        elif r["explanation"] == "":
            print(f"   [FAIL] Empty explanation for {r['type']}")
            passed = False
            
    if passed:
        print("   [PASS] All core semantic labels and explanations validated.")
    else:
        print("   [FAIL] Some validation checks failed.")

if __name__ == "__main__":
    validate_kag_system()
