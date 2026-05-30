import os
import torch
import re

import logging
import sys
from typing import Dict, Any, List

from core.orchestrator.pipeline import Paper2CodePipeline
from core.paper_to_code_generator import PaperToCodeGenerator
from core.rag.tensor_tracker import TensorTracker, TensorMismatchError
from core.architecture_graph import ArchitectureGraph, GraphNode

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ViT-Benchmark")

class ViTBenchmark:
    def __init__(self):
        self.pipeline = Paper2CodePipeline()
        self.generator = PaperToCodeGenerator()
        self.tracker = TensorTracker()
        self.results = []

    def run_benchmark(self):
        logger.info("================================================================")
        logger.info("STARTING VISION TRANSFORMER VALIDATION BENCHMARK PIPELINE")
        logger.info("================================================================")
        
        # Test Case 1: End-to-End Extraction & Execution
        self.test_valid_vit_e2e()
        
        # Test Case 2: Negative Test - Invalid Embedding Consistency
        self.test_invalid_embed_dim()
        
        # Test Case 3: Negative Test - Invalid Head Divisibility
        self.test_invalid_heads()
        
        # Test Case 4: Negative Test - Malformed Residual Shape Mismatch
        self.test_residual_mismatch()
        
        self.generate_report()

    def test_valid_vit_e2e(self):
        logger.info("TEST 1: End-to-End Engine Validation (Graph -> Tensor -> Code -> Exec)")
        config = {
            "name": "ViT-Base-Benchmark",
            "layers": [
                {"type": "patchembedding", "params": {"patch_size": 16, "embed_dim": 768}},
                {"type": "transformerblock", "params": {"num_heads": 12, "hidden_size": 768}},
                {"type": "sequence_pooling", "params": {}},
                {"type": "linear", "params": {"hidden_size": 1000}}
            ],
            "connections": [
                ["layer_0", "layer_1"], ["layer_1", "layer_2"], ["layer_2", "layer_3"]
            ]
        }



        
        try:
            # 0. Force skeleton for deterministic benchmark
            self.generator.groq_available = False
            
            # 1. Pipeline Execution (Graph + Tensor + Visual + Explanation)
            logger.info(f"  - Config being parsed: {config}")
            result = self.pipeline.run_single(config)
            graph = result["graph"]
            logger.info(f"  - Graph nodes: {[n.type for n in graph.nodes]}")
            logger.info(f"  - Graph & Tensor validation success: {graph.name}")

            
            # 2. Shape Verification
            last_node = graph.nodes[-1]
            logger.info(f"  - Output Shape: {last_node.output_shape}")
            assert last_node.output_shape == ("B", 1000)
            
            # 3. Code Generation
            code, source = self.generator._generate_code(config, graph)
            logger.info(f"  - Code Generated. Source: {source}")
            logger.info(f"  - Generated Code:\n{code}")
            assert source == "skeleton"


            
            # 4. Executable Forward Pass
            import torch.nn as nn
            exec_scope = {'torch': torch, 'nn': nn}
            exec(code, exec_scope)
            
            model_class = None
            # Find the main model class (should match sanitized graph name)
            target_name = re.sub(r"[^A-Za-z0-9]", "", graph.name).lower()
            for key, val in exec_scope.items():
                if isinstance(val, type) and issubclass(val, nn.Module) and key.lower() == target_name:
                    model_class = val
                    break

            
            model = model_class()
            x = torch.randn(1, 3, 224, 224)
            out = model(x)
            logger.info(f"  - Forward pass success. Output shape: {out.shape}")
            assert out.shape == (1, 1000)
            
            self.results.append({"test": "E2E Engine Validation", "status": "PASS"})

            
        except Exception as e:
            logger.error(f"  - Test 1 Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.results.append({"test": "E2E Valid ViT", "status": "FAIL", "error": str(e)})

    def test_invalid_embed_dim(self):
        logger.info("TEST 2: Invalid Embedding Dimension Consistency")
        graph = ArchitectureGraph(name="Mismatch-ViT")
        # patch embed -> 768
        graph.add_node(GraphNode(id="p1", type="patchembedding", label="Patch", params={"patch_size": 16, "embed_dim": 768}))
        # pos embed -> 512 (Mismatch!)
        graph.add_node(GraphNode(id="pos", type="positionalembedding", label="Pos", params={"embed_dim": 512}))
        graph.add_edge("p1", "pos")
        
        try:
            self.tracker.propagate_shapes(graph)
            logger.error("  - Failed: TensorTracker did not catch dimension mismatch")
            self.results.append({"test": "Invalid Embed Dim", "status": "FAIL", "error": "Mismatch not caught"})
        except TensorMismatchError as e:
            logger.info(f"  - Passed: Caught expected mismatch: {e}")
            self.results.append({"test": "Invalid Embed Dim", "status": "PASS"})

    def test_invalid_heads(self):
        logger.info("TEST 3: Invalid Attention Head Divisibility")
        graph = ArchitectureGraph(name="Bad-Heads-ViT")
        # embed 768
        graph.add_node(GraphNode(id="p1", type="patchembedding", label="Patch", params={"patch_size": 16, "embed_dim": 768}))
        # 10 heads (768 not divisible by 10)
        graph.add_node(GraphNode(id="att", type="split_heads", label="Split", params={"num_heads": 10}))
        graph.add_edge("p1", "att")
        
        try:
            self.tracker.propagate_shapes(graph)
            logger.error("  - Failed: TensorTracker did not catch head divisibility error")
            self.results.append({"test": "Invalid Heads", "status": "FAIL", "error": "Indivisibility not caught"})
        except TensorMismatchError as e:
            logger.info(f"  - Passed: Caught expected error: {e}")
            self.results.append({"test": "Invalid Heads", "status": "PASS"})

    def test_residual_mismatch(self):
        logger.info("TEST 4: Malformed Residual Shape Mismatch")
        graph = ArchitectureGraph(name="Residual-Mismatch")
        # Input 768
        graph.add_node(GraphNode(id="in", type="linear", label="In", params={"hidden_size": 768}))
        # Path A: Identity (768)
        # Path B: Projection to 512
        graph.add_node(GraphNode(id="proj", type="linear", label="Proj", params={"hidden_size": 512}))
        # Merge (Mismatch: 768 + 512)
        graph.add_node(GraphNode(id="add", type="residual_add", label="Add", params={}))
        
        graph.add_edge("in", "proj")
        graph.add_edge("proj", "add")
        graph.add_edge("in", "add", edge_type="skip")
        
        try:
            self.tracker.propagate_shapes(graph, initial_shape=("B", 196, 768))
            logger.error("  - Failed: TensorTracker did not catch residual shape mismatch")
            self.results.append({"test": "Residual Mismatch", "status": "FAIL", "error": "Mismatch not caught"})
        except TensorMismatchError as e:
            logger.info(f"  - Passed: Caught expected mismatch: {e}")
            self.results.append({"test": "Residual Mismatch", "status": "PASS"})

    def generate_report(self):
        logger.info("\n" + "="*50)
        logger.info("VIT VALIDATION BENCHMARK REPORT")
        logger.info("="*50)
        
        passed = 0
        for res in self.results:
            status = res["status"]
            icon = "✅" if status == "PASS" else "❌"
            logger.info(f"{icon} {res['test']}: {status}")
            if status == "PASS": passed += 1
            if "error" in res:
                logger.info(f"   Error: {res['error']}")
                
        logger.info("="*50)
        logger.info(f"FINAL SCORE: {passed}/{len(self.results)}")
        logger.info("="*50)
        
        if passed < len(self.results):
            sys.exit(1)

if __name__ == "__main__":
    benchmark = ViTBenchmark()
    benchmark.run_benchmark()
