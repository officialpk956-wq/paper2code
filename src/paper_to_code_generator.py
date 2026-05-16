"""
Paper to Code Generator: PDF/arXiv → Working PyTorch Code

Complete pipeline:
1. Extract text from PDF or fetch from arXiv
2. Classify sections (method, experiments, etc)
3. Extract architecture specification
4. Generate complete PyTorch code
5. Return architecture graph + code + explanation
"""

import re
import json
import inspect
import importlib
import requests
from typing import Optional, Dict, Any

from src.section_splitter import process_text
from src.architecture_extractor import extract_architecture
from src.llm_client import llm_complete, GROQ_API_KEY
from src.orchestrator.pipeline import Paper2CodePipeline
from src.architecture_graph import ArchitectureGraph
from src.codegen import _generate_skeleton


class PaperToCodeGenerator:
    """
    Complete pipeline: Research paper → Runnable PyTorch code + architecture graph.
    """

    def __init__(self):
        self.pipeline = Paper2CodePipeline()
        self.groq_available = bool(GROQ_API_KEY)

    def from_pdf(self, pdf_bytes: bytes, paper_name: str = "paper") -> Dict[str, Any]:
        """
        Process PDF bytes through the complete pipeline.

        Args:
            pdf_bytes: Raw PDF file bytes
            paper_name: Name to use for the paper

        Returns:
            Dict with keys:
            - paper_name: str
            - spec: dict (architecture spec)
            - graph: ArchitectureGraph
            - code: str (complete PyTorch code)
            - explanation: str
            - family: str
            - code_source: "builder" | "llm" | "skeleton"
        """
        # Extract text from PDF
        import io
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber not installed")

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_pages = []
            for page in pdf.pages[:30]:  # Cap at 30 pages
                text = page.extract_text()
                if text:
                    text_pages.append(text)

        raw_text = "\n\n".join(text_pages)
        return self._run_pipeline(raw_text, paper_name)

    def from_arxiv(self, url: str) -> Dict[str, Any]:
        """
        Fetch PDF from arXiv URL and process it.

        Args:
            url: arXiv URL (e.g., https://arxiv.org/abs/1512.03385)

        Returns:
            Same dict as from_pdf()
        """
        # Extract arXiv ID
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
        if not match:
            raise ValueError(
                f"Invalid arXiv URL: {url}. Expected format: "
                "https://arxiv.org/abs/1512.03385"
            )

        arxiv_id = match.group(1)
        paper_name = arxiv_id.replace(".", "_")

        # Fetch PDF from arXiv
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()

        return self.from_pdf(response.content, paper_name)

    def _run_pipeline(self, text: str, paper_name: str) -> Dict[str, Any]:
        """
        Run the full extraction and code generation pipeline.

        Args:
            text: Raw text from PDF
            paper_name: Name for the paper

        Returns:
            Complete result dict with code and graph
        """
        # Step 1: Classify sections
        sections = process_text(text)

        # Step 2: Extract architecture specification
        spec = extract_architecture(sections, paper_name)

        # Step 3: Run through pipeline to get graph + explanation
        pipeline_result = self.pipeline.run_single(spec)
        graph = pipeline_result["graph"]
        explanation = pipeline_result["explanation"]

        # Step 4: Generate complete PyTorch code
        code, code_source = self._generate_code(spec, graph)

        return {
            "paper_name": paper_name,
            "spec": spec,
            "graph": graph,
            "code": code,
            "explanation": explanation,
            "family": spec.get("model_family", "unknown"),
            "code_source": code_source,
        }

    def _generate_code(self, spec: Dict[str, Any], graph: ArchitectureGraph) -> tuple:
        """
        Generate complete PyTorch code from architecture spec.

        Strategy (priority order):
        1. Known family (resnet/unet/vit/transformer) → use actual builder source
        2. Unknown family + GROQ available → LLM generation
        3. Fallback → enhanced skeleton

        Returns:
            (code: str, source: str) where source is "builder" | "llm" | "skeleton"
        """
        family = spec.get("model_family", "").lower()

        # Strategy 1: Known families → Load builder source code
        if family in ("resnet", "unet", "vit", "transformer"):
            try:
                code = self._builder_code(family)
                return code, "builder"
            except Exception as e:
                print(f"Failed to load builder code for {family}: {e}")
                # Fall through to LLM/skeleton

        # Strategy 2: Unknown family + GROQ available → LLM generation
        if self.groq_available:
            try:
                code = self._llm_generate_code(spec, graph)
                # Validate syntax
                import ast
                ast.parse(code)
                return code, "llm"
            except Exception as e:
                print(f"LLM code generation failed: {e}")
                # Fall through to skeleton

        # Strategy 3: Fallback to skeleton
        code = _generate_skeleton(graph)
        return code, "skeleton"

    def _builder_code(self, family: str) -> str:
        """
        Load actual builder source code for known architecture families.

        Maps family name → builder module and class → returns source code.
        """
        builder_map = {
            "resnet": ("src.model_builder", "ResNetBuilder"),
            "unet": ("src.unet_builder", "UNetBuilder"),
            "vit": ("src.vit_builder", "ViTBuilder"),
            "transformer": ("src.transformer_builder", "TransformerBuilder"),
            "yolo": ("src.yolo_builder", "YOLOv3Builder"),
            "ddpm": ("src.ddpm_builder", "DDPMBuilder"),
        }

        if family not in builder_map:
            raise ValueError(f"Unknown family: {family}")

        mod_path, cls_name = builder_map[family]
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        source = inspect.getsource(cls)

        # Prepend schema comment
        header = f"# Auto-generated from research paper\n# Architecture: {family.upper()}\n# Generated by Paper2CodeGenerator\n\n"
        return header + source

    def _llm_generate_code(
        self, spec: Dict[str, Any], graph: ArchitectureGraph
    ) -> str:
        """
        Use LLM to generate complete PyTorch code from architecture spec.

        Provides structured prompt with layers, connections, and schema.
        Returns complete, runnable nn.Module code.
        """
        # Build layer description
        layers_desc = "\n".join(
            [f"  - {n.label} ({n.type}): {n.params}" for n in graph.nodes]
        )

        # Build connections description
        connections_desc = "\n".join(
            [f"  - {e.source} → {e.target} ({e.edge_type})" for e in graph.edges]
        )

        # Build spec excerpt
        spec_json = json.dumps(spec, indent=2)[:2000]

        prompt = f"""You are an expert PyTorch developer. Generate complete, runnable PyTorch code for a neural network architecture extracted from a research paper.

Architecture Name: {spec.get('name', 'Unknown')}
Family: {spec.get('model_family', 'unknown').upper()}

Layers in the network:
{layers_desc}

Connections (data flow):
{connections_desc}

Architecture Specification:
{spec_json}

REQUIREMENTS:
1. Create a complete nn.Module class with proper __init__ and forward() methods
2. Define all layers in __init__ with correct tensor dimensions
3. forward() must handle all data flow including skip/residual connections if present
4. Include necessary imports: torch, torch.nn as nn
5. Add shape comments on key tensor operations for clarity
6. The code must be immediately runnable (can be instantiated and called)
7. Use appropriate PyTorch layer classes (Conv2d, Linear, etc.)

REFERENCE IMPLEMENTATION GUIDE:
- Use `nn.MultiheadAttention` (lowercase 'h') with `embed_dim`, `num_heads`, and `batch_first=True`.
- Use `nn.LayerNorm(normalized_shape)`.
- For classification, use `nn.Linear(in_features, num_classes)`.
- If a 'sequence_pooling' layer is mentioned, implement it using `x.mean(dim=1)` in forward().
- Use standard imports: `import torch`, `import torch.nn as nn`.

Return ONLY valid Python code. No markdown, no triple backticks, no explanation."""

        code = llm_complete(prompt)

        # Remove markdown code blocks if present
        code = re.sub(r"^```python\n", "", code)
        code = re.sub(r"\n```$", "", code)

        return code.strip()
