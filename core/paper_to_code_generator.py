"""
Paper to Code Generator: PDF/arXiv → Working PyTorch Code

Complete pipeline:
1. Extract text from PDF or fetch from arXiv
2. Extract architecture specification via ConfigExtractor (BM25 + few-shot + self-correction)
3. Construct ArchitectureGraph and generate educational explanation
4. Generate complete PyTorch code (known-family builder or LLM/skeleton)
5. Validate code (in-process for trusted builders, E2B sandbox for LLM)
6. Bounded repair loop (up to 3 attempts with diagnostic feedback)
"""

import importlib
import inspect
import json
import pprint
import re
from copy import deepcopy
from typing import Any

import httpx

from core.architecture_extractor import extract_architecture
from core.architecture_graph import ArchitectureGraph
from core.classification import classify_architecture, infer_family_from_name
from core.codegen import _generate_skeleton
from core.llm_client import GROQ_API_KEY, llm_complete
from core.orchestrator.pipeline import Paper2CodePipeline
from core.rag.config_extractor import ConfigExtractor
from core.section_splitter import process_text


class PaperToCodeGenerator:
    """
    Complete pipeline: Research paper → Runnable PyTorch code + architecture graph.
    """

    def __init__(self):
        self.pipeline = Paper2CodePipeline()
        self.config_extractor = ConfigExtractor()
        self.groq_available = bool(GROQ_API_KEY)

    def from_pdf(self, file_obj, paper_name: str = "paper") -> dict[str, Any]:
        """
        Process PDF stream through the complete pipeline.

        Args:
            file_obj: File-like object containing the PDF
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
            - verification_report: dict
            - generation_status: "success" | "needs_review"
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber not installed")

        try:
            with pdfplumber.open(file_obj) as pdf:
                text_pages = []
                for page in pdf.pages[:30]:  # Cap at 30 pages
                    text = page.extract_text()
                    if text:
                        text_pages.append(text)

            raw_text = "\n\n".join(text_pages)
            if not raw_text.strip():
                raise ValueError(
                    "Could not extract any text from the PDF. It might be corrupted, empty, or image-only."
                )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to parse PDF: {str(e)}")

        return self._run_pipeline(raw_text, paper_name)

    def from_arxiv(self, url: str) -> dict[str, Any]:
        """
        Fetch PDF from arXiv URL and process it.

        Args:
            url: arXiv URL (e.g., https://arxiv.org/abs/1512.03385)

        Returns:
            Same dict as from_pdf()
        """
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
        if not match:
            raise ValueError(
                f"Invalid arXiv URL: {url}. Expected format: https://arxiv.org/abs/1512.03385"
            )

        arxiv_id = match.group(1)
        paper_name = arxiv_id.replace(".", "_")

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = httpx.get(pdf_url, timeout=30.0)
        response.raise_for_status()

        import io

        return self.from_pdf(io.BytesIO(response.content), paper_name)

    def _run_pipeline(self, text: str, paper_name: str) -> dict[str, Any]:
        """
        Run the full extraction, code generation, validation, and repair pipeline.

        Args:
            text: Raw text from PDF
            paper_name: Name for the paper

        Returns:
            Complete result dict with code, graph, and verification report
        """
        # Step 1: Primary Extraction via ConfigExtractor (RAG / Few-shot / Normalizer)
        config_dict = None
        legacy_spec: dict[str, Any] | None = None
        try:
            config_dict = self.config_extractor.extract_from_text(text)
        except Exception as e:
            print(f"ConfigExtractor encountered an error: {e}")

        # Fallback to legacy section splitter + extract_architecture if ConfigExtractor produced empty layers
        if not config_dict or not config_dict.get("layers"):
            sections = process_text(text)
            legacy_spec = extract_architecture(sections, paper_name)
            if not legacy_spec.get("model_family") and not legacy_spec.get("stages"):
                raise ValueError("No architecture could be detected.")
            config_dict = self._spec_to_config_dict(legacy_spec, paper_name)

        # Step 2: Build ArchitectureGraph and explanation
        pipeline_result = self.pipeline.run_single(config_dict)
        graph = pipeline_result["graph"]
        if len(graph.nodes) < 2:
            raise ValueError("No architecture could be detected.")
        explanation = pipeline_result["explanation"]

        # Step 3: Classify architecture family deterministically
        classified_family = classify_architecture(graph)
        legacy_family = None
        if legacy_spec is not None:
            legacy_family = legacy_spec.get("model_family") or legacy_spec.get("family")
            if str(legacy_family or "").strip().lower() in ("", "unknown", "none"):
                legacy_family = None
            elif legacy_family:
                legacy_family = infer_family_from_name(str(legacy_family)) or str(
                    legacy_family
                ).strip().lower()

        family = (
            legacy_family
            or infer_family_from_name(config_dict.get("name"))
            or infer_family_from_name(paper_name)
            or classified_family
            or "unknown"
        )
        family = infer_family_from_name(str(family)) or str(family).strip().lower()
        spec = legacy_spec or self._config_dict_to_builder_spec(
            config_dict, family, paper_name
        )
        spec["model_family"] = family
        spec["family"] = family

        # Step 4: Generate PyTorch code
        code, code_source = self._generate_code(spec, graph)
        verification_report = self.validate_generated_code(code, code_source, spec)

        # Step 5: Bounded repair loop (max 3 total attempts)
        attempts = [self._verification_attempt(1, code_source, verification_report)]

        current_attempt = 1
        max_attempts = 3

        while (
            not verification_report.get("passed")
            and current_attempt < max_attempts
            and self.groq_available
        ):
            current_attempt += 1
            repaired_code = self._repair_code(
                code=code,
                verification_report=verification_report,
                spec=spec,
                graph=graph,
                attempt=current_attempt,
            )
            if not repaired_code or repaired_code.strip() == code.strip():
                break

            code = repaired_code
            code_source = "llm" if code_source == "builder" else code_source
            verification_report = self.validate_generated_code(code, code_source, spec)
            attempts.append(
                self._verification_attempt(current_attempt, code_source, verification_report)
            )
            if verification_report.get("passed"):
                break

        verification_report["attempts"] = attempts
        verification_report["total_attempts"] = len(attempts)
        verification_report["final_attempt"] = len(attempts)

        return {
            "paper_name": paper_name,
            "spec": spec,
            "graph": graph,
            "code": code,
            "explanation": explanation,
            "family": family,
            "code_source": code_source,
            "verification_report": verification_report,
            "generation_status": (
                "success" if verification_report.get("passed") else "needs_review"
            ),
        }

    @staticmethod
    def _verification_attempt(
        attempt: int, code_source: str, report: dict[str, Any]
    ) -> dict[str, Any]:
        """Return a compact index plus the complete diagnostic for one validation."""
        diagnostic = deepcopy(report)
        diagnostic.pop("attempts", None)
        return {
            "attempt": attempt,
            "code_source": code_source,
            "passed": bool(report.get("passed")),
            "status": report.get("status"),
            "stage": report.get("stage"),
            "error": report.get("error"),
            "checks": deepcopy(report.get("checks")),
            "input_shape": deepcopy(report.get("input_shape")),
            "output_shape": deepcopy(report.get("output_shape")),
            "entrypoint_class": report.get("entrypoint_class"),
            "report": diagnostic,
        }

    def _config_dict_to_builder_spec(
        self, config_dict: dict[str, Any], family: str, paper_name: str
    ) -> dict[str, Any]:
        """
        Synthesize a builder-ready specification from an extracted ConfigDict.
        """
        layers = config_dict.get("layers") or []
        name = config_dict.get("name") or paper_name

        spec: dict[str, Any] = {
            "name": name,
            "model_family": family,
            "family": family,
            "layers": layers,
            "connections": config_dict.get("connections") or [],
        }

        stem_params: dict[str, Any] = {}
        block_params: dict[str, Any] = {}
        stages: list[dict[str, Any]] = []
        input_params: dict[str, Any] = {}
        output_params: dict[str, Any] = {}

        for layer in layers:
            l_type = (layer.get("type") or "").lower()
            params = layer.get("params") or {}

            if "conv" in l_type:
                ch = params.get("channels") or params.get("out_channels")
                if ch and "out_channels" not in stem_params:
                    stem_params["out_channels"] = ch
                k = params.get("kernel_size") or params.get("kernel")
                if k:
                    stem_params["kernel"] = k
                if "stride" in params and params["stride"] is not None:
                    stem_params["stride"] = params["stride"]
                if "padding" in params and params["padding"] is not None:
                    stem_params["padding"] = params["padding"]

            elif l_type == "patchembedding":
                stem_params["patch_size"] = params.get("patch_size") or 16
                stem_params["embed_dim"] = params.get("embed_dim") or 192
                stem_params["in_channels"] = params.get("in_channels") or 3
                stem_params["num_patches"] = params.get("num_patches") or 196

            elif l_type in ("multiheadattention", "transformerblock"):
                if "num_heads" in params and params["num_heads"] is not None:
                    block_params["num_heads"] = params["num_heads"]
                d_model = (
                    params.get("d_model")
                    or params.get("embed_dim")
                    or params.get("hidden_size")
                )
                if d_model:
                    block_params["d_model"] = d_model
                    stem_params["d_model"] = d_model
                if "ffn_dim" in params and params["ffn_dim"] is not None:
                    block_params["ffn_dim"] = params["ffn_dim"]

            elif l_type in ("residualblock", "bottleneckblock"):
                ch = params.get("channels") or params.get("out_channels")
                if ch:
                    stages.append({"out_channels": ch, "num_blocks": 1, "repeats": 1})

            elif l_type == "linear":
                num_classes = (
                    params.get("num_classes")
                    or params.get("channels")
                    or params.get("hidden_size")
                )
                if num_classes:
                    output_params["num_classes"] = num_classes

        if stem_params:
            spec["stem"] = {"params": stem_params}
        if block_params:
            spec["block"] = {"params": block_params}
        if stages:
            spec["stages"] = stages
        if input_params:
            spec["input"] = input_params
        if output_params:
            spec["output"] = output_params

        return spec

    _STEM_TYPE_MAP = {
        "conv": "conv2d",
        "patch_embed": "patchembedding",
    }
    _BLOCK_TYPE_MAP = {
        "bottleneck": "bottleneckblock",
        "bottleneck_residual": "bottleneckblock",
        "basic": "residualblock",
        "residual": "residualblock",
        "transformer": "transformerblock",
        "transformer_encoder": "transformerblock",
    }
    _HEAD_TYPE_MAP = {
        "global_average_pooling": "avgpool2d",
        "gap": "avgpool2d",
        "avgpool": "avgpool2d",
        "fc": "linear",
        "dense": "linear",
    }

    def _spec_to_config_dict(self, spec: dict[str, Any], paper_name: str) -> dict[str, Any]:
        """
        Translate a BASE_MODEL_SCHEMA-shaped spec (model_family/stem/block/
        stages/head — see core/schemas_base.py) into the ConfigDict format
        ParsingAgentImpl/ConfigParsingAgent consume: a flat
        {"name", "layers", "connections"} shape.
        """
        layers: list[dict[str, Any]] = []

        stem = spec.get("stem") or {}
        stem_type = stem.get("type")
        if stem_type:
            layers.append(
                {
                    "type": self._STEM_TYPE_MAP.get(stem_type, stem_type),
                    "params": stem.get("params") or {},
                }
            )

        block = spec.get("block") or {}
        block_type = block.get("type")
        mapped_block_type = self._BLOCK_TYPE_MAP.get(block_type, block_type or "block")
        for stage in spec.get("stages") or []:
            num_blocks = stage.get("num_blocks") or stage.get("repeats") or 1
            stage_params = {
                **(block.get("params") or {}),
                **{k: v for k, v in stage.items() if k not in ("num_blocks", "repeats")},
            }
            for _ in range(max(1, int(num_blocks))):
                layers.append({"type": mapped_block_type, "params": dict(stage_params)})

        head = spec.get("head") or {}
        head_type = head.get("type")
        mapped_head_type = self._HEAD_TYPE_MAP.get(head_type, head_type)
        if head_type:
            layers.append({"type": mapped_head_type, "params": head.get("params") or {}})

        output = spec.get("output") or {}
        if output.get("num_classes") and mapped_head_type != "linear":
            layers.append({"type": "linear", "params": {"channels": output["num_classes"]}})

        while len(layers) < 2:
            layers.append({"type": "identity", "params": {}})

        connections = [[f"layer_{i}", f"layer_{i + 1}"] for i in range(len(layers) - 1)]
        name = spec.get("variant") or spec.get("model_family") or paper_name

        return {"name": str(name), "layers": layers, "connections": connections}

    def _generate_code(self, spec: dict[str, Any], graph: ArchitectureGraph) -> tuple:
        """
        Generate complete PyTorch code from architecture spec.

        Strategy (priority order):
        1. Known family (resnet/unet/vit/transformer) → use actual builder source
        2. Unknown family + GROQ available → LLM generation
        3. Fallback → enhanced skeleton

        Returns:
            (code: str, source: str) where source is "builder" | "llm" | "skeleton"
        """
        family = str(spec.get("model_family") or spec.get("family") or "").strip().lower()

        # Strategy 1: Known families → Load self-contained builder source code
        if family in ("resnet", "unet", "vit", "transformer"):
            try:
                code = self._builder_code(family, spec)
                return code, "builder"
            except Exception as e:
                print(f"Failed to load builder code for {family}: {e}")

        # Strategy 2: Unknown family + GROQ available → LLM generation
        if self.groq_available:
            try:
                code = self._llm_generate_code(spec, graph)
                import ast

                ast.parse(code)
                return code, "llm"
            except Exception as e:
                print(f"LLM code generation failed: {e}")

        # Strategy 3: Fallback to skeleton
        code = _generate_skeleton(graph)
        return code, "skeleton"

    def _builder_code(self, family: str, spec: dict[str, Any] | None = None) -> str:
        """
        Load actual builder source code for known architecture families.
        """
        builder_map = {
            "resnet": ("core.model_builder", "ResNetBuilder"),
            "unet": ("core.unet_builder", "UNetBuilder"),
            "vit": ("core.vit_builder", "ViTBuilder"),
            "transformer": ("core.transformer_model_builder", "TransformerModelBuilder"),
            "yolo": ("core.yolo_builder", "YOLOv3Builder"),
            "ddpm": ("core.ddpm_builder", "DDPMBuilder"),
        }

        if family not in builder_map:
            raise ValueError(f"Unknown family: {family}")

        if family in ("resnet", "unet", "vit", "transformer"):
            return self._self_contained_builder_code(family, spec or {})

        mod_path, cls_name = builder_map[family]
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        source = inspect.getsource(cls)

        header = f"# Auto-generated from research paper\n# Architecture: {family.upper()}\n# Generated by Paper2CodeGenerator\n\n"
        return header + source

    def _self_contained_builder_code(self, family: str, spec: dict[str, Any]) -> str:
        """Return a standalone PyTorch module for a supported known family."""
        from core.blocks_resnet import Bottleneck
        from core.blocks_transformer import TransformerEncoderBlock
        from core.blocks_unet import DoubleConv
        from core.blocks_vit import PatchEmbedding
        from core.model_builder import ResNetBuilder
        from core.transformer_model_builder import TransformerModelBuilder
        from core.unet_builder import UNetBuilder
        from core.vit_builder import ViTBuilder

        prepared = self._prepare_builder_schema(family, spec)
        dependencies: dict[str, list[type]] = {
            "resnet": [Bottleneck, ResNetBuilder],
            "unet": [DoubleConv, UNetBuilder],
            "vit": [PatchEmbedding, TransformerEncoderBlock, ViTBuilder],
            "transformer": [TransformerEncoderBlock, TransformerModelBuilder],
        }
        entrypoints = {
            "resnet": "ResNetBuilder",
            "unet": "UNetBuilder",
            "vit": "ViTBuilder",
            "transformer": "TransformerModelBuilder",
        }

        class_source = "\n\n".join(inspect.getsource(cls) for cls in dependencies[family])
        default_schema = pprint.pformat(prepared, width=100, sort_dicts=False)
        entrypoint = entrypoints[family]

        if family in ("resnet", "unet", "vit"):
            channels = int((prepared.get("input") or {}).get("channels") or 3)
            spatial = (prepared.get("input") or {}).get("spatial_dims") or (
                [256, 256] if family == "unet" else [224, 224]
            )
            height, width = int(spatial[0]), int(spatial[1])
            test_input_expr = f"torch.randn(1, {channels}, {height}, {width})"
        else:
            seq_len = int((prepared.get("input") or {}).get("seq_len") or 64)
            vocab_size = int((prepared.get("input") or {}).get("vocab_size") or 10000)
            test_input_expr = f"torch.randint(0, {vocab_size}, (1, {seq_len}))"

        return (
            "# Auto-generated from a research paper\n"
            f"# Architecture: {family.upper()}\n\n"
            "import math\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "from torch.nn import functional as F\n\n"
            f"DEFAULT_SCHEMA = {default_schema}\n\n"
            f"{class_source}\n\n"
            "if __name__ == '__main__':\n"
            f"    model = {entrypoint}(DEFAULT_SCHEMA).eval()\n"
            "    with torch.no_grad():\n"
            f"        output = model({test_input_expr})\n"
            "    print(tuple(output.shape))\n"
        )

    @staticmethod
    def _prepare_builder_schema(family: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Fill missing builder fields while retaining extracted paper values."""
        if family == "resnet":
            default = {
                "model_family": "resnet",
                "input": {"channels": 3, "spatial_dims": [224, 224]},
                "output": {"num_classes": 1000},
                "stem": {"params": {"out_channels": 64}},
                "stages": [
                    {
                        "num_blocks": 3,
                        "in_channels": 64,
                        "out_channels": 64,
                        "expansion": 4,
                        "stride": 1,
                        "downsample": True,
                    },
                    {
                        "num_blocks": 4,
                        "in_channels": 256,
                        "out_channels": 128,
                        "expansion": 4,
                        "stride": 2,
                        "downsample": True,
                    },
                    {
                        "num_blocks": 6,
                        "in_channels": 512,
                        "out_channels": 256,
                        "expansion": 4,
                        "stride": 2,
                        "downsample": True,
                    },
                    {
                        "num_blocks": 3,
                        "in_channels": 1024,
                        "out_channels": 512,
                        "expansion": 4,
                        "stride": 2,
                        "downsample": True,
                    },
                ],
            }
        elif family == "unet":
            default = {
                "model_family": "unet",
                "input": {"channels": 3, "spatial_dims": [256, 256]},
                "output": {"num_classes": 2},
                "encoder": [64, 128, 256, 512],
                "bottleneck": 1024,
                "decoder": [512, 256, 128, 64],
            }
        elif family == "transformer":
            default = {
                "model_family": "transformer",
                "input": {"vocab_size": 10000, "max_seq_len": 512, "seq_len": 64},
                "output": {"num_classes": 1000},
                "stem": {"params": {"d_model": 512}},
                "block": {
                    "params": {
                        "d_model": 512,
                        "num_heads": 8,
                        "ffn_dim": 2048,
                        "dropout": 0.1,
                    }
                },
                "stages": [{"repeats": 6}],
            }
        else:
            default = {
                "model_family": "vit",
                "input": {"channels": 3, "spatial_dims": [224, 224]},
                "output": {"num_classes": 1000},
                "stem": {
                    "params": {
                        "in_channels": 3,
                        "patch_size": 16,
                        "embed_dim": 192,
                        "num_patches": 196,
                    }
                },
                "block": {
                    "params": {
                        "d_model": 192,
                        "num_heads": 3,
                        "ffn_dim": 768,
                        "dropout": 0.1,
                    }
                },
                "stages": [{"repeats": 2}],
            }

        merged = dict(default)
        if spec:
            for key, value in spec.items():
                if value not in (None, [], {}):
                    if isinstance(value, dict) and isinstance(merged.get(key), dict):
                        merged[key] = {
                            **merged[key],
                            **{k: v for k, v in value.items() if v is not None},
                        }
                    else:
                        merged[key] = value

        if family in ("resnet", "vit", "transformer"):
            extracted_stages = merged.get("stages") or []
            if extracted_stages:
                default_stages = default["stages"]
                merged_stages = []
                for i, stage in enumerate(extracted_stages):
                    if isinstance(stage, dict):
                        tpl = default_stages[i % len(default_stages)]
                        merged_stages.append(
                            {**tpl, **{k: v for k, v in stage.items() if v is not None}}
                        )
                merged["stages"] = merged_stages or default_stages
            else:
                merged["stages"] = default["stages"]

        if family == "resnet":
            user_stem = merged.get("stem") or {}
            stem = {
                **default["stem"],
                **{k: v for k, v in user_stem.items() if v is not None},
            }
            user_params = stem.get("params") or {}
            stem["params"] = {
                **default["stem"]["params"],
                **{k: v for k, v in user_params.items() if v is not None},
            }
            merged["stem"] = stem
        elif family in ("vit", "transformer"):
            for key in ("stem", "block"):
                user_sec = merged.get(key) or {}
                section = {
                    **default[key],
                    **{k: v for k, v in user_sec.items() if v is not None},
                }
                user_params = section.get("params") or {}
                section["params"] = {
                    **default[key]["params"],
                    **{k: v for k, v in user_params.items() if v is not None},
                }
                merged[key] = section

        user_input = merged.get("input") or {}
        merged["input"] = {
            **default["input"],
            **{k: v for k, v in user_input.items() if v is not None},
        }

        user_output = merged.get("output") or {}
        merged["output"] = {
            **default["output"],
            **{k: v for k, v in user_output.items() if v is not None},
        }
        if not (merged.get("output") or {}).get("num_classes"):
            merged["output"]["num_classes"] = default["output"]["num_classes"]

        return merged

    def validate_generated_code(
        self, code: str, code_source: str, spec: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run syntax and execution verification (in-process for trusted builders,
        E2B sandbox for untrusted LLM/skeleton code).
        """
        import ast

        family = str(
            spec.get("model_family") or spec.get("family") or "unknown"
        ).strip().lower()
        report: dict[str, Any] = {
            "phase": 2,
            "passed": False,
            "checks": {"syntax": False, "exec": False, "forward": False},
            "family": family,
            "code_source": code_source,
        }

        try:
            ast.parse(code)
            report["checks"]["syntax"] = True
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"SyntaxError: {exc}"
            return report

        # Strategy A: Builder code for known families → in-process execution verification
        if code_source == "builder" and family in ("resnet", "unet", "vit", "transformer"):
            try:
                namespace: dict[str, Any] = {"__name__": "paper2code_generated"}
                exec(compile(code, "<paper2code-generated>", "exec"), namespace)
                report["checks"]["exec"] = True

                entrypoint = {
                    "resnet": "ResNetBuilder",
                    "unet": "UNetBuilder",
                    "vit": "ViTBuilder",
                    "transformer": "TransformerModelBuilder",
                }[family]
                schema = namespace["DEFAULT_SCHEMA"]
                model = namespace[entrypoint](schema).eval()

                if family in ("resnet", "unet", "vit"):
                    channels = int((schema.get("input") or {}).get("channels") or 3)
                    spatial = (schema.get("input") or {}).get("spatial_dims") or (
                        [256, 256] if family == "unet" else [224, 224]
                    )
                    input_shape = [1, channels, int(spatial[0]), int(spatial[1])]
                    test_input = namespace["torch"].randn(*input_shape)
                else:
                    seq_len = int((schema.get("input") or {}).get("seq_len") or 64)
                    vocab_size = int((schema.get("input") or {}).get("vocab_size") or 10000)
                    input_shape = [1, seq_len]
                    test_input = namespace["torch"].randint(0, vocab_size, (1, seq_len))

                torch = namespace["torch"]
                with torch.no_grad():
                    output = model(test_input)
                output_shape = list(output.shape)

                expected_classes = int(
                    (schema.get("output") or {}).get("num_classes") or 1000
                )
                if family in ("resnet", "vit", "transformer") and output_shape != [
                    1,
                    expected_classes,
                ]:
                    raise ValueError(f"Unexpected classification output shape: {output_shape}")
                if family == "unet" and output_shape[-2:] != input_shape[-2:]:
                    raise ValueError(f"U-Net did not preserve spatial dimensions: {output_shape}")

                report.update(
                    {
                        "passed": True,
                        "status": "success",
                        "entrypoint_class": entrypoint,
                        "input_shape": input_shape,
                        "output_shape": output_shape,
                    }
                )
                report["checks"]["forward"] = True
            except Exception as exc:
                report["status"] = "needs_review"
                report["error"] = f"{type(exc).__name__}: {exc}"
            return report

        # Strategy B: Untrusted LLM or skeleton code → E2B sandboxed execution.
        # Runs a real forward pass and checks the output shape, not just
        # "did the class instantiate" -- an earlier version only checked
        # instantiation and unconditionally marked forward=True.
        report["stage"] = "sandbox"
        try:
            from backend.services.e2b_service import run_code_in_sandbox

            candidates = self._e2b_test_input_candidates(spec)
            expected_classes = (spec.get("output") or {}).get("num_classes")

            # Must run BEFORE `code` -- generated code almost always has its
            # own top-level `import torch`, which would raise immediately if
            # placed after `code` in file order (Python executes top to
            # bottom; a bootstrap appended after `code` never gets reached).
            bootstrap = (
                "import sys, subprocess\n"
                "\n"
                "# The default E2B sandbox template has numpy/pandas but not\n"
                "# torch (a ~2GB dependency) -- install it on first use, same\n"
                "# pattern as backend/services/pytorch_parser.py.\n"
                "try:\n"
                "    import torch  # noqa: F401\n"
                "except ImportError:\n"
                "    subprocess.run(\n"
                "        [sys.executable, '-m', 'pip', 'install', 'torch',\n"
                "         '--index-url', 'https://download.pytorch.org/whl/cpu', '--quiet'],\n"
                "        check=True,\n"
                "    )\n"
            )

            harness = (
                "import json\n"
                "import torch\n"
                "import torch.nn as nn\n"
                "\n"
                "result = {'ok': False, 'error': None, 'output_shape': None,\n"
                "          'class_name': None, 'input_used': None}\n"
                "try:\n"
                "    module_classes = [\n"
                "        cls for _name, cls in list(globals().items())\n"
                "        if isinstance(cls, type) and issubclass(cls, nn.Module)\n"
                "        and cls.__module__ == '__main__'\n"
                "    ]\n"
                "    if not module_classes:\n"
                "        raise RuntimeError('No nn.Module subclass found in generated code')\n"
                "    target_cls = module_classes[-1]\n"
                "    result['class_name'] = target_cls.__name__\n"
                "    model = target_cls()\n"
                "    model.eval()\n"
                "\n"
                f"    candidates = {candidates!r}\n"
                "    last_err = None\n"
                "    output = None\n"
                "    used_input = None\n"
                "    for expr in candidates:\n"
                "        try:\n"
                "            test_input = eval(expr)\n"
                "            with torch.no_grad():\n"
                "                output = model(test_input)\n"
                "            used_input = expr\n"
                "            break\n"
                "        except Exception as e:\n"
                "            last_err = e\n"
                "            continue\n"
                "    if output is None:\n"
                "        raise last_err or RuntimeError('No candidate input shape worked')\n"
                "\n"
                "    result['ok'] = True\n"
                "    result['output_shape'] = list(output.shape)\n"
                "    result['input_used'] = used_input\n"
                "except Exception as e:\n"
                "    result['error'] = f'{type(e).__name__}: {e}'\n"
                "\n"
                "print('__PAPER2CODE_RESULT__' + json.dumps(result))\n"
                "sys.exit(0 if result['ok'] else 1)\n"
            )
            sandbox_code = bootstrap + "\n\n" + code + "\n\n" + harness
            # Cold-start pip install of torch can take 2-4 minutes; this only
            # runs inside the async Celery task, never the sync upload path.
            sandbox_res = run_code_in_sandbox(sandbox_code, run_timeout_ms=300_000)
            report["sandbox"] = {
                "exit_code": sandbox_res.get("exit_code"),
                "time_ms": sandbox_res.get("time_ms"),
                "failure_kind": None,
            }

            parsed_result = None
            for line in (sandbox_res.get("stdout") or "").splitlines():
                if line.startswith("__PAPER2CODE_RESULT__"):
                    try:
                        parsed_result = json.loads(line[len("__PAPER2CODE_RESULT__") :])
                    except Exception:
                        pass
                    break

            if parsed_result and parsed_result.get("ok"):
                report["checks"]["exec"] = True
                report["checks"]["forward"] = True
                output_shape = parsed_result.get("output_shape")
                report["output_shape"] = output_shape
                report["entrypoint_class"] = parsed_result.get("class_name")
                if (
                    expected_classes
                    and output_shape
                    and output_shape != [1, int(expected_classes)]
                    and len(output_shape) == 2
                ):
                    report["status"] = "needs_review"
                    report["error"] = (
                        f"Unexpected classification output shape: {output_shape}, "
                        f"expected [1, {int(expected_classes)}]"
                    )
                else:
                    report["passed"] = True
                    report["status"] = "success"
            elif parsed_result:
                # Ran, but instantiation or the forward pass failed -- exec
                # happened, just not successfully.
                report["checks"]["exec"] = True
                report["status"] = "needs_review"
                report["error"] = parsed_result.get("error") or "Sandbox validation failed"
                report["sandbox"]["failure_kind"] = "runtime"
            else:
                report["status"] = "needs_review"
                sandbox_error = (
                    sandbox_res.get("stderr")
                    or sandbox_res.get("stdout")
                    or "Sandbox execution failed"
                )
                report["error"] = sandbox_error
                report["sandbox"]["failure_kind"] = self._sandbox_failure_kind(
                    sandbox_error
                )
        except Exception as exc:
            report["status"] = "needs_review"
            report["error"] = f"E2BError: {exc}"
            report["sandbox"] = {
                "exit_code": None,
                "time_ms": None,
                "failure_kind": self._sandbox_failure_kind(str(exc)),
            }

        return report

    @staticmethod
    def _sandbox_failure_kind(message: str) -> str:
        """Classify expected sandbox failures for repair logic and UI diagnostics."""
        lowered = str(message or "").lower()
        if "time limit" in lowered or "timed out" in lowered or "timeout" in lowered:
            return "timeout"
        if "not configured" in lowered or "api key" in lowered:
            return "configuration"
        if any(token in lowered for token in ("out of memory", "memory limit", "killed", "resource limit")):
            return "resource_limit"
        if any(token in lowered for token in ("network", "connection", "dns", "name resolution")):
            return "network"
        return "runtime"

    @staticmethod
    def _e2b_test_input_candidates(spec: dict[str, Any]) -> list[str]:
        """
        Build candidate synthetic-input expressions from the extracted spec.
        Unknown/LLM-generated code's family isn't known to be image-like or
        sequence-like ahead of time, so try each shape the spec's input hints
        at (or both common defaults if the spec has neither).
        """
        inp = spec.get("input") or {}
        candidates: list[str] = []

        channels = inp.get("channels")
        spatial = inp.get("spatial_dims")
        if channels or spatial:
            c = int(channels or 3)
            # spatial_dims can legitimately come back as a bare int/float
            # (e.g. the LLM describing a single "224x224" as just 224)
            # rather than a [h, w] list -- dims[0] on a scalar raised
            # "'int' object is not subscriptable" and crashed validation
            # for every repair attempt identically, since this runs before
            # any generated code executes.
            if isinstance(spatial, (list, tuple)) and spatial:
                h = int(spatial[0])
                w = int(spatial[1]) if len(spatial) > 1 else h
            elif isinstance(spatial, (int, float)) and spatial:
                h = w = int(spatial)
            else:
                h = w = 224
            candidates.append(f"torch.randn(1, {c}, {h}, {w})")

        vocab_size = inp.get("vocab_size")
        seq_len = inp.get("seq_len")
        if vocab_size or seq_len:
            v = int(vocab_size or 10000)
            s = int(seq_len or 64)
            candidates.append(f"torch.randint(0, {v}, (1, {s}))")

        if not candidates:
            candidates = ["torch.randn(1, 3, 224, 224)", "torch.randint(0, 10000, (1, 64))"]

        return candidates

    def _repair_code(
        self,
        code: str,
        verification_report: dict[str, Any],
        spec: dict[str, Any],
        graph: ArchitectureGraph,
        attempt: int = 2,
    ) -> str | None:
        """
        Targeted code repair using LLM with structured diagnostic feedback.
        Returns None (rather than raising) if the repair call itself fails --
        e.g. a rate limit -- so the bounded repair loop in _run_pipeline can
        stop gracefully on the last known verification_report instead of an
        unhandled exception crashing the whole upload.
        """
        error = verification_report.get("error", "Code verification failed.")
        stage = verification_report.get("stage", "verification")
        expected_shape = (
            verification_report.get("output_shape") or (spec.get("output") or {})
        )
        input_shape = (
            verification_report.get("input_shape") or (spec.get("input") or {})
        )

        prompt = f"""You are an expert PyTorch developer repairing broken neural network code.
The following PyTorch code was generated for architecture {spec.get("name", "Model")} ({spec.get("model_family", "unknown")}), but failed execution verification.

DIAGNOSTIC REPORT:
- Failure Stage: {stage}
- Error: {error}
- Expected Output / Specification: {expected_shape}
- Input Shape: {input_shape}

CURRENT FAILING CODE:
```python
{code}
```

TASK:
Fix the error in the PyTorch code so that:
1. It imports correctly (torch, torch.nn as nn, etc.).
2. The nn.Module instantiates without error.
3. The forward() pass runs successfully on the input tensor and matches the expected output shape.
4. Keep the architecture intact.

Return ONLY the corrected valid Python code. No explanation, no markdown backticks."""

        try:
            response = llm_complete(prompt)
        except Exception as exc:
            print(f"Repair LLM call failed (attempt {attempt}): {exc}")
            return None
        cleaned = re.sub(r"^```(?:python)?\n?", "", response.strip())
        cleaned = re.sub(r"\n?```$", "", cleaned)
        return cleaned.strip()

    def _llm_generate_code(self, spec: dict[str, Any], graph: ArchitectureGraph) -> str:
        """
        Use LLM to generate complete PyTorch code from architecture spec.

        Provides structured prompt with layers, connections, and schema.
        Returns complete, runnable nn.Module code.
        """
        layers_desc = "\n".join(
            [f"  - {n.label} ({n.type}): {n.params}" for n in graph.nodes]
        )
        connections_desc = "\n".join(
            [f"  - {e.source} → {e.target} ({e.edge_type})" for e in graph.edges]
        )
        spec_json = json.dumps(spec, indent=2)[:2000]

        prompt = f"""You are an expert PyTorch developer. Generate complete, runnable PyTorch code for a neural network architecture extracted from a research paper.

Architecture Name: {spec.get("name", "Unknown")}
Family: {spec.get("model_family", "unknown").upper()}

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
        code = re.sub(r"^```python\n", "", code)
        code = re.sub(r"\n```$", "", code)

        return code.strip()
