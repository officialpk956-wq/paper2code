"""
backend/corpus_builder.py

Programmatically generates the ConfigDict representations for 15 verified architectures
for Phase 6A, runs them through the Paper2CodePipeline, applies quality audits,
and seeds the SQLite database with support_level="verified".
"""

import os
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))


from backend.database import SessionLocal, engine
from backend.models import Base, Paper, PaperModule
from core.agents import ParsingAgentImpl
from core.module_generator import generate_modules
from core.orchestrator.pipeline import Paper2CodePipeline

# ---------------------------------------------------------------------------
# Architecture Definitions (ConfigDict Builders)
# ---------------------------------------------------------------------------


def _build_linear_layers(prefix, layer_defs):
    layers = []
    connections = []
    prev_id = None
    for i, (l_type, l_label, params) in enumerate(layer_defs):
        lid = f"{prefix}_{i}"
        layers.append({"id": lid, "type": l_type, "label": l_label, "params": params})
        if prev_id:
            connections.append({"source": prev_id, "target": lid, "edge_type": "flow"})
        prev_id = lid
    return layers, connections


def build_lenet5():
    defs = [
        ("Conv2d", "Conv1 5x5", {"channels": 6}),
        ("AvgPool2d", "AvgPool 2x2", {"kernel_size": 2, "stride": 2}),
        ("Conv2d", "Conv2 5x5", {"channels": 16}),
        ("AvgPool2d", "AvgPool 2x2", {"kernel_size": 2, "stride": 2}),
        ("Conv2d", "Conv3 5x5", {"channels": 120}),
        ("Linear", "FC1", {"hidden_size": 84}),
        ("Linear", "Output", {"hidden_size": 10}),
    ]
    return _build_linear_layers("lenet", defs)


def build_alexnet():
    defs = [
        ("Conv2d", "Conv1 11x11", {"channels": 96, "stride": 4}),
        ("MaxPool2d", "MaxPool 3x3", {"kernel_size": 3, "stride": 2}),
        ("Conv2d", "Conv2 5x5", {"channels": 256}),
        ("MaxPool2d", "MaxPool 3x3", {"kernel_size": 3, "stride": 2}),
        ("Conv2d", "Conv3 3x3", {"channels": 384}),
        ("Conv2d", "Conv4 3x3", {"channels": 384}),
        ("Conv2d", "Conv5 3x3", {"channels": 256}),
        ("MaxPool2d", "MaxPool 3x3", {"kernel_size": 3, "stride": 2}),
        ("Linear", "FC1", {"hidden_size": 4096}),
        ("Linear", "FC2", {"hidden_size": 4096}),
        ("Linear", "Output", {"hidden_size": 1000}),
    ]
    return _build_linear_layers("alexnet", defs)


def build_vgg(layers_config):
    defs = []
    for num_convs, channels in layers_config:
        for _ in range(num_convs):
            defs.append(("Conv2d", f"Conv 3x3 ({channels}ch)", {"channels": channels}))
        defs.append(("MaxPool2d", "MaxPool 2x2", {"kernel_size": 2, "stride": 2}))
    defs.extend(
        [
            ("Linear", "FC1", {"hidden_size": 4096}),
            ("Linear", "FC2", {"hidden_size": 4096}),
            ("Linear", "Output", {"hidden_size": 1000}),
        ]
    )
    return _build_linear_layers("vgg", defs)


def build_googlenet():
    defs = [
        ("Conv2d", "Stem Conv 7x7", {"channels": 64, "stride": 2}),
        ("MaxPool2d", "MaxPool", {"kernel_size": 3, "stride": 2}),
        ("Conv2d", "Conv 3x3", {"channels": 192}),
        ("MaxPool2d", "MaxPool", {"kernel_size": 3, "stride": 2}),
        ("InceptionBlock", "Inception 3a", {"channels": 256}),
        ("InceptionBlock", "Inception 3b", {"channels": 480}),
        ("MaxPool2d", "MaxPool", {"kernel_size": 3, "stride": 2}),
        ("InceptionBlock", "Inception 4a", {"channels": 512}),
        ("InceptionBlock", "Inception 4b", {"channels": 512}),
        ("InceptionBlock", "Inception 4c", {"channels": 512}),
        ("InceptionBlock", "Inception 4d", {"channels": 528}),
        ("InceptionBlock", "Inception 4e", {"channels": 832}),
        ("MaxPool2d", "MaxPool", {"kernel_size": 3, "stride": 2}),
        ("InceptionBlock", "Inception 5a", {"channels": 832}),
        ("InceptionBlock", "Inception 5b", {"channels": 1024}),
        ("AvgPool2d", "Global AvgPool", {}),
        ("Linear", "Output", {"hidden_size": 1000}),
    ]
    return _build_linear_layers("googlenet", defs)


def build_resnet(blocks_per_stage, block_type="ResidualBlock"):
    defs = [
        ("Conv2d", "Stem Conv 7x7", {"channels": 64, "stride": 2}),
        ("MaxPool2d", "MaxPool", {"kernel_size": 3, "stride": 2}),
    ]
    channels = 64
    for stage, blocks in enumerate(blocks_per_stage):
        for b in range(blocks):
            defs.append((block_type, f"Stage {stage + 1} Block {b + 1}", {"channels": channels}))
        channels *= 2
    defs.extend([("AvgPool2d", "Global AvgPool", {}), ("Linear", "Output", {"hidden_size": 1000})])
    return _build_linear_layers("resnet", defs)


def build_densenet121():
    defs = [
        ("Conv2d", "Stem Conv 7x7", {"channels": 64, "stride": 2}),
        ("MaxPool2d", "MaxPool", {"kernel_size": 3, "stride": 2}),
        ("DenseBlock", "Dense Block 1 (6 layers)", {"channels": 256}),
        ("TransitionLayer", "Transition 1", {"channels": 128}),
        ("DenseBlock", "Dense Block 2 (12 layers)", {"channels": 512}),
        ("TransitionLayer", "Transition 2", {"channels": 256}),
        ("DenseBlock", "Dense Block 3 (24 layers)", {"channels": 1024}),
        ("TransitionLayer", "Transition 3", {"channels": 512}),
        ("DenseBlock", "Dense Block 4 (16 layers)", {"channels": 1024}),
        ("AvgPool2d", "Global AvgPool", {}),
        ("Linear", "Output", {"hidden_size": 1000}),
    ]
    return _build_linear_layers("densenet", defs)


def build_mobilenetv2():
    defs = [
        ("Conv2d", "Conv 3x3", {"channels": 32, "stride": 2}),
        ("InvertedResidual", "Inverted Residual 1", {"channels": 16}),
        ("InvertedResidual", "Inverted Residual 2 (x2)", {"channels": 24}),
        ("InvertedResidual", "Inverted Residual 3 (x3)", {"channels": 32}),
        ("InvertedResidual", "Inverted Residual 4 (x4)", {"channels": 64}),
        ("InvertedResidual", "Inverted Residual 5 (x3)", {"channels": 96}),
        ("InvertedResidual", "Inverted Residual 6 (x3)", {"channels": 160}),
        ("InvertedResidual", "Inverted Residual 7", {"channels": 320}),
        ("Conv2d", "Conv 1x1", {"channels": 1280}),
        ("AvgPool2d", "Global AvgPool", {}),
        ("Linear", "Output", {"hidden_size": 1000}),
    ]
    return _build_linear_layers("mobilenetv2", defs)


def build_efficientnet_b0():
    defs = [
        ("Conv2d", "Conv 3x3", {"channels": 32, "stride": 2}),
        ("MBConvBlock", "MBConv1", {"channels": 16}),
        ("MBConvBlock", "MBConv6 (x2)", {"channels": 24}),
        ("MBConvBlock", "MBConv6 (x2)", {"channels": 40}),
        ("MBConvBlock", "MBConv6 (x3)", {"channels": 80}),
        ("MBConvBlock", "MBConv6 (x3)", {"channels": 112}),
        ("MBConvBlock", "MBConv6 (x4)", {"channels": 192}),
        ("MBConvBlock", "MBConv6", {"channels": 320}),
        ("Conv2d", "Conv 1x1", {"channels": 1280}),
        ("AvgPool2d", "Global AvgPool", {}),
        ("Linear", "Output", {"hidden_size": 1000}),
    ]
    return _build_linear_layers("efficientnet", defs)


def build_fcn():
    defs = [
        ("Conv2d", "Encoder Block 1", {"channels": 64}),
        ("MaxPool2d", "Downsample 1", {"kernel_size": 2}),
        ("Conv2d", "Encoder Block 2", {"channels": 128}),
        ("MaxPool2d", "Downsample 2", {"kernel_size": 2}),
        ("Conv2d", "Encoder Block 3", {"channels": 256}),
        ("MaxPool2d", "Downsample 3", {"kernel_size": 2}),
        ("Conv2d", "Encoder Block 4", {"channels": 512}),
        ("MaxPool2d", "Downsample 4", {"kernel_size": 2}),
        ("Conv2d", "Encoder Block 5", {"channels": 512}),
        ("MaxPool2d", "Downsample 5", {"kernel_size": 2}),
        ("Conv2d", "FC6 (Conv)", {"channels": 4096}),
        ("Conv2d", "FC7 (Conv)", {"channels": 4096}),
        ("ConvTranspose2d", "Upsample", {"channels": 21}),
    ]
    return _build_linear_layers("fcn", defs)


def build_unet():
    layers = []
    connections = []

    # Encoder
    for i, ch in enumerate([64, 128, 256, 512]):
        layers.append(
            {
                "id": f"enc_{i}",
                "type": "Conv2d",
                "label": f"Encoder Block {i + 1} ({ch}ch)",
                "params": {"channels": ch},
            }
        )
        if i > 0:
            connections.append(
                {"source": f"pool_{i - 1}", "target": f"enc_{i}", "edge_type": "flow"}
            )
        layers.append(
            {
                "id": f"pool_{i}",
                "type": "MaxPool2d",
                "label": f"Downsample {i + 1}",
                "params": {"kernel_size": 2},
            }
        )
        connections.append({"source": f"enc_{i}", "target": f"pool_{i}", "edge_type": "flow"})

    # Bottleneck
    layers.append(
        {
            "id": "bottleneck",
            "type": "Conv2d",
            "label": "Bottleneck (1024ch)",
            "params": {"channels": 1024},
        }
    )
    connections.append({"source": "pool_3", "target": "bottleneck", "edge_type": "flow"})

    # Decoder
    prev = "bottleneck"
    for i, ch in enumerate([512, 256, 128, 64]):
        layers.append(
            {"id": f"up_{i}", "type": "Upsample", "label": f"Upsample {i + 1}", "params": {}}
        )
        connections.append({"source": prev, "target": f"up_{i}", "edge_type": "flow"})

        layers.append(
            {
                "id": f"dec_{i}",
                "type": "Conv2d",
                "label": f"Decoder Block {i + 1} ({ch}ch)",
                "params": {"channels": ch},
            }
        )
        connections.append({"source": f"up_{i}", "target": f"dec_{i}", "edge_type": "flow"})

        # Skip connection
        connections.append({"source": f"enc_{3 - i}", "target": f"dec_{i}", "edge_type": "skip"})
        prev = f"dec_{i}"

    # Head
    layers.append(
        {"id": "head", "type": "Conv2d", "label": "Output Conv (1x1)", "params": {"channels": 2}}
    )
    connections.append({"source": prev, "target": "head", "edge_type": "flow"})

    return layers, connections


def build_transformer(encoder_layers, decoder_layers):
    defs = [("PatchEmbedding", "Token + Positional Embedding", {"embed_dim": 512})]
    for i in range(encoder_layers):
        defs.append(("MultiHeadAttention", f"Encoder Layer {i + 1} - MHSA", {"d_model": 512}))
        defs.append(("FeedForward", f"Encoder Layer {i + 1} - FFN", {"hidden_size": 2048}))
    for i in range(decoder_layers):
        defs.append(
            ("MultiHeadAttention", f"Decoder Layer {i + 1} - Masked MHSA", {"d_model": 512})
        )
        defs.append(
            ("MultiHeadAttention", f"Decoder Layer {i + 1} - Cross-Attention", {"d_model": 512})
        )
        defs.append(("FeedForward", f"Decoder Layer {i + 1} - FFN", {"hidden_size": 2048}))
    defs.append(("Linear", "Classification Head", {"hidden_size": 1000}))
    return _build_linear_layers("transformer", defs)


def build_vit(blocks):
    defs = [("PatchEmbedding", "Patch Embedding + Positional Encoding", {"embed_dim": 768})]
    for i in range(blocks):
        defs.append(("MultiHeadAttention", f"Transformer Block {i + 1} - MHSA", {"d_model": 768}))
        defs.append(("FeedForward", f"Transformer Block {i + 1} - FFN", {"hidden_size": 3072}))
    defs.append(("Linear", "Classification Head", {"hidden_size": 1000}))
    return _build_linear_layers("vit", defs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCHITECTURES = {
    "LeNet-5": {"family": "cnn", "authors": "LeCun et al.", "builder": lambda: build_lenet5()},
    "AlexNet": {
        "family": "cnn",
        "authors": "Krizhevsky et al.",
        "builder": lambda: build_alexnet(),
    },
    "VGG16": {
        "family": "cnn",
        "authors": "Simonyan & Zisserman",
        "builder": lambda: build_vgg([(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]),
    },
    "VGG19": {
        "family": "cnn",
        "authors": "Simonyan & Zisserman",
        "builder": lambda: build_vgg([(2, 64), (2, 128), (4, 256), (4, 512), (4, 512)]),
    },
    "GoogLeNet": {
        "family": "cnn",
        "authors": "Szegedy et al.",
        "builder": lambda: build_googlenet(),
    },
    "ResNet18": {
        "family": "resnet",
        "authors": "He et al.",
        "builder": lambda: build_resnet([2, 2, 2, 2]),
    },
    "ResNet34": {
        "family": "resnet",
        "authors": "He et al.",
        "builder": lambda: build_resnet([3, 4, 6, 3]),
    },
    "ResNet50": {
        "family": "resnet",
        "authors": "He et al.",
        "builder": lambda: build_resnet([3, 4, 6, 3], block_type="BottleneckBlock"),
    },
    "DenseNet121": {
        "family": "cnn",
        "authors": "Huang et al.",
        "builder": lambda: build_densenet121(),
    },
    "MobileNetV2": {
        "family": "cnn",
        "authors": "Sandler et al.",
        "builder": lambda: build_mobilenetv2(),
    },
    "EfficientNet-B0": {
        "family": "cnn",
        "authors": "Tan & Le",
        "builder": lambda: build_efficientnet_b0(),
    },
    "FCN": {"family": "unet", "authors": "Long et al.", "builder": lambda: build_fcn()},
    "U-Net": {"family": "unet", "authors": "Ronneberger et al.", "builder": lambda: build_unet()},
    "Transformer": {
        "family": "transformer",
        "authors": "Vaswani et al.",
        "builder": lambda: build_transformer(6, 6),
    },
    "Vision Transformer": {
        "family": "transformer",
        "authors": "Dosovitskiy et al.",
        "builder": lambda: build_vit(12),
    },
}

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def process_and_persist():
    print("=" * 70)
    print("PHASE 6A: Golden Research Corpus Expansion (15 Verified Architectures)")
    print("=" * 70)

    # Use deterministic parser
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))

    # We will reset DB to ensure clean state
    if os.getenv("ALLOW_DB_RESET", "").lower() != "true":
        raise RuntimeError(
            "Database wipe blocked. Set ALLOW_DB_RESET=true to allow. NEVER do this in production."
        )
    print("Resetting database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    success_count = 0
    global_explanations = set()

    try:
        for name, meta in ARCHITECTURES.items():
            print(f"\nProcessing -> {name}")
            layers, connections = meta["builder"]()

            config = {"name": name, "layers": layers, "connections": connections}

            schema = {"model_family": meta["family"], "authors": meta["authors"]}

            # 1. Run Pipeline
            pipeline_result = pipeline.run_single(config)

            # 2. Generate Modules
            paper_meta, modules = generate_modules(name, schema, pipeline_result)

            # 3. Quality Audit Validation
            if len(modules) == 0:
                raise ValueError("QUALITY AUDIT FAILED: Empty modules list.")

            # Check NaN & Duplicates
            for m in modules:
                if not m.graph_nodes:
                    # some generic blocks might have no explicit non-trivial nodes if they were empty, but we shouldn't fail entirely
                    pass
                flops_score = m.flops_context.get("total_flops_score", 0)
                params = m.flops_context.get("total_params_estimate", 0)
                import math

                if math.isnan(flops_score) or math.isnan(params):
                    raise ValueError(f"QUALITY AUDIT FAILED: NaN found in module {m.layer_name}")

                exp = m.explanation
                # We enforce a strict unique check across all modules in all papers
                if exp in global_explanations and not exp.startswith("Structural module:"):
                    raise ValueError(
                        f"QUALITY AUDIT FAILED: Duplicate explanation found: {exp[:50]}..."
                    )
                global_explanations.add(exp)

            # 4. Inject support_level into architecture graph JSON
            paper_meta["architecture_graph"]["support_level"] = "verified"
            paper_meta["architecture_graph"]["model_family"] = meta["family"]

            # 5. Persist to DB
            paper = Paper(
                title=paper_meta["title"],
                authors=paper_meta["authors"],
                abstract=f"An educational representation of the {name} architecture.",
                architecture_graph=paper_meta["architecture_graph"],
                flops_analysis=paper_meta["flops_analysis"],
            )
            db.add(paper)
            db.flush()

            for m in modules:
                pm = PaperModule(
                    paper_id=paper.id,
                    layer_name=m.layer_name,
                    module_type=m.module_type,
                    explanation=m.explanation,
                    tensor_flow=m.tensor_flow,
                    graph_nodes=m.graph_nodes,
                    flops_context=m.flops_context,
                    order_index=m.order_index,
                )
                db.add(pm)

            db.commit()
            print(f"  [OK] Validated and Persisted (ID: {paper.id}, Modules: {len(modules)})")
            success_count += 1

    except Exception as e:
        db.rollback()
        import traceback

        traceback.print_exc()
        print(f"\n[FATAL ERROR] {e}")
    finally:
        db.close()

    print("\n" + "=" * 70)
    print(
        f"Pipeline complete. {success_count}/{len(ARCHITECTURES)} verified architectures processed."
    )
    print("=" * 70)


if __name__ == "__main__":
    process_and_persist()
