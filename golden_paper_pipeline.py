"""
golden_paper_pipeline.py

TensorTonic MVP — Golden Paper Set Processor.

Processes the 3 benchmark papers through the full pipeline:
  1. ResNet (He et al. 2015)
  2. Transformer (Vaswani et al. 2017)
  3. U-Net (Ronneberger et al. 2015)

Steps:
  STEP 1 — Run each schema through Paper2CodePipeline.run_single()
  STEP 2 — Pass result to Module Generation Engine
  STEP 3 — Persist Paper + PaperModule records to SQLite DB
  STEP 4 — Print Validation Report
  STEP 5 — Print Human Review Report
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from core.orchestrator.pipeline import Paper2CodePipeline
from core.agents import ParsingAgentImpl
from core.module_generator import generate_modules, LearningModule
from backend.database import SessionLocal
from backend.models import Paper, PaperModule


# ---------------------------------------------------------------------------
# Golden Paper Definitions
# ---------------------------------------------------------------------------

GOLDEN_PAPERS = [
    {
        "paper_name": "Deep Residual Learning for Image Recognition (ResNet)",
        "schema_path": "outputs/code_ready/resnet_he_2015.json",
        "architecture_type": "Convolutional Neural Network (CNN)",
    },
    {
        "paper_name": "Attention Is All You Need (Transformer)",
        "schema_path": "outputs/code_ready/attention_all_you_need_2017.json",
        "architecture_type": "Transformer / Self-Attention",
    },
    {
        "paper_name": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "schema_path": "outputs/code_ready_unet/unet_ronneberger_2015.json",
        "architecture_type": "Encoder-Decoder (U-Net)",
    },
]


# ---------------------------------------------------------------------------
# Step 1+2: Run pipeline and generate modules
# ---------------------------------------------------------------------------

def schema_to_config(schema: dict, paper_name: str) -> dict:
    """
    Convert a code_ready JSON schema (builder spec) to a ConfigDict
    that the Paper2CodePipeline.run_single() can consume.

    The builder specs describe the architecture at a high level
    (stem, block, stages, head). We expand them into a flat layers list.
    """
    family = schema.get("model_family", "unknown").lower()
    layers = []
    connections = []
    prev_id = None

    def add_layer(layer_id, layer_type, label, params=None):
        nonlocal prev_id
        entry = {"id": layer_id, "type": layer_type, "label": label, "params": params or {}}
        layers.append(entry)
        if prev_id:
            connections.append({"source": prev_id, "target": layer_id, "edge_type": "flow"})
        prev_id = layer_id

    stem = schema.get("stem", {})
    stem_type = stem.get("type", "conv2d")
    stem_params = stem.get("params", {})
    add_layer("stem", stem_type, f"Stem ({stem_type})", stem_params)

    if family in ("resnet",):
        add_layer("maxpool", "MaxPool2d", "Max Pool", {"kernel_size": 3, "stride": 2})
        for stage in schema.get("stages", []):
            stage_name = stage.get("name", "stage")
            num_blocks = stage.get("num_blocks", 1)
            out_ch = stage.get("out_channels", 64)
            for b in range(num_blocks):
                bid = f"{stage_name}_block{b}"
                add_layer(bid, "ResidualBlock", f"{stage_name} Block {b+1}",
                          {"channels": out_ch, "stride": stage.get("stride", 1)})
        add_layer("avgpool", "AvgPool2d", "Global Average Pool", {})
        num_classes = schema.get("output", {}).get("num_classes") or 1000
        add_layer("fc", "Linear", f"FC ({num_classes} classes)", {"hidden_size": num_classes})

    elif family == "transformer":
        block = schema.get("block", {})
        block_params = block.get("params", {})
        d_model = block_params.get("d_model", 512)
        num_heads = block_params.get("num_heads", 8)
        vocab = schema.get("input", {}).get("vocab_size", 10000)
        add_layer("embed", "PatchEmbedding", "Token + Positional Embedding",
                  {"embed_dim": d_model, "vocab_size": vocab})
        repeats = schema.get("stages", [{}])[0].get("repeats", 6)
        for i in range(repeats):
            lid = f"enc_layer_{i}"
            add_layer(f"{lid}_mhsa", "MultiHeadAttention",
                      f"Encoder Layer {i+1} — MHSA",
                      {"d_model": d_model, "heads": num_heads})
            add_layer(f"{lid}_ffn", "FeedForward",
                      f"Encoder Layer {i+1} — FFN",
                      {"hidden_size": block_params.get("ffn_dim", 2048)})
            add_layer(f"{lid}_norm", "LayerNorm",
                      f"Encoder Layer {i+1} — LayerNorm", {"channels": d_model})
        num_classes = schema.get("output", {}).get("num_classes") or 1000
        add_layer("head", "Linear", f"Classification Head ({num_classes} classes)",
                  {"hidden_size": num_classes})

    elif family == "unet":
        encoder_channels = schema.get("encoder", [64, 128, 256, 512])
        bottleneck_ch = schema.get("bottleneck", 1024)
        decoder_channels = schema.get("decoder", [512, 256, 128, 64])
        num_classes = schema.get("output", {}).get("num_classes") or 2

        for i, ch in enumerate(encoder_channels):
            add_layer(f"enc_{i}", "Conv2d", f"Encoder Block {i+1} ({ch}ch)", {"channels": ch})
            if i < len(encoder_channels) - 1:
                add_layer(f"enc_{i}_pool", "MaxPool2d", f"Downsample {i+1}", {"kernel_size": 2})

        add_layer("bottleneck", "Conv2d", f"Bottleneck ({bottleneck_ch}ch)",
                  {"channels": bottleneck_ch})

        for i, ch in enumerate(decoder_channels):
            add_layer(f"dec_{i}_up", "Upsample", f"Upsample {i+1}", {})
            add_layer(f"dec_{i}", "Conv2d", f"Decoder Block {i+1} ({ch}ch)", {"channels": ch})

        add_layer("head", "Conv2d", f"Output Conv (1x1, {num_classes} classes)",
                  {"channels": num_classes})

        # Add U-Net skip connections between contracting path (encoder) and expanding path (decoder)
        for i in range(len(decoder_channels)):
            enc_idx = len(encoder_channels) - 1 - i
            connections.append({
                "source": f"enc_{enc_idx}",
                "target": f"dec_{i}",
                "edge_type": "skip"
            })
    else:
        # Generic fallback for unknown families
        for i, stage in enumerate(schema.get("stages", [])):
            add_layer(f"stage_{i}", "Conv2d", f"Stage {i+1}",
                      {"channels": stage.get("out_channels", 64)})

    return {
        "name": paper_name,
        "layers": layers,
        "connections": connections,
    }


def process_paper(paper_def: dict, pipeline: Paper2CodePipeline) -> tuple:
    """Run one paper through the engine. Returns (paper_meta, modules, family)."""
    schema_path = Path(paper_def["schema_path"])
    schema = json.loads(schema_path.read_text())

    print(f"\n  -> Loading schema: {schema_path.name}")
    print(f"    Family: {schema.get('model_family', 'unknown').upper()}")

    # Convert builder spec to ConfigDict format
    config = schema_to_config(schema, paper_def["paper_name"])
    print(f"    Layers in ConfigDict: {len(config['layers'])}")

    pipeline_result = pipeline.run_single(config)
    graph = pipeline_result["graph"]

    print(f"    Graph nodes: {len(graph.nodes)}")
    print(f"    KAG motifs:  {pipeline_result.get('kag_motifs', [])}")

    paper_meta, modules = generate_modules(
        paper_name=paper_def["paper_name"],
        schema=schema,
        pipeline_result=pipeline_result,
    )
    return paper_meta, modules, schema.get("model_family", "unknown")


# ---------------------------------------------------------------------------
# Step 3: Persist to database
# ---------------------------------------------------------------------------

def persist(paper_meta: dict, modules: list, db) -> Paper:
    """Upsert a Paper and its PaperModules into the database."""
    # Check if already exists
    existing = db.query(Paper).filter(Paper.title == paper_meta["title"]).first()
    if existing:
        # Delete old modules, re-insert
        db.query(PaperModule).filter(PaperModule.paper_id == existing.id).delete()
        paper = existing
        paper.abstract = paper_meta["abstract"]
        paper.architecture_graph = paper_meta["architecture_graph"]
        paper.flops_analysis = paper_meta["flops_analysis"]
    else:
        paper = Paper(
            title=paper_meta["title"],
            authors=paper_meta.get("authors"),
            abstract=paper_meta["abstract"],
            architecture_graph=paper_meta["architecture_graph"],
            flops_analysis=paper_meta["flops_analysis"],
        )
        db.add(paper)

    db.flush()  # get paper.id

    for m in modules:
        pm = PaperModule(
            paper_id=paper.id,
            layer_name=m.layer_name,
            module_type=m.module_type,
            explanation=m.explanation,
            tensor_flow=m.tensor_flow,
            graph_nodes=m.graph_nodes,
            flops_context={**(m.flops_context or {}), "confidence": m.confidence},
            order_index=m.order_index,
        )
        db.add(pm)

    db.commit()
    db.refresh(paper)
    return paper


# ---------------------------------------------------------------------------
# Step 4: Validation Report
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLDS = {
    "high":   0.80,
    "medium": 0.60,
    "low":    0.40,
}

def _confidence_label(score: float) -> str:
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return "✅ HIGH"
    if score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "⚠️  MEDIUM"
    if score >= CONFIDENCE_THRESHOLDS["low"]:
        return "🔶 LOW"
    return "❌ VERY LOW"


def print_validation_report(results: list):
    print("\n" + "=" * 70)
    print("POSTGRESQL VALIDATION REPORT — STEP 4: MODULE VALIDATION")
    print("=" * 70)

    for paper_name, arch_type, modules in results:
        print(f"\n{'─' * 60}")
        print(f"  Paper Name:        {paper_name}")
        print(f"  Architecture Type: {arch_type}")
        print(f"  Generated Modules: {len(modules)}")
        print(f"{'─' * 60}")

        for m in modules:
            node_ids = [n["node_id"] for n in m.graph_nodes]
            has_tensor = bool(m.tensor_flow)
            flops_score = m.flops_context.get("total_flops_score", 0)
            params = m.flops_context.get("total_params_estimate", 0)
            conf_label = _confidence_label(m.confidence)

            print(f"\n  [{m.order_index + 1:02d}] {m.layer_name}")
            print(f"        Type:            {m.module_type}")
            print(f"        Source Nodes:    {node_ids or '(no non-trivial nodes)'}")
            print(f"        Tensor Data:     {'YES' if has_tensor else 'NO'}")
            print(f"        FLOPs Score:     {flops_score}")
            print(f"        Param Estimate:  {params:,}")
            print(f"        Confidence:      {m.confidence:.2f}  {conf_label}")

            # Reasoning data preview (first 120 chars)
            reasoning_preview = m.explanation[:120].replace("\n", " ")
            print(f"        Reasoning Data:  \"{reasoning_preview}...\"")


# ---------------------------------------------------------------------------
# Step 5: Human Review Report
# ---------------------------------------------------------------------------

def print_human_review_report(results: list):
    print("\n" + "=" * 70)
    print("STEP 5: HUMAN REVIEW REPORT")
    print("=" * 70)

    for paper_name, arch_type, modules in results:
        print(f"\n{'─' * 60}")
        print(f"  Paper: {paper_name}")
        print(f"{'─' * 60}")

        good = [m for m in modules if m.confidence >= CONFIDENCE_THRESHOLDS["high"]]
        weak = [m for m in modules if CONFIDENCE_THRESHOLDS["low"] <= m.confidence < CONFIDENCE_THRESHOLDS["medium"]]
        very_weak = [m for m in modules if m.confidence < CONFIDENCE_THRESHOLDS["low"]]

        # Good modules
        if good:
            print(f"\n  ✅ GOOD MODULES ({len(good)}):")
            for m in good:
                print(f"     • [{m.order_index + 1:02d}] {m.layer_name}  (confidence: {m.confidence:.2f})")

        # Weak modules
        if weak:
            print(f"\n  ⚠️  WEAK MODULES ({len(weak)}):")
            for m in weak:
                reason = []
                if not m.graph_nodes:
                    reason.append("no source nodes")
                if len(m.explanation) <= 50:
                    reason.append("thin explanation")
                if not m.tensor_flow:
                    reason.append("no tensor data")
                print(f"     • [{m.order_index + 1:02d}] {m.layer_name}  → Issues: {', '.join(reason) or 'low confidence'}")

        if very_weak:
            print(f"\n  ❌ VERY WEAK MODULES ({len(very_weak)}):")
            for m in very_weak:
                print(f"     • [{m.order_index + 1:02d}] {m.layer_name}  (confidence: {m.confidence:.2f})")

        # Ordering assessment
        module_names = [m.layer_name for m in modules]
        print(f"\n  📋 MODULE ORDER:")
        for i, name in enumerate(module_names):
            print(f"     {i + 1}. {name}")

        # Missing concept detection
        family = arch_type.lower()
        missing = []
        joined = " ".join(module_names).lower()

        if "transformer" in family or "attention" in family:
            if "positional" not in joined and "embedding" not in joined:
                missing.append("Positional Encoding not detected as dedicated module")
            if "feed forward" not in joined and "ffn" not in joined:
                missing.append("Feed-Forward Network sub-layer not isolated")
            if "decoder" not in joined:
                missing.append("Decoder path not detected (Transformer is encoder-only in this schema)")

        if "resnet" in family or "cnn" in family.lower():
            if "global" not in joined and "pool" not in joined:
                missing.append("Global Average Pooling not detected as separate module")

        if "unet" in family or "encoder-decoder" in family:
            if "skip" not in joined and "concatenat" not in joined:
                missing.append("Skip connections not represented as explicit module")
            if "bottleneck" not in joined and "bridge" not in joined:
                missing.append("Bottleneck/Bridge may not be isolated correctly")

        if missing:
            print(f"\n  🔍 MISSING / WEAK CONCEPTS:")
            for m_item in missing:
                print(f"     ⚠️  {m_item}")
        else:
            print(f"\n  🔍 MISSING CONCEPTS: None detected")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TensorTonic MVP — Golden Paper Set Pipeline")
    print("Papers: ResNet | Transformer | U-Net")
    print("=" * 70)

    # Initialize pipeline with deterministic parser (no LLM)
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))

    all_results = []
    db = SessionLocal()

    try:
        for paper_def in GOLDEN_PAPERS:
            print(f"\n{'=' * 60}")
            print(f"STEP 1+2: Processing -> {paper_def['paper_name']}")
            print(f"{'=' * 60}")

            try:
                paper_meta, modules, family = process_paper(paper_def, pipeline)
            except Exception as e:
                print(f"  [ERROR] Failed to process {paper_def['paper_name']}: {e}")
                import traceback
                traceback.print_exc()
                continue

            print(f"  -> Modules generated: {len(modules)}")

            print(f"\nSTEP 3: Persisting to database...")
            try:
                paper_record = persist(paper_meta, modules, db)
                print(f"  -> Paper ID: {paper_record.id}")
                print(f"  -> Modules persisted: {len(paper_record.modules)}")
            except Exception as e:
                db.rollback()
                print(f"  [ERROR] DB persist failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            all_results.append((
                paper_def["paper_name"],
                paper_def["architecture_type"],
                modules,
            ))

    finally:
        db.close()

    # Print reports
    if all_results:
        print_validation_report(all_results)
        print_human_review_report(all_results)

    print("\n" + "=" * 70)
    print(f"Pipeline complete. {len(all_results)}/{len(GOLDEN_PAPERS)} papers processed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
