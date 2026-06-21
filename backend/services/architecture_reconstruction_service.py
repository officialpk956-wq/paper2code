"""
Architecture Reconstruction Service for Phase 14A.

Deterministic (no LLM) reconstruction of neural network architecture blueprints
from uploaded paper text.  Produces an ArchitectureBlueprint stored at:
  paper.architecture_graph.ingestion.architecture_blueprint
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

from core.rag.flops_engine import FLOPsEngine

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Component:
    id: str
    name: str
    type: str   # conv | attention | embedding | pooling | mlp | normalization
                # activation | skip_connection | upsample | downsample | head
    shape: List[int]  # OUTPUT shape
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Connection:
    source: str
    target: str
    connection_type: str  # sequential | residual | concat | cross_attention

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArchitectureBlueprint:
    id: str
    name: str
    paper_id: Optional[int]
    components: List[Component]
    connections: List[Connection]
    input_shape: List[int]
    output_shape: List[int]
    estimated_parameters: int
    estimated_flops: float
    confidence_score: float
    tensor_flow: List[Dict[str, Any]]
    architecture_href: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# FLOPs type mapping
# ---------------------------------------------------------------------------

_COMP_TO_FLOPS_TYPE: Dict[str, str] = {
    "conv":            "conv2d",
    "attention":       "multiheadattention",
    "embedding":       "patchembedding",
    "pooling":         "",
    "mlp":             "feedforward",
    "normalization":   "layernorm",
    "activation":      "",
    "skip_connection": "",
    "upsample":        "upsample",
    "downsample":      "conv2d",
    "head":            "linear",
}


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _extract_signals(
    title: str, abstract: str, sections: List[Dict]
) -> Dict[str, Optional[int]]:
    """Extract numeric architecture parameters from paper text."""
    body = " ".join(s.get("content", "") for s in sections[:6])
    text = f"{title} {abstract} {body}".lower()

    def first_int(*patterns: str) -> Optional[int]:
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                for g in m.groups():
                    if g is not None:
                        try:
                            val = int(g)
                            if val > 0:
                                return val
                        except ValueError:
                            pass
        return None

    return {
        "depth": first_int(
            r"\b(\d{1,3})[- ]?(?:encoder\s+)?layer",
            r"\bL\s*=\s*(\d{1,3})\b",
            r"\bN\s*=\s*(\d{1,3})\s*(?:encoder|layer|block)",
        ),
        "num_heads": first_int(
            r"\b(\d+)\s*(?:attention\s+)?head",
        ),
        "d_model": first_int(
            r"\bd_?model\s*[=:]\s*(\d+)",
            r"\bhidden.?(?:size|dim(?:ension)?)\s*[=:]\s*(\d+)",
            r"\bembedding\s*(?:dim(?:ension)?\s*)?[=:]\s*(\d+)",
            r"\b(\d{3,4})[- ]dimensional",
        ),
        "ffn_dim": first_int(
            r"\bffn.?(?:dim|size)\s*[=:]\s*(\d+)",
            r"\bintermediate.?size\s*[=:]\s*(\d+)",
            r"\bfeed.?forward.*?(\d{3,4})\b",
        ),
        "patch_size": first_int(
            r"\bpatch.?size\s*[=:]\s*(\d+)",
            r"\b(\d+)\s*[×x]\s*\d+\s*patch",
        ),
        "image_size": first_int(
            r"\b(\d{3})\s*[×x]\s*\d+\s*(?:image|resolution|pixel)",
        ),
        "channels": first_int(
            r"\b(\d+)\s*(?:filter|channel)s?(?:\s|$|,)",
            r"\bC\s*=\s*(\d+)\b",
        ),
        "num_classes": first_int(
            r"\b(\d{2,5})\s*class",
            r"\bnum_?classes\s*[=:]\s*(\d+)",
        ),
        "vocab_size": first_int(
            r"\bvocab(?:ulary)?\s*(?:size\s*)?[=:]\s*(\d+)",
            r"\b(\d{4,6})\s*-?\s*(?:token|subword)",
        ),
        "seq_len": first_int(
            r"\bmax(?:imum)?\s*(?:sequence|seq)\s*length\s*[=:]\s*(\d+)",
            r"\bcontext\s*(?:window|length)\s*[=:]\s*(\d+)",
            r"\bseq(?:uence)?\s*(?:length|len)\s*[=:]\s*(\d+)",
        ),
    }


# ---------------------------------------------------------------------------
# Primary architecture detection from knowledge graph
# ---------------------------------------------------------------------------

def _detect_primary_arch(
    knowledge_graph: Dict,
) -> Tuple[str, Optional[str]]:
    """Return (canonical_arch_name, arch_href) from the knowledge graph."""
    nodes = knowledge_graph.get("nodes", [])
    edges = knowledge_graph.get("edges", [])
    node_by_id: Dict[str, Dict] = {n["id"]: n for n in nodes}
    paper_node = next((n for n in nodes if n.get("type") == "paper"), None)
    if not paper_node:
        return "unknown", None

    paper_id = paper_node["id"]
    for relation in ("introduces", "uses"):
        for edge in edges:
            if edge.get("source") == paper_id and edge.get("relation") == relation:
                target = node_by_id.get(edge.get("target", ""), {})
                if target.get("type") == "architecture":
                    return target["name"], target.get("href")

    return "unknown", None


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _compute_confidence(
    arch_name: str,
    kg_edges: List[Dict],
    sections: List[Dict],
    equations: List[Dict],
    signals: Dict[str, Optional[int]],
) -> float:
    score = 0.0

    introduces = any(e.get("relation") == "introduces" for e in kg_edges)
    uses_known = arch_name != "unknown" and any(e.get("relation") == "uses" for e in kg_edges)

    if introduces:
        score += 0.35
    elif uses_known:
        score += 0.15

    section_titles_lower = {s.get("title", "").lower() for s in sections}
    structural_titles = {"methods", "approach", "method", "model architecture", "model", "architecture"}
    if section_titles_lower & structural_titles:
        score += 0.20

    n_signals = sum(1 for v in signals.values() if v is not None)
    score += min(0.25, n_signals * 0.05)

    if len(equations) >= 3:
        score += 0.10
    elif equations:
        score += 0.05

    return round(min(score, 1.0), 2)


# ---------------------------------------------------------------------------
# Tensor-flow simulation with FLOPs estimation
# ---------------------------------------------------------------------------

def _simulate_tensor_flow(
    components: List[Component],
    connections: List[Connection],
    input_shape: List[int],
    flops_engine: FLOPsEngine,
) -> List[Dict[str, Any]]:
    sequential_prev: Dict[str, str] = {}
    for conn in connections:
        if conn.connection_type == "sequential" and conn.target not in sequential_prev:
            sequential_prev[conn.target] = conn.source

    shape_by_id: Dict[str, List[int]] = {}
    flow: List[Dict[str, Any]] = []

    for i, comp in enumerate(components):
        prev_id = sequential_prev.get(comp.id)
        if prev_id and prev_id in shape_by_id:
            in_shape = shape_by_id[prev_id]
        elif i == 0:
            in_shape = input_shape
        else:
            in_shape = comp.shape

        out_shape = comp.shape
        shape_by_id[comp.id] = out_shape

        flops_type = comp.metadata.get("flops_type") or _COMP_TO_FLOPS_TYPE.get(comp.type, "")
        flops_mflops = 0.0
        params_m = 0.0

        if flops_type:
            if comp.type == "attention" and "ffn_dim" in comp.metadata:
                fr_mhsa = flops_engine.estimate(
                    node_id=comp.id,
                    node_type="multiheadattention",
                    in_shape=tuple(in_shape),
                    out_shape=tuple(out_shape),
                    params=comp.metadata,
                    label=comp.name,
                )
                fr_ffn = flops_engine.estimate(
                    node_id=comp.id + "_ffn",
                    node_type="feedforward",
                    in_shape=tuple(in_shape),
                    out_shape=tuple(out_shape),
                    params={
                        "embed_dim": comp.metadata.get("embed_dim", 768),
                        "ff_dim": comp.metadata.get("ffn_dim", 3072),
                    },
                    label=comp.name + " FFN",
                )
                flops_mflops = fr_mhsa.flops_mflops + fr_ffn.flops_mflops
                params_m = fr_mhsa.params_M + fr_ffn.params_M
            else:
                fr = flops_engine.estimate(
                    node_id=comp.id,
                    node_type=flops_type,
                    in_shape=tuple(in_shape),
                    out_shape=tuple(out_shape),
                    params=comp.metadata,
                    label=comp.name,
                )
                flops_mflops = fr.flops_mflops
                params_m = fr.params_M

        repeat = int(comp.metadata.get("repeat_count", 1))
        flow.append({
            "component_id": comp.id,
            "component_name": comp.name,
            "type": comp.type,
            "input_shape": list(in_shape),
            "output_shape": list(out_shape),
            "flops_mflops": round(flops_mflops * repeat, 4),
            "params_M": round(params_m * repeat, 4),
        })

    return flow


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------

_RESNET_CONFIGS: Dict[int, Tuple[List[int], str, List[int]]] = {
    18:  ([2, 2, 2, 2], "basic",      [64,  128,  256,  512]),
    34:  ([3, 4, 6, 3], "basic",      [64,  128,  256,  512]),
    50:  ([3, 4, 6, 3], "bottleneck", [256, 512,  1024, 2048]),
    101: ([3, 4, 23, 3],"bottleneck", [256, 512,  1024, 2048]),
    152: ([3, 8, 36, 3],"bottleneck", [256, 512,  1024, 2048]),
}


def _resnet_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    depth = signals.get("depth") or 50
    # Find nearest known config
    if depth not in _RESNET_CONFIGS:
        distances = {k: abs(k - depth) for k in _RESNET_CONFIGS}
        depth = min(distances, key=distances.get)

    stage_d, variant, out_ch = _RESNET_CONFIGS[depth]
    n_classes = signals.get("num_classes") or n_classes or 1000

    components = [
        Component("stem",   f"Stem (7×7 Conv, stride=2)", "conv",
                  [1, 64, 112, 112],
                  {"in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2,
                   "channels": 64}),
        Component("pool1",  "Max Pool (3×3, stride=2)",   "pooling",
                  [1, 64, 56, 56],
                  {"kernel_size": 3, "stride": 2}),
        Component("stage1", f"Stage 1 ×{stage_d[0]} ({variant})", "conv",
                  [1, out_ch[0], 56, 56],
                  {"channels": out_ch[0], "stride": 1, "kernel_size": 3,
                   "has_residual": True, "repeat_count": stage_d[0],
                   "flops_type": "residualblock"}),
        Component("stage2", f"Stage 2 ×{stage_d[1]} ({variant})", "conv",
                  [1, out_ch[1], 28, 28],
                  {"channels": out_ch[1], "stride": 2, "kernel_size": 3,
                   "has_residual": True, "repeat_count": stage_d[1],
                   "flops_type": "residualblock"}),
        Component("stage3", f"Stage 3 ×{stage_d[2]} ({variant})", "conv",
                  [1, out_ch[2], 14, 14],
                  {"channels": out_ch[2], "stride": 2, "kernel_size": 3,
                   "has_residual": True, "repeat_count": stage_d[2],
                   "flops_type": "residualblock"}),
        Component("stage4", f"Stage 4 ×{stage_d[3]} ({variant})", "conv",
                  [1, out_ch[3], 7, 7],
                  {"channels": out_ch[3], "stride": 2, "kernel_size": 3,
                   "has_residual": True, "repeat_count": stage_d[3],
                   "flops_type": "residualblock"}),
        Component("avgpool","Global Avg Pool",             "pooling",
                  [1, out_ch[3], 1, 1], {}),
        Component("head",   f"Classifier (→{n_classes})",  "head",
                  [1, n_classes],
                  {"in_features": out_ch[3], "out_features": n_classes}),
    ]
    connections = [
        Connection("stem",   "pool1",  "sequential"),
        Connection("pool1",  "stage1", "sequential"),
        Connection("stage1", "stage2", "sequential"),
        Connection("stage2", "stage3", "sequential"),
        Connection("stage3", "stage4", "sequential"),
        Connection("stage4", "avgpool","sequential"),
        Connection("avgpool","head",   "sequential"),
    ]
    return components, connections, [1, 3, 224, 224], [1, n_classes]


def _vit_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    embed_dim = signals.get("d_model") or 768
    num_heads  = signals.get("num_heads") or 12
    num_layers = signals.get("depth") or 12
    patch_size = signals.get("patch_size") or 16
    img_size   = signals.get("image_size") or 224
    ffn_dim    = signals.get("ffn_dim") or embed_dim * 4
    n_classes  = signals.get("num_classes") or n_classes or 1000

    num_patches = (img_size // patch_size) ** 2
    seq_len = num_patches + 1  # +1 for CLS token

    components = [
        Component("patch_embed", f"Patch Embedding ({patch_size}×{patch_size})", "embedding",
                  [1, num_patches, embed_dim],
                  {"patch_size": patch_size, "embed_dim": embed_dim, "in_channels": 3}),
        Component("cls_pos",     "CLS + Positional Embedding", "embedding",
                  [1, seq_len, embed_dim],
                  {"embed_dim": embed_dim, "vocab_size": seq_len}),
        Component("encoder",     f"Transformer Encoder ×{num_layers}", "attention",
                  [1, seq_len, embed_dim],
                  {"embed_dim": embed_dim, "num_heads": num_heads, "ffn_dim": ffn_dim,
                   "seq_len": seq_len, "repeat_count": num_layers}),
        Component("norm",        "Layer Norm",                 "normalization",
                  [1, seq_len, embed_dim],
                  {}),
        Component("head",        f"MLP Head (→{n_classes})",   "head",
                  [1, n_classes],
                  {"in_features": embed_dim, "out_features": n_classes}),
    ]
    connections = [
        Connection("patch_embed", "cls_pos",  "sequential"),
        Connection("cls_pos",     "encoder",  "sequential"),
        Connection("encoder",     "norm",     "sequential"),
        Connection("norm",        "head",     "sequential"),
    ]
    return components, connections, [1, 3, img_size, img_size], [1, n_classes]


def _unet_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    base_ch  = signals.get("channels") or 64
    n_out    = signals.get("num_classes") or n_classes or 2

    c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16

    components = [
        Component("enc1", f"Encoder 1 (×2 Conv, {c1}ch)",    "conv",
                  [1, c1, 256, 256],
                  {"in_channels": 1, "out_channels": c1, "channels": c1, "kernel_size": 3}),
        Component("enc2", f"Encoder 2 (×2 Conv, {c2}ch)",    "conv",
                  [1, c2, 128, 128],
                  {"in_channels": c1, "out_channels": c2, "channels": c2, "kernel_size": 3}),
        Component("enc3", f"Encoder 3 (×2 Conv, {c3}ch)",    "conv",
                  [1, c3, 64, 64],
                  {"in_channels": c2, "out_channels": c3, "channels": c3, "kernel_size": 3}),
        Component("enc4", f"Encoder 4 (×2 Conv, {c4}ch)",    "conv",
                  [1, c4, 32, 32],
                  {"in_channels": c3, "out_channels": c4, "channels": c4, "kernel_size": 3}),
        Component("bottleneck", f"Bottleneck (×2 Conv, {c5}ch)", "conv",
                  [1, c5, 16, 16],
                  {"in_channels": c4, "out_channels": c5, "channels": c5, "kernel_size": 3}),
        Component("up4",  "Upsample ×2",                       "upsample",
                  [1, c5, 32, 32], {}),
        Component("dec4", f"Decoder 4 (×2 Conv, {c4}ch)",    "conv",
                  [1, c4, 32, 32],
                  {"in_channels": c5 + c4, "out_channels": c4, "channels": c4, "kernel_size": 3}),
        Component("up3",  "Upsample ×2",                       "upsample",
                  [1, c4, 64, 64], {}),
        Component("dec3", f"Decoder 3 (×2 Conv, {c3}ch)",    "conv",
                  [1, c3, 64, 64],
                  {"in_channels": c4 + c3, "out_channels": c3, "channels": c3, "kernel_size": 3}),
        Component("up2",  "Upsample ×2",                       "upsample",
                  [1, c3, 128, 128], {}),
        Component("dec2", f"Decoder 2 (×2 Conv, {c2}ch)",    "conv",
                  [1, c2, 128, 128],
                  {"in_channels": c3 + c2, "out_channels": c2, "channels": c2, "kernel_size": 3}),
        Component("up1",  "Upsample ×2",                       "upsample",
                  [1, c2, 256, 256], {}),
        Component("dec1", f"Decoder 1 (×2 Conv, {c1}ch)",    "conv",
                  [1, c1, 256, 256],
                  {"in_channels": c2 + c1, "out_channels": c1, "channels": c1, "kernel_size": 3}),
        Component("head", f"Output Conv (1×1, {n_out}ch)",   "head",
                  [1, n_out, 256, 256],
                  {"in_features": c1, "out_features": n_out}),
    ]
    connections = [
        Connection("enc1",       "enc2",       "sequential"),
        Connection("enc2",       "enc3",       "sequential"),
        Connection("enc3",       "enc4",       "sequential"),
        Connection("enc4",       "bottleneck", "sequential"),
        Connection("bottleneck", "up4",        "sequential"),
        Connection("up4",        "dec4",       "sequential"),
        Connection("dec4",       "up3",        "sequential"),
        Connection("up3",        "dec3",       "sequential"),
        Connection("dec3",       "up2",        "sequential"),
        Connection("up2",        "dec2",       "sequential"),
        Connection("dec2",       "up1",        "sequential"),
        Connection("up1",        "dec1",       "sequential"),
        Connection("dec1",       "head",       "sequential"),
        # Skip connections
        Connection("enc1", "dec1", "concat"),
        Connection("enc2", "dec2", "concat"),
        Connection("enc3", "dec3", "concat"),
        Connection("enc4", "dec4", "concat"),
    ]
    return components, connections, [1, 1, 256, 256], [1, n_out, 256, 256]


def _transformer_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    embed_dim  = signals.get("d_model") or 512
    num_heads  = signals.get("num_heads") or 8
    num_layers = signals.get("depth") or 6
    ffn_dim    = signals.get("ffn_dim") or embed_dim * 4
    seq_len    = signals.get("seq_len") or 128
    vocab_size = signals.get("vocab_size") or 10000
    n_out      = signals.get("num_classes") or vocab_size

    components = [
        Component("tok_embed",      "Token Embedding",              "embedding",
                  [1, seq_len, embed_dim],
                  {"embed_dim": embed_dim, "vocab_size": vocab_size,
                   "flops_type": "token_embedding"}),
        Component("pos_embed",      "Positional Encoding",          "embedding",
                  [1, seq_len, embed_dim],
                  {"embed_dim": embed_dim, "vocab_size": seq_len,
                   "flops_type": "positionalembedding"}),
        Component("enc_stack",      f"Encoder Stack ×{num_layers}", "attention",
                  [1, seq_len, embed_dim],
                  {"embed_dim": embed_dim, "num_heads": num_heads, "ffn_dim": ffn_dim,
                   "seq_len": seq_len, "repeat_count": num_layers}),
        Component("dec_stack",      f"Decoder Stack ×{num_layers}", "attention",
                  [1, seq_len, embed_dim],
                  {"embed_dim": embed_dim, "num_heads": num_heads, "ffn_dim": ffn_dim,
                   "seq_len": seq_len, "repeat_count": num_layers}),
        Component("head",           f"Linear + Softmax (→{n_out})", "head",
                  [1, n_out],
                  {"in_features": embed_dim, "out_features": n_out}),
    ]
    connections = [
        Connection("tok_embed",  "pos_embed",  "sequential"),
        Connection("pos_embed",  "enc_stack",  "sequential"),
        Connection("enc_stack",  "dec_stack",  "cross_attention"),
        Connection("dec_stack",  "head",       "sequential"),
    ]
    return components, connections, [1, seq_len], [1, n_out]


def _gan_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    latent_dim = 100
    img_ch     = signals.get("channels") or 3
    img_size   = signals.get("image_size") or 64

    components = [
        Component("noise",   "Noise Input z",          "embedding",
                  [1, latent_dim],
                  {"embed_dim": latent_dim, "vocab_size": latent_dim,
                   "flops_type": "token_embedding"}),
        Component("fc",      "FC + Reshape (4×4×512)", "mlp",
                  [1, 512, 4, 4],
                  {"embed_dim": latent_dim, "ff_dim": 4 * 4 * 512}),
        Component("up1",     "ConvTranspose ×2 (256ch)","upsample",
                  [1, 256, 8, 8], {}),
        Component("up2",     "ConvTranspose ×2 (128ch)","upsample",
                  [1, 128, 16, 16], {}),
        Component("up3",     "ConvTranspose ×2 (64ch)", "upsample",
                  [1, 64, 32, 32], {}),
        Component("out_conv",f"Output Conv Tanh ({img_ch}ch)", "head",
                  [1, img_ch, img_size, img_size],
                  {"in_features": 64, "out_features": img_ch}),
    ]
    connections = [
        Connection("noise", "fc",    "sequential"),
        Connection("fc",    "up1",   "sequential"),
        Connection("up1",   "up2",   "sequential"),
        Connection("up2",   "up3",   "sequential"),
        Connection("up3",   "out_conv","sequential"),
    ]
    return components, connections, [1, latent_dim], [1, img_ch, img_size, img_size]


def _vae_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    img_ch    = signals.get("channels") or 3
    img_size  = signals.get("image_size") or 64
    latent_d  = signals.get("d_model") or 128

    components = [
        Component("enc_conv1", "Encoder Conv 1 (32ch)", "conv",
                  [1, 32, img_size // 2, img_size // 2],
                  {"in_channels": img_ch, "out_channels": 32, "channels": 32,
                   "kernel_size": 4, "stride": 2}),
        Component("enc_conv2", "Encoder Conv 2 (64ch)", "conv",
                  [1, 64, img_size // 4, img_size // 4],
                  {"in_channels": 32, "out_channels": 64, "channels": 64,
                   "kernel_size": 4, "stride": 2}),
        Component("enc_fc",    "Encoder FC → μ, σ",    "mlp",
                  [1, latent_d * 2],
                  {"embed_dim": 64 * (img_size // 4) ** 2, "ff_dim": latent_d * 2}),
        Component("latent",    f"Latent Space z ({latent_d}d)", "embedding",
                  [1, latent_d],
                  {"embed_dim": latent_d, "vocab_size": latent_d,
                   "flops_type": "token_embedding"}),
        Component("dec_fc",    "Decoder FC",            "mlp",
                  [1, 64 * (img_size // 4) ** 2],
                  {"embed_dim": latent_d,
                   "ff_dim": 64 * (img_size // 4) ** 2}),
        Component("dec_conv1", "Decoder ConvT ×2 (32ch)","upsample",
                  [1, 32, img_size // 2, img_size // 2], {}),
        Component("dec_conv2", f"Output Conv (→{img_ch}ch)", "head",
                  [1, img_ch, img_size, img_size],
                  {"in_features": 32, "out_features": img_ch}),
    ]
    connections = [
        Connection("enc_conv1", "enc_conv2", "sequential"),
        Connection("enc_conv2", "enc_fc",    "sequential"),
        Connection("enc_fc",    "latent",    "sequential"),
        Connection("latent",    "dec_fc",    "sequential"),
        Connection("dec_fc",    "dec_conv1", "sequential"),
        Connection("dec_conv1", "dec_conv2", "sequential"),
    ]
    return components, connections, [1, img_ch, img_size, img_size], [1, img_ch, img_size, img_size]


def _diffusion_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    img_ch   = signals.get("channels") or 3
    img_size = signals.get("image_size") or 64
    base_ch  = 64

    components = [
        Component("noise_in",   "Noisy Image Input x_t",   "conv",
                  [1, base_ch, img_size, img_size],
                  {"in_channels": img_ch, "out_channels": base_ch, "channels": base_ch, "kernel_size": 3}),
        Component("time_embed", "Timestep Embedding",       "embedding",
                  [1, base_ch * 4],
                  {"embed_dim": base_ch * 4, "vocab_size": 1000,
                   "flops_type": "token_embedding"}),
        Component("unet_enc1",  "U-Net Encoder 1 (64ch)",  "conv",
                  [1, base_ch, img_size, img_size],
                  {"channels": base_ch, "kernel_size": 3}),
        Component("unet_enc2",  "U-Net Encoder 2 (128ch)", "conv",
                  [1, base_ch * 2, img_size // 2, img_size // 2],
                  {"channels": base_ch * 2, "kernel_size": 3}),
        Component("bottleneck", "Bottleneck + Self-Attn",  "attention",
                  [1, (img_size // 4) ** 2, base_ch * 4],
                  {"embed_dim": base_ch * 4, "num_heads": 8,
                   "ffn_dim": base_ch * 16,
                   "seq_len": (img_size // 4) ** 2}),
        Component("unet_dec2",  "U-Net Decoder 2 (128ch)", "upsample",
                  [1, base_ch * 2, img_size // 2, img_size // 2], {}),
        Component("unet_dec1",  "U-Net Decoder 1 (64ch)",  "upsample",
                  [1, base_ch, img_size, img_size], {}),
        Component("out_conv",   f"Output Conv (→{img_ch}ch)", "head",
                  [1, img_ch, img_size, img_size],
                  {"in_features": base_ch, "out_features": img_ch}),
    ]
    connections = [
        Connection("noise_in",   "unet_enc1",  "sequential"),
        Connection("unet_enc1",  "unet_enc2",  "sequential"),
        Connection("unet_enc2",  "bottleneck", "sequential"),
        Connection("bottleneck", "unet_dec2",  "sequential"),
        Connection("unet_dec2",  "unet_dec1",  "sequential"),
        Connection("unet_dec1",  "out_conv",   "sequential"),
        # Skip connections
        Connection("unet_enc1", "unet_dec1", "concat"),
        Connection("unet_enc2", "unet_dec2", "concat"),
        # Time embedding feeds into each U-Net block
        Connection("time_embed", "unet_enc1", "residual"),
    ]
    return components, connections, [1, img_ch, img_size, img_size], [1, img_ch, img_size, img_size]


def _lstm_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    hidden = signals.get("d_model") or 256
    seq_len = signals.get("seq_len") or 100
    vocab   = signals.get("vocab_size") or 10000
    n_out   = signals.get("num_classes") or n_classes or vocab
    layers  = signals.get("depth") or 2

    components = [
        Component("embed",  "Token Embedding",       "embedding",
                  [1, seq_len, hidden],
                  {"embed_dim": hidden, "vocab_size": vocab,
                   "flops_type": "token_embedding"}),
        Component("lstm",   f"LSTM ×{layers} Layers", "mlp",
                  [1, seq_len, hidden],
                  {"embed_dim": hidden, "ff_dim": hidden * 4, "repeat_count": layers}),
        Component("dropout","Dropout",               "normalization",
                  [1, seq_len, hidden], {}),
        Component("head",   f"Output FC (→{n_out})", "head",
                  [1, n_out],
                  {"in_features": hidden, "out_features": n_out}),
    ]
    connections = [
        Connection("embed",   "lstm",    "sequential"),
        Connection("lstm",    "dropout", "sequential"),
        Connection("dropout", "head",    "sequential"),
    ]
    return components, connections, [1, seq_len], [1, n_out]


def _generic_template(
    signals: Dict[str, Optional[int]],
    n_classes: int,
) -> Tuple[List[Component], List[Connection], List[int], List[int]]:
    embed_dim = signals.get("d_model") or 256
    n_out     = signals.get("num_classes") or n_classes or 1000

    components = [
        Component("input",   "Input",               "embedding",
                  [1, embed_dim],
                  {"embed_dim": embed_dim, "vocab_size": embed_dim,
                   "flops_type": "token_embedding"}),
        Component("encoder", "Encoder",              "mlp",
                  [1, embed_dim],
                  {"embed_dim": embed_dim, "ff_dim": embed_dim * 4}),
        Component("head",    f"Output (→{n_out})", "head",
                  [1, n_out],
                  {"in_features": embed_dim, "out_features": n_out}),
    ]
    connections = [
        Connection("input",   "encoder", "sequential"),
        Connection("encoder", "head",    "sequential"),
    ]
    return components, connections, [1, embed_dim], [1, n_out]


# Maps canonical architecture names to template functions
_TEMPLATE_REGISTRY: Dict[str, Any] = {
    "ResNet":            _resnet_template,
    "VGG":               _resnet_template,
    "AlexNet":           _resnet_template,
    "GoogLeNet":         _resnet_template,
    "InceptionV3":       _resnet_template,
    "DenseNet":          _resnet_template,
    "EfficientNet":      _resnet_template,
    "ViT":               _vit_template,
    "Vision Transformer":_vit_template,
    "Swin Transformer":  _vit_template,
    "U-Net":             _unet_template,
    "FCN":               _unet_template,
    "Transformer":       _transformer_template,
    "BERT":              _transformer_template,
    "GPT":               _transformer_template,
    "T5":                _transformer_template,
    "LLaMA":             _transformer_template,
    "Seq2Seq":           _transformer_template,
    "GAN":               _gan_template,
    "VAE":               _vae_template,
    "Autoencoder":       _vae_template,
    "DDPM":              _diffusion_template,
    "Stable Diffusion":  _diffusion_template,
    "Latent Diffusion":  _diffusion_template,
    "LSTM":              _lstm_template,
    "GRU":               _lstm_template,
    "RNN":               _lstm_template,
    "MoE":               _transformer_template,
    "CLIP":              _vit_template,
    "DINO":              _vit_template,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reconstruct_architecture(
    paper_title: str,
    abstract: str,
    sections: List[Dict],
    equations: List[Dict],
    knowledge_graph: Dict,
    paper_id: Optional[int] = None,
    n_classes: int = 1000,
) -> Dict[str, Any]:
    """
    Build an ArchitectureBlueprint from paper text and knowledge graph.

    Returns a dict ready for JSON serialization and persistence in
    paper.architecture_graph.ingestion.architecture_blueprint.
    """
    signals = _extract_signals(paper_title, abstract, sections)
    arch_name, arch_href = _detect_primary_arch(knowledge_graph)
    template_fn = _TEMPLATE_REGISTRY.get(arch_name, _generic_template)

    components, connections, input_shape, output_shape = template_fn(
        signals, n_classes
    )

    flops_engine = FLOPsEngine()
    tensor_flow = _simulate_tensor_flow(
        components, connections, input_shape, flops_engine
    )

    total_flops = sum(s["flops_mflops"] for s in tensor_flow)
    total_params_m = sum(s["params_M"] for s in tensor_flow)
    total_params = int(total_params_m * 1_000_000)

    confidence = _compute_confidence(
        arch_name=arch_name,
        kg_edges=knowledge_graph.get("edges", []),
        sections=sections,
        equations=equations,
        signals=signals,
    )

    blueprint_id = f"bp-{paper_id or 0}-{arch_name.lower().replace(' ', '-')}"

    blueprint = ArchitectureBlueprint(
        id=blueprint_id,
        name=arch_name if arch_name != "unknown" else paper_title,
        paper_id=paper_id,
        components=components,
        connections=connections,
        input_shape=input_shape,
        output_shape=output_shape,
        estimated_parameters=total_params,
        estimated_flops=round(total_flops, 4),
        confidence_score=confidence,
        tensor_flow=tensor_flow,
        architecture_href=arch_href,
    )
    return blueprint.to_dict()
