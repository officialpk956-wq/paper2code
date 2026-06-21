"""
Knowledge extraction service for Phase 13B.

Deterministic (no LLM) extraction of architectures, concepts, datasets,
and metrics from ingested paper text.  Produces a KnowledgeGraph dict that
is persisted inside Paper.architecture_graph.ingestion.knowledge_graph.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Maps lowercase pattern → (canonical display name, optional content slug)
# Slug must match an entry in src/content/architectures/
ARCHITECTURE_REGISTRY: List[Tuple[str, str, Optional[str]]] = [
    # ── Transformers ──────────────────────────────────────────────────────
    (r"\btransformer\b",              "Transformer",        "transformer"),
    (r"\battention is all you need\b","Transformer",        "transformer"),
    (r"\bbert\b",                     "BERT",               "bert"),
    (r"\bgpt(?:-\d)?\b",              "GPT",                "gpt"),
    (r"\bvision transformer\b",       "Vision Transformer", "vit"),
    (r"\bvit\b",                      "ViT",                "vit"),
    (r"\bclip\b",                     "CLIP",               "clip"),
    (r"\bllama\b",                    "LLaMA",              "llama"),
    (r"\bt5\b",                       "T5",                 "t5"),
    (r"\bswin transformer\b",         "Swin Transformer",   "swin"),
    (r"\bswin\b",                     "Swin Transformer",   "swin"),
    # ── CNNs ─────────────────────────────────────────────────────────────
    (r"\bresnet\b",                   "ResNet",             "resnet"),
    (r"\bdeep residual\b",            "ResNet",             "resnet"),
    (r"\balexnet\b",                  "AlexNet",            "alexnet"),
    (r"\bvgg\b",                      "VGG",                "vgg16"),
    (r"\bgooglenet\b",                "GoogLeNet",          "googlenet"),
    (r"\binceptionv?3?\b",            "InceptionV3",        "inceptionv3"),
    (r"\bdensenet\b",                 "DenseNet",           "densenet"),
    (r"\bdense(?:ly connected)?\b",   "DenseNet",           "densenet"),
    (r"\befficientnet\b",             "EfficientNet",       "efficientnet"),
    (r"\blenet\b",                    "LeNet",              "lenet"),
    # ── Segmentation / Detection ──────────────────────────────────────────
    (r"\bu(?:-| )?net\b",             "U-Net",              "unet"),
    (r"\bdeeplabv?3\+?\b",            "DeepLabV3+",         "deeplabv3plus"),
    (r"\bfcn\b",                      "FCN",                "fcn"),
    (r"\byolo\b",                     "YOLO",               None),
    # ── Generative ───────────────────────────────────────────────────────
    (r"\bgan\b",                      "GAN",                "gan"),
    (r"\bgenerative adversarial\b",   "GAN",                "gan"),
    (r"\bddpm\b",                     "DDPM",               "diffusion"),
    (r"\bdenoising diffusion\b",      "DDPM",               "diffusion"),
    (r"\blatent diffusion\b",         "Latent Diffusion",   "stable-diffusion"),
    (r"\bstable diffusion\b",         "Stable Diffusion",   "stable-diffusion"),
    (r"\bvae\b",                      "VAE",                "vae"),
    (r"\bvariational autoencoder\b",  "VAE",                "vae"),
    (r"\bautoencoder\b",              "Autoencoder",        "ae"),
    # ── Sequential ───────────────────────────────────────────────────────
    (r"\blstm\b",                     "LSTM",               "lstm"),
    (r"\bgru\b",                      "GRU",                "gru"),
    (r"\brnn\b",                      "RNN",                "rnn"),
    (r"\bseq2seq\b",                  "Seq2Seq",            "seq2seq"),
    (r"\bsequence.to.sequence\b",     "Seq2Seq",            "seq2seq"),
    # ── Self-supervised / Misc ────────────────────────────────────────────
    (r"\bdino\b",                     "DINO",               "dino"),
    (r"\bmoe\b",                      "MoE",                "moe"),
    (r"\bmixture of experts\b",       "MoE",                "moe"),
    (r"\bmobilenet\b",                "MobileNet",          None),
]

CONCEPT_REGISTRY: List[Tuple[str, str]] = [
    (r"\battention mechanism\b",    "Attention Mechanism"),
    (r"\bself.attention\b",         "Self-Attention"),
    (r"\bcross.attention\b",        "Cross-Attention"),
    (r"\bmulti.head attention\b",   "Multi-Head Attention"),
    (r"\bresidual connection\b",    "Residual Connection"),
    (r"\bskip connection\b",        "Skip Connection"),
    (r"\blayer norm(?:alization)?\b","Layer Normalization"),
    (r"\bbatch norm(?:alization)?\b","Batch Normalization"),
    (r"\bgroup norm(?:alization)?\b","Group Normalization"),
    (r"\bdropout\b",                "Dropout"),
    (r"\bdiffusion process\b",      "Diffusion Process"),
    (r"\bcontrastive learning\b",   "Contrastive Learning"),
    (r"\btokenization\b",           "Tokenization"),
    (r"\bpositional encoding\b",    "Positional Encoding"),
    (r"\bfeed.forward network\b",   "Feed-Forward Network"),
    (r"\bself.supervised\b",        "Self-Supervised Learning"),
    (r"\bknowledge distillation\b", "Knowledge Distillation"),
    (r"\btransfer learning\b",      "Transfer Learning"),
    (r"\bfine.tun(?:ing|e)\b",      "Fine-Tuning"),
    (r"\bembedding(?:s)?\b",        "Embeddings"),
]

DATASET_REGISTRY: List[Tuple[str, str]] = [
    (r"\bimagenet\b",     "ImageNet"),
    (r"\bcoco\b",         "COCO"),
    (r"\bcifar.?10\b",    "CIFAR-10"),
    (r"\bcifar.?100\b",   "CIFAR-100"),
    (r"\bmnist\b",        "MNIST"),
    (r"\bwikitext\b",     "WikiText"),
    (r"\bsquad\b",        "SQuAD"),
    (r"\blibrispeech\b",  "LibriSpeech"),
    (r"\bade20k\b",       "ADE20K"),
    (r"\bpascal voc\b",   "Pascal VOC"),
    (r"\bkitti\b",        "KITTI"),
    (r"\bceleba\b",       "CelebA"),
    (r"\bopenwebtext\b",  "OpenWebText"),
    (r"\bc4\b",           "C4"),
    (r"\blaion\b",        "LAION"),
    (r"\bglue\b",         "GLUE"),
    (r"\bsuperglue\b",    "SuperGLUE"),
]

METRIC_REGISTRY: List[Tuple[str, str]] = [
    (r"\btop.1 accuracy\b",  "Top-1 Accuracy"),
    (r"\btop.5 accuracy\b",  "Top-5 Accuracy"),
    (r"\baccuracy\b",        "Accuracy"),
    (r"\bf1.?(?:score)?\b",  "F1 Score"),
    (r"\bbleu\b",            "BLEU"),
    (r"\brouge\b",           "ROUGE"),
    (r"\biou\b",             "IoU"),
    (r"\bdice\b",            "Dice"),
    (r"\bperplexity\b",      "Perplexity"),
    (r"\bmap\b",             "mAP"),
    (r"\bfid\b",             "FID"),
    (r"\bpsnr\b",            "PSNR"),
    (r"\bssim\b",            "SSIM"),
    (r"\bexact match\b",     "Exact Match"),
    (r"\bprecision\b",       "Precision"),
    (r"\brecall\b",          "Recall"),
]

# Static architecture → concept mappings
ARCH_CONCEPT_MAP: Dict[str, List[str]] = {
    "Transformer":        ["Multi-Head Attention", "Positional Encoding", "Layer Normalization", "Feed-Forward Network"],
    "Vision Transformer": ["Multi-Head Attention", "Positional Encoding", "Layer Normalization"],
    "ViT":                ["Multi-Head Attention", "Positional Encoding", "Layer Normalization"],
    "BERT":               ["Self-Attention", "Layer Normalization"],
    "GPT":                ["Self-Attention", "Layer Normalization"],
    "LLaMA":              ["Self-Attention", "Layer Normalization"],
    "T5":                 ["Cross-Attention", "Self-Attention"],
    "Swin Transformer":   ["Self-Attention"],
    "CLIP":               ["Contrastive Learning"],
    "DINO":               ["Self-Supervised Learning"],
    "ResNet":             ["Residual Connection", "Batch Normalization"],
    "DenseNet":           ["Skip Connection", "Batch Normalization"],
    "U-Net":              ["Skip Connection"],
    "GAN":                ["Dropout"],
    "DDPM":               ["Diffusion Process"],
    "Latent Diffusion":   ["Diffusion Process"],
    "Stable Diffusion":   ["Diffusion Process"],
    "VAE":                ["Embeddings"],
}

# Architecture lineage (derives_from)
ARCH_LINEAGE: Dict[str, List[str]] = {
    "BERT":               ["Transformer"],
    "GPT":                ["Transformer"],
    "ViT":                ["Transformer"],
    "Vision Transformer": ["Transformer"],
    "T5":                 ["Transformer"],
    "LLaMA":              ["Transformer"],
    "Swin Transformer":   ["Transformer", "Vision Transformer"],
    "Stable Diffusion":   ["Latent Diffusion", "VAE"],
    "Latent Diffusion":   ["DDPM"],
    "CLIP":               ["Transformer"],
    "DINO":               ["Vision Transformer"],
    "DenseNet":           ["ResNet"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _search_text(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, re.IGNORECASE))


def _safe_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "-", name.lower()).strip("-")


def _collect_text(
    title: str,
    abstract: str,
    sections: List[Dict[str, Any]],
) -> str:
    section_text = " ".join(s.get("content", "") for s in sections)
    return f"{title} {abstract} {section_text}"


# ---------------------------------------------------------------------------
# Extraction functions
# ---------------------------------------------------------------------------

def extract_architectures(
    title: str,
    abstract: str,
    sections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return unique architecture nodes found in the paper text."""
    full_text = _collect_text(title, abstract, sections)
    title_lower = title.lower()

    seen: set[str] = set()
    nodes: List[Dict[str, Any]] = []

    for pattern, canonical, slug in ARCHITECTURE_REGISTRY:
        if not _search_text(pattern, full_text):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)

        # Determine relation: "introduces" if the arch name is in the paper title
        in_title = _search_text(pattern, title_lower)

        nodes.append({
            "id": f"arch-{_safe_id(canonical)}",
            "name": canonical,
            "type": "architecture",
            "href": f"/architectures/{slug}" if slug else None,
            "_in_title": in_title,
        })

    return nodes


def extract_concepts(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return unique concept nodes found in section content."""
    section_text = " ".join(s.get("content", "") for s in sections)
    if not section_text.strip():
        return []

    seen: set[str] = set()
    nodes: List[Dict[str, Any]] = []

    for pattern, canonical in CONCEPT_REGISTRY:
        if not _search_text(pattern, section_text):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)

        nodes.append({
            "id": f"concept-{_safe_id(canonical)}",
            "name": canonical,
            "type": "concept",
            "href": None,
        })

    return nodes


def extract_datasets(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return unique dataset nodes found in section content."""
    section_text = " ".join(s.get("content", "") for s in sections)
    if not section_text.strip():
        return []

    seen: set[str] = set()
    nodes: List[Dict[str, Any]] = []

    for pattern, canonical in DATASET_REGISTRY:
        if not _search_text(pattern, section_text):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)

        nodes.append({
            "id": f"dataset-{_safe_id(canonical)}",
            "name": canonical,
            "type": "dataset",
            "href": None,
        })

    return nodes


def extract_metrics(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return unique metric nodes found in section content."""
    section_text = " ".join(s.get("content", "") for s in sections)
    if not section_text.strip():
        return []

    seen: set[str] = set()
    nodes: List[Dict[str, Any]] = []

    for pattern, canonical in METRIC_REGISTRY:
        if not _search_text(pattern, section_text):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)

        nodes.append({
            "id": f"metric-{_safe_id(canonical)}",
            "name": canonical,
            "type": "metric",
            "href": None,
        })

    return nodes


def _build_equation_nodes(equations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for eq in equations[:10]:
        label = eq.get("text", "")[:40]
        nodes.append({
            "id": eq.get("id", f"eq-{len(nodes)}"),
            "name": label,
            "type": "equation",
            "href": None,
        })
    return nodes


# ---------------------------------------------------------------------------
# Relationship extraction
# ---------------------------------------------------------------------------

def _build_edges(
    paper_id: str,
    arch_nodes: List[Dict[str, Any]],
    concept_nodes: List[Dict[str, Any]],
    dataset_nodes: List[Dict[str, Any]],
    metric_nodes: List[Dict[str, Any]],
    eq_nodes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    arch_by_name: Dict[str, Dict[str, Any]] = {n["name"]: n for n in arch_nodes}
    concept_by_name: Dict[str, Dict[str, Any]] = {n["name"]: n for n in concept_nodes}

    # Paper → Architecture
    for arch in arch_nodes:
        rel = "introduces" if arch.get("_in_title") else "uses"
        edges.append({"source": paper_id, "target": arch["id"], "relation": rel})

    # Paper → Dataset
    for ds in dataset_nodes:
        edges.append({"source": paper_id, "target": ds["id"], "relation": "evaluates_on"})

    # Paper → Metric
    for m in metric_nodes:
        edges.append({"source": paper_id, "target": m["id"], "relation": "reports"})

    # Paper → Equation
    for eq in eq_nodes:
        edges.append({"source": paper_id, "target": eq["id"], "relation": "derives_from"})

    # Architecture → Concept (static map)
    for arch in arch_nodes:
        for concept_name in ARCH_CONCEPT_MAP.get(arch["name"], []):
            if concept_name in concept_by_name:
                edges.append({
                    "source": arch["id"],
                    "target": concept_by_name[concept_name]["id"],
                    "relation": "uses",
                })

    # Architecture → Architecture (lineage)
    for arch in arch_nodes:
        for parent_name in ARCH_LINEAGE.get(arch["name"], []):
            if parent_name in arch_by_name:
                edges.append({
                    "source": arch["id"],
                    "target": arch_by_name[parent_name]["id"],
                    "relation": "derives_from",
                })

    return edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_knowledge_graph(
    paper_title: str,
    abstract: str,
    sections: List[Dict[str, Any]],
    equations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a KnowledgeGraph from extracted paper content.

    Returns a dict with ``nodes`` and ``edges`` lists suitable for JSON
    persistence and SVG rendering.
    """
    paper_id = f"paper-{_safe_id(paper_title[:60])}"

    arch_nodes = extract_architectures(paper_title, abstract, sections)
    concept_nodes = extract_concepts(sections)
    dataset_nodes = extract_datasets(sections)
    metric_nodes = extract_metrics(sections)
    eq_nodes = _build_equation_nodes(equations)

    paper_node: Dict[str, Any] = {
        "id": paper_id,
        "name": paper_title[:80],
        "type": "paper",
        "href": None,
    }

    # Strip private metadata before returning
    clean_arch = [{k: v for k, v in n.items() if k != "_in_title"} for n in arch_nodes]

    all_nodes = (
        [paper_node]
        + clean_arch
        + concept_nodes
        + dataset_nodes
        + metric_nodes
        + eq_nodes
    )

    edges = _build_edges(
        paper_id, arch_nodes, concept_nodes, dataset_nodes, metric_nodes, eq_nodes
    )

    return {"nodes": all_nodes, "edges": edges}
