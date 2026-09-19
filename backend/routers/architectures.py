import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Paper
from core.architecture_graph import ArchitectureGraph, GraphEdge, GraphNode
from core.rag.diff_engine import GraphDiffEngine
from core.rag.knowledge_graph import KnowledgeGraph

router = APIRouter(prefix="/api/architectures", tags=["Architectures"])

# Catalogue slugs whose backing paper is titled by a different name.
_SLUG_ALIASES = {
    "vit": "vision transformer",
    "vggnet": "vgg16",
    "googlenet-inception-v1": "googlenet",
    "u-net": "u-net",
    "mobilenet-v1": "mobilenet",
    "efficientnet": "efficientnet-b0",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def resolve_paper_by_slug(db: Session, slug: str) -> Paper | None:
    """Map a catalogue slug ("resnet-50", "vit", "u-net") to the paper that backs it.

    The catalogue uses hyphenated lowercase slugs; papers are titled freely
    ("ResNet50", "Vision Transformer", "MobileNetV2"). A raw ILIKE substring
    match on the title found almost none of them -- "vit", "bert", "resnet-50"
    all 404ed -- so the compare page was dead for its primary use case.
    Both sides are reduced to [a-z0-9]; exact beats prefix beats substring;
    papers with an architecture graph beat those without; ties go to the
    shortest title so "resnet" picks ResNet18, not ResNet50.
    """
    wanted = _norm(_SLUG_ALIASES.get(slug.lower(), slug))
    if not wanted:
        return None
    best: tuple[int, int, int, Paper] | None = None  # (score, has_graph, -len, paper)
    for paper in db.query(Paper).all():
        title = _norm(paper.title)
        if not title:
            continue
        if title == wanted:
            score = 3
        elif title.startswith(wanted) or wanted.startswith(title):
            score = 2
        elif wanted in title or title in wanted:
            score = 1
        else:
            continue
        key = (score, 1 if paper.architecture_graph else 0, -len(title), paper)
        if best is None or key[:3] > best[:3]:
            best = key
    return best[3] if best else None


def dict_to_arch_graph(name: str, data: dict) -> ArchitectureGraph:
    graph = ArchitectureGraph(name=name, metadata=data.get("metadata", {}))
    for n_data in data.get("nodes", []):
        node = GraphNode(
            id=n_data.get("id", ""),
            type=n_data.get("type", ""),
            label=n_data.get("label", ""),
            params=n_data.get("params", {}),
            block=n_data.get("block"),
            description=n_data.get("description"),
            semantic_params=n_data.get("semantic_params", {}),
        )
        graph.add_node(node)
    for e_data in data.get("edges", []):
        graph.add_edge(
            e_data.get("source"), e_data.get("target"), edge_type=e_data.get("edge_type", "flow")
        )
    return graph


@router.get("/compare")
def compare_architectures(
    paper_a: int | None = None,
    paper_b: int | None = None,
    a_slug: str | None = None,
    b_slug: str | None = None,
    db: Session = Depends(get_db),
):
    if not ((paper_a and paper_b) or (a_slug and b_slug)):
        raise HTTPException(
            status_code=422, detail="Must provide either paper_a and paper_b, or a_slug and b_slug"
        )

    pa = db.query(Paper).filter(Paper.id == paper_a).first() if paper_a else None
    if pa is None and a_slug:
        pa = resolve_paper_by_slug(db, a_slug)
    pb = db.query(Paper).filter(Paper.id == paper_b).first() if paper_b else None
    if pb is None and b_slug:
        pb = resolve_paper_by_slug(db, b_slug)

    # Say which side failed. "One or both papers not found" left the user
    # guessing which of two selections to change.
    missing = [
        label
        for label, paper in ((a_slug or f"paper {paper_a}", pa), (b_slug or f"paper {paper_b}", pb))
        if paper is None
    ]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"No analysed paper backs {', '.join(repr(m) for m in missing)} yet; "
            "pick an architecture that has been processed.",
        )

    lacking = [p.title for p in (pa, pb) if not p.architecture_graph]
    if lacking:
        return {
            "status": "incomplete",
            "message": f"No architecture graph yet for {', '.join(repr(t) for t in lacking)}.",
        }

    graph_a = dict_to_arch_graph(pa.title, pa.architecture_graph)
    graph_b = dict_to_arch_graph(pb.title, pb.architecture_graph)

    engine = GraphDiffEngine()
    try:
        diff_result = engine.compare(graph_a, graph_b)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "paper_a": {"id": pa.id, "title": pa.title},
        "paper_b": {"id": pb.id, "title": pb.title},
        "diff": diff_result,
    }


SLUG_NODES = {
    "transformer": ["transformerblock", "multiheadattention", "layernorm", "linear"],
    "resnet": ["residualblock", "conv2d", "batchnorm2d"],
    "bert": ["transformer_encoder", "multiheadattention", "layernorm"],
    "vit": ["patchembedding", "transformerblock", "layernorm"],
    "gpt": ["transformer_decoder", "causal_attention", "layernorm"],
    "llama": ["transformer_decoder", "causal_attention", "layernorm"],
}


@router.get("/{slug}/knowledge-relations")
def get_knowledge_relations(slug: str):
    kg = KnowledgeGraph()
    G = kg.graph

    if slug in SLUG_NODES:
        node_names = set(SLUG_NODES[slug])
    else:
        node_names = set()
        parts = slug.split("-")
        for node in G.nodes:
            if slug in node:
                node_names.add(node)
            else:
                for part in parts:
                    if part and part in node:
                        node_names.add(node)
                        break

    if not node_names:
        return {"nodes": [], "constraints": []}

    response_nodes = []

    for node_id in node_names:
        if node_id in G.nodes:
            props = G.nodes[node_id]
            response_nodes.append(
                {
                    "id": node_id,
                    "type": props.get("type", ""),
                    "dimensionality": props.get("dimensionality", ""),
                }
            )

    constraints = []
    # Check all edges to see if they involve any of our nodes
    for u, v, edge_data in G.edges(data=True):
        if u in node_names or v in node_names:
            rel = edge_data.get("relation")
            if rel in ["COMPATIBLE", "INCOMPATIBLE", "REQUIRES_FLATTEN"]:
                constraints.append(
                    {"from": u, "to": v, "relation": rel, "reason": edge_data.get("reason", "")}
                )

    # ensure unique constraints
    seen = set()
    unique_constraints = []
    for c in constraints:
        tup = (c["from"], c["to"], c["relation"], c["reason"])
        if tup not in seen:
            seen.add(tup)
            unique_constraints.append(c)

    # ensure unique nodes
    unique_nodes = []
    seen_nodes = set()
    for n in response_nodes:
        if n["id"] not in seen_nodes:
            seen_nodes.add(n["id"])
            unique_nodes.append(n)

    return {"nodes": unique_nodes, "constraints": unique_constraints}
