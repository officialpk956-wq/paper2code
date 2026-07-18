from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Paper
from core.architecture_graph import ArchitectureGraph, GraphEdge, GraphNode
from core.rag.diff_engine import GraphDiffEngine
from core.rag.knowledge_graph import KnowledgeGraph

router = APIRouter(prefix="/api/architectures", tags=["Architectures"])

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
            semantic_params=n_data.get("semantic_params", {})
        )
        graph.add_node(node)
    for e_data in data.get("edges", []):
        graph.add_edge(e_data.get("source"), e_data.get("target"), edge_type=e_data.get("edge_type", "flow"))
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
        raise HTTPException(status_code=422, detail="Must provide either paper_a and paper_b, or a_slug and b_slug")

    # Fetch paper A
    pa = None
    if paper_a:
        pa = db.query(Paper).filter(Paper.id == paper_a).first()
    if not pa and a_slug:
        pa = db.query(Paper).filter(Paper.title.ilike(f"%{a_slug}%")).first()

    # Fetch paper B
    pb = None
    if paper_b:
        pb = db.query(Paper).filter(Paper.id == paper_b).first()
    if not pb and b_slug:
        pb = db.query(Paper).filter(Paper.title.ilike(f"%{b_slug}%")).first()

    if not pa or not pb:
        raise HTTPException(status_code=404, detail="One or both papers not found")

    if not pa.architecture_graph or not pb.architecture_graph:
        return {"status": "incomplete", "message": "One or both papers lack architecture data"}

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
        "diff": diff_result
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
            response_nodes.append({
                "id": node_id,
                "type": props.get("type", ""),
                "dimensionality": props.get("dimensionality", "")
            })

    constraints = []
    # Check all edges to see if they involve any of our nodes
    for u, v, edge_data in G.edges(data=True):
        if u in node_names or v in node_names:
            rel = edge_data.get("relation")
            if rel in ["COMPATIBLE", "INCOMPATIBLE", "REQUIRES_FLATTEN"]:
                constraints.append({
                    "from": u,
                    "to": v,
                    "relation": rel,
                    "reason": edge_data.get("reason", "")
                })

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

    return {
        "nodes": unique_nodes,
        "constraints": unique_constraints
    }
