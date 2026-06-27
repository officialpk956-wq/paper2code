import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
import dataclasses

from backend.database import get_db
from backend.models import Paper, PaperModule

from core.implementation.code_mapper import get_module_implementation, get_architecture_implementation
from core.implementation.training_config import get_training_config, get_hyperparameter_explanations
from core.implementation.cost_estimator import estimate_training_cost
from core.implementation.reproduction_cards import get_reproduction_card

from core.lab.diff_engine import compute_diff
from core.lab.mutator import apply_mutations, MUTATION_REGISTRY
from core.lab.hypothesis_engine import hypothesis_engine
from core.lab.tradeoff_analyzer import tradeoff_scatter, get_efficiency_frontiers, get_tradeoff_summary
from core.visualizer_resnet import build_resnet18_graph
from core.visualizer_vit import build_vit_graph
from core.visualizer_unet import build_unet_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Lab"])

def get_paper_info(paper) -> tuple[str, str, str, str]:
    arch_graph = paper.architecture_graph or {}
    classification = arch_graph.get("classification")
    status = arch_graph.get("status", "Published")
    support = arch_graph.get("support_level", "experimental")
    
    if classification:
        return paper.title, classification, status, support

    model_family = arch_graph.get("model_family")
    if model_family in ("cnn", "resnet"):
        return paper.title, "CNN", status, support
    elif model_family == "unet":
        return paper.title, "Encoder-Decoder", status, support
    elif model_family == "transformer":
        return paper.title, "Transformer", status, support

    title_lower = paper.title.lower()
    if "resnet" in title_lower:
        return paper.title, "CNN", status, support
    elif "transformer" in title_lower:
        return paper.title, "Transformer", status, support
    elif "u-net" in title_lower or "unet" in title_lower:
        return paper.title, "Encoder-Decoder", status, support
    return paper.title, "Unknown", status, support

class CostEstimatorRequest(BaseModel):
    architecture: str
    dataset_size: int = 1000000
    batch_size: int = 32
    gpu_type: str = "A100"
    epochs: Optional[int] = None
    mixed_precision: bool = False
    gradient_checkpointing: bool = False

class LabMutateRequest(BaseModel):
    architecture: str
    mutations: list[Dict[str, Any]]
    config: Dict[str, Any] = {}

class LabPredictRequest(BaseModel):
    architecture: str
    mutations: list[Dict[str, Any]]
    hypothesis: Dict[str, Any]

class LabExperimentRequest(BaseModel):
    architecture: str
    mutations: list[Dict[str, Any]]
    hypothesis: Optional[Dict[str, Any]] = None

def _build_base_graph(architecture: str, config: Dict[str, Any] = {}):
    if architecture == "ResNet":
        return build_resnet18_graph(
            base_channels=config.get("base_channels", 64),
            stages=config.get("stages", 4),
            blocks_per_stage=config.get("blocks_per_stage", 2),
        )
    elif architecture in ("Transformer", "ViT"):
        return build_vit_graph(
            hidden_size=config.get("hidden_size", 768),
            num_heads=config.get("num_heads", 12),
            depth=config.get("depth", 4),
        )
    elif architecture == "U-Net":
        return build_unet_graph(
            base_channels=config.get("base_channels", 64),
            stages=config.get("stages", 3),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported architecture: '{architecture}'. Valid: ResNet, Transformer, U-Net")

def _graph_to_api_dict(graph) -> Dict[str, Any]:
    return {
        "name": graph.name,
        "nodes": [
            {
                "id": n.id,
                "type": n.type,
                "label": n.label,
                "params": n.params or {},
                "block": n.block,
                "semantic_params": n.semantic_params or {},
            }
            for n in graph.nodes
        ],
        "edges": [
            {"source": e.source, "target": e.target, "edge_type": e.edge_type}
            for e in graph.edges
        ],
    }

def _graph_metrics(graph) -> Dict[str, Any]:
    from core.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
    m = estimate_metrics_from_graph(graph)
    mem = estimate_activation_memory(graph, batch_size=1, input_spatial=224)
    mem_mb = round(sum(r["mem_mb"] for r in mem), 2)
    return {
        "flops_score": m["total_flops_score"],
        "params": m["total_params_estimate"],
        "depth": m["depth"],
        "memory_mb": mem_mb,
    }

@router.get("/implementation/{paper_id}")
def get_architecture_implementation_view(paper_id: int, db: Session = Depends(get_db)):
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type, status, _ = get_paper_info(p)
    modules_data = []
    for m in p.modules:
        modules_data.append({
            "id": m.id,
            "layer_name": m.layer_name,
            "module_type": m.module_type,
            "explanation": m.explanation or "",
            "graph_nodes": m.graph_nodes if isinstance(m.graph_nodes, list) else [],
        })

    impl_view = get_architecture_implementation(
        paper_title=title,
        classification=arch_type,
        modules=modules_data,
    )
    impl_view["paper_id"] = paper_id
    impl_view["architecture_graph"] = p.architecture_graph or {}
    return impl_view

@router.get("/modules/{module_id}/implementation")
def get_module_implementation_view(module_id: int, db: Session = Depends(get_db)):
    m = db.query(PaperModule).filter(PaperModule.id == module_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Module not found")

    raw_graph = m.graph_nodes
    params = {}
    if isinstance(raw_graph, list) and raw_graph:
        params = raw_graph[0].get("params", {})
    elif isinstance(raw_graph, dict):
        nodes = raw_graph.get("nodes", [])
        if nodes:
            params = nodes[0].get("params", {})

    impl = get_module_implementation(m.module_type or "", params)
    paper_title = m.paper.title if m.paper else "Unknown"

    return {
        "module_id": module_id,
        "layer_name": m.layer_name,
        "module_type": m.module_type,
        "paper_title": paper_title,
        "explanation": m.explanation or "",
        "implementation": impl,
    }

@router.get("/training/{paper_id}")
def get_training_pipeline(paper_id: int, db: Session = Depends(get_db)):
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type, status, _ = get_paper_info(p)
    config = get_training_config(arch_type)
    return {
        "paper_id": paper_id,
        "paper_title": title,
        "architecture_type": arch_type,
        "training_config": config,
    }

@router.get("/hyperparameters")
def get_hyperparameters():
    return {"hyperparameters": get_hyperparameter_explanations()}

@router.post("/training-estimator")
def training_cost_estimator(request: CostEstimatorRequest):
    try:
        result = estimate_training_cost(
            architecture=request.architecture,
            dataset_size=request.dataset_size,
            batch_size=request.batch_size,
            gpu_type=request.gpu_type,
            epochs=request.epochs,
            mixed_precision=request.mixed_precision,
            gradient_checkpointing=request.gradient_checkpointing,
        )
        return result
    except Exception as e:
        logger.error(f"Cost estimator error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reproduction/{paper_id}")
def get_reproduction_card_view(paper_id: int, db: Session = Depends(get_db)):
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type, status, _ = get_paper_info(p)
    card = get_reproduction_card(arch_type)
    return {
        "paper_id": paper_id,
        "paper_title": title,
        "architecture_type": arch_type,
        "reproduction_card": card,
    }

@router.post("/lab/mutate")
def lab_mutate(request: LabMutateRequest):
    try:
        before_graph = _build_base_graph(request.architecture, request.config)
        after_graph  = apply_mutations(before_graph, request.mutations)

        diff = compute_diff(before_graph, after_graph)

        return {
            "architecture": request.architecture,
            "mutations_applied": request.mutations,
            "before": {
                "graph": _graph_to_api_dict(before_graph),
                "metrics": _graph_metrics(before_graph),
            },
            "after": {
                "graph": _graph_to_api_dict(after_graph),
                "metrics": _graph_metrics(after_graph),
            },
            "diff": diff,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab mutate error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lab/predict")
def lab_predict(request: LabPredictRequest):
    try:
        before_graph = _build_base_graph(request.architecture)
        after_graph  = apply_mutations(before_graph, request.mutations)
        diff         = compute_diff(before_graph, after_graph)

        hyp = dict(request.hypothesis)
        if "mutation_type" not in hyp and request.mutations:
            hyp["mutation_type"] = request.mutations[0].get("type", "")

        score_result = hypothesis_engine.score_prediction(hyp, diff)

        return {
            "architecture": request.architecture,
            "mutations_applied": request.mutations,
            "diff": diff,
            "before_metrics": _graph_metrics(before_graph),
            "after_metrics":  _graph_metrics(after_graph),
            "scoring": score_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab predict error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lab/experiment")
def lab_experiment(request: LabExperimentRequest):
    try:
        before_graph = _build_base_graph(request.architecture)
        after_graph  = apply_mutations(before_graph, request.mutations)
        diff         = compute_diff(before_graph, after_graph)

        result = hypothesis_engine.build_experiment_result(
            hypothesis=request.hypothesis,
            mutations_applied=request.mutations,
            architecture=request.architecture,
            actual_diff=diff,
        )

        return {
            "experiment": result,
            "diff": diff,
            "before_metrics": _graph_metrics(before_graph),
            "after_metrics":  _graph_metrics(after_graph),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab experiment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lab/tradeoffs")
def lab_tradeoffs(architecture: str = "ResNet"):
    try:
        base_graph = _build_base_graph(architecture)
        points     = tradeoff_scatter(base_graph, architecture)
        frontier   = get_efficiency_frontiers(points)

        summaries = {mut: get_tradeoff_summary(mut) for mut in MUTATION_REGISTRY.keys()}

        return {
            "architecture": architecture,
            "scatter_points": frontier,
            "tradeoff_summaries": summaries,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab tradeoffs error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lab/prediction-prompt")
def lab_prediction_prompt(mutation_type: str = "increase_depth", architecture: str = "ResNet"):
    try:
        prompt = hypothesis_engine.generate_prediction_prompt(mutation_type, architecture)
        return prompt
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lab/mutations")
def lab_mutation_list():
    descriptions = {
        "increase_depth":    {"label": "Increase Depth",       "icon": "fa-layer-group",     "description": "Add extra blocks to deepen the network", "category": "depth"},
        "decrease_depth":    {"label": "Decrease Depth",       "icon": "fa-compress-alt",    "description": "Remove blocks to create a shallower network", "category": "depth"},
        "increase_width":    {"label": "Increase Width",       "icon": "fa-expand",          "description": "Scale up channel sizes (×1.5)", "category": "width"},
        "decrease_width":    {"label": "Decrease Width",       "icon": "fa-compress",        "description": "Scale down channel sizes (×0.5)", "category": "width"},
        "add_residual":      {"label": "Add Skip Connections", "icon": "fa-code-branch",     "description": "Inject residual skip connections between blocks", "category": "structure"},
        "remove_residual":   {"label": "Remove Skip Connections","icon": "fa-ban",            "description": "Remove all skip/residual edges", "category": "structure"},
        "add_attention":     {"label": "Add Attention Block",  "icon": "fa-brain",           "description": "Insert a Multi-Head Attention node", "category": "attention"},
        "change_patch_size": {"label": "Change Patch Size",    "icon": "fa-th",              "description": "Change ViT patch size (affects token count)", "category": "embedding"},
        "change_hidden_dim": {"label": "Change Hidden Dim",    "icon": "fa-sliders",         "description": "Adjust transformer hidden dimension (d_model)", "category": "embedding"},
    }
    return {"mutations": descriptions}

@router.get("/system-design/patterns")
def get_system_patterns(db: Session = Depends(get_db)):
    return []
