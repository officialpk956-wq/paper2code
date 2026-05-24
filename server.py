from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import io

from src.orchestrator.pipeline import Paper2CodePipeline
from src.paper_to_code_generator import PaperToCodeGenerator
from src.rag.config_extractor import ConfigExtractor
from src.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
from src.explainers.graph_explainer import explain_node
from src.codegen import _node_to_layer

app = FastAPI(title="Paper2Code API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline = Paper2CodePipeline()
extractor = ConfigExtractor()
generator = PaperToCodeGenerator()

class TextRequest(BaseModel):
    text: str
    
class CompareRequest(BaseModel):
    text_a: str
    text_b: str

class GraphRequest(BaseModel):
    name: str = "Architect Session"
    layers: list[Dict[str, Any]]

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

def _build_response(spec, result, code, code_source):
    graph = result["graph"]
    metrics = estimate_metrics_from_graph(graph)
    mem_data = estimate_activation_memory(graph, batch_size=1, input_spatial=224)
    total_mem_mb = sum(row['mem_mb'] for row in mem_data) if mem_data else 0
    
    # Generate layer breakdown for explanation
    layer_breakdown = []
    for node in graph.nodes:
        node_code = _node_to_layer(node) or f"# Custom block implementation needed for {node.type}"
        layer_breakdown.append({
            "id": node.id,
            "label": node.label,
            "type": node.type,
            "params": node.params,
            "semantic": node.semantic_params,
            "description": node.description,
            "explanation": explain_node(node),
            "code_snippet": node_code
        })
        
    return {
        "name": graph.name,
        "svg": result["visual"]["graphviz_dot"],
        "explanation": result["explanation"],
        "metadata": result["metadata"],
        "code": code,
        "code_source": code_source,
        "layer_breakdown": layer_breakdown,
        "metrics": {
            "flops_score": metrics["total_flops_score"],
            "params": metrics["total_params_estimate"],
            "depth": metrics["depth"],
            "memory_mb": round(total_mem_mb, 1)
        },
        "tensor_trace": graph.metadata.get("tensor_trace", []),
        "cross_attention_events": graph.metadata.get("cross_attention_events", []),
        "flops_events": graph.metadata.get("flops_events", []),
        "kag_motifs": result.get("kag_motifs", []),
        "kag_anomalies": result.get("kag_anomalies", []),
        "kag_semantic_roles": result.get("kag_semantic_roles", {}),
    }

@app.post("/api/parse_pdf")
async def parse_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        paper_name = file.filename.replace(".pdf", "")
        # Use generator's built-in PDF extraction which handles pdfplumber safely
        result_dict = generator.from_pdf(pdf_bytes, paper_name)
        
        # result_dict already contains graph, explanation, code, code_source.
        # But we want to reuse _build_response to get metrics and layer_breakdown
        return _build_response(
            spec={"model_family": result_dict.get("family", "")}, 
            result=result_dict, 
            code=result_dict["code"], 
            code_source=result_dict["code_source"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/parse_text")
def parse_text(request: TextRequest):
    try:
        spec = extractor.extract_from_text(request.text)
        result = pipeline.run_single(spec)
        code, code_source = generator._generate_code(spec, result["graph"])
        return _build_response(spec, result, code, code_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare_text")
def compare_text(request: CompareRequest):
    try:
        result = pipeline.run_comparison_from_text(request.text_a, request.text_b)
        return {
            "name_a": result["graph_a"].name,
            "name_b": result["graph_b"].name,
            "svg_a": result["visual_a"]["graphviz_dot"],
            "svg_b": result["visual_b"]["graphviz_dot"],
            "explanation": result["explanation"],
            "metadata": result["metadata"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_graph")
def analyze_graph(request: GraphRequest):
    try:
        # Normalize the layers received from the frontend
        # The frontend sends: [{"type": "Conv2D", "params": {...}}, ...]
        # Pipeline expects normalized ConfigDict.
        
        # We'll use the normalizer directly
        from src.rag.normalizer import normalize_config
        config = normalize_config({
            "name": request.name,
            "layers": request.layers
        })
        
        # Run pipeline starting from config (bypassing extraction)
        result = pipeline.run_single(request.name, config)
        
        return _build_response(
            spec={"model_family": "Architect"}, 
            result=result, 
            code=result.get("code", ""), 
            code_source="Architect Session"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Added reload=True so you don't have to restart the server manually when we edit Python files!
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
