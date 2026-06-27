import pytest
from unittest.mock import patch, MagicMock
from core.agents.ingestion_agent import (
    run_ingestion, should_revise, classify_paper, commit_result, IngestionState
)

@patch("core.agents.ingestion_agent.extract_architecture")
@patch("core.agents.ingestion_agent.process_text")
@patch("core.agents.ingestion_agent.llm_complete")
def test_run_ingestion_returns_final_code(mock_llm, mock_pt, mock_ea):
    mock_pt.return_value = {"method": "some text"}
    mock_ea.return_value = {"layers": []}
    mock_llm.return_value = "resnet"
    
    # We also mock code generation and linting to ensure it completes fast
    with patch("core.paper_to_code_generator.generate_pytorch_code", return_value="import torch", create=True):
        result = run_ingestion(paper_id=1, paper_name="Test Paper", raw_text="raw text")
        
    assert "final_code" in result
    assert result["final_code"] == "import torch"
    assert result.get("error") is None

def test_should_revise():
    # should_revise() returns "commit" when lint_errors is empty
    state = IngestionState(lint_errors=[], revision_count=0, error=None)
    assert should_revise(state) == "commit"
    
    # should_revise() returns "revise" when lint_errors is non-empty and revision_count < 3
    state = IngestionState(lint_errors=["SyntaxError: invalid syntax"], revision_count=2, error=None)
    assert should_revise(state) == "revise"
    
    # should_revise() returns "commit" when revision_count >= 3
    state = IngestionState(lint_errors=["SyntaxError: invalid syntax"], revision_count=3, error=None)
    assert should_revise(state) == "commit"
    
    # should_revise() returns "commit" when state has "error"
    state = IngestionState(lint_errors=["SyntaxError: invalid syntax"], revision_count=1, error="Some error")
    assert should_revise(state) == "commit"

@patch("core.agents.ingestion_agent.llm_complete")
@patch("core.agents.ingestion_agent.infer_family_from_name")
def test_classify_paper_fallback(mock_infer, mock_llm):
    # classify_paper() calls infer_family_from_name first, only falls back to LLM when it returns None
    
    # Case 1: infer_family_from_name returns a family
    mock_infer.return_value = "unet"
    state = IngestionState(paper_name="Unet Paper", sections={})
    result = classify_paper(state)
    assert result["model_family"] == "unet"
    mock_llm.assert_not_called()
    
    # Case 2: infer_family_from_name returns None, falls back to LLM
    mock_infer.return_value = None
    mock_llm.return_value = "vit"
    state = IngestionState(paper_name="Unknown Paper", sections={"method": "Vision Transformer"})
    result = classify_paper(state)
    assert result["model_family"] == "vit"
    mock_llm.assert_called_once()

def test_commit_result():
    # commit_result() stores code in final_code key
    state = IngestionState(code="import torch\nprint('hello')")
    result = commit_result(state)
    assert result["final_code"] == "import torch\nprint('hello')"
