"""Regression: an empty LLM completion must retry, then raise -- never return ''."""
import types, pytest
from unittest.mock import patch
import core.llm_client as lc

def _resp(content):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])

def test_empty_completion_retries_then_raises():
    lc._circuit_open = False; lc._failure_count = 0
    with patch("litellm.completion", return_value=_resp("")) as m, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="empty completion"):
            lc.llm_complete("hi")
    assert m.call_count == 3, f"expected 3 attempts, got {m.call_count}"

def test_empty_then_success_recovers():
    lc._circuit_open = False; lc._failure_count = 0
    with patch("litellm.completion", side_effect=[_resp(""), _resp('{"ok": 1}')]), patch("time.sleep"):
        assert lc.llm_complete("hi") == '{"ok": 1}'

def test_normal_completion_unaffected():
    lc._circuit_open = False; lc._failure_count = 0
    with patch("litellm.completion", return_value=_resp("hello")):
        assert lc.llm_complete("hi") == "hello"
