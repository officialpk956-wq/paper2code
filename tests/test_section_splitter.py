"""
Regression tests for core.section_splitter.

Covers a bug found during Phase 1 live-upload verification: when the LLM
classifies a single text chunk into multiple sections at once (a valid,
common response shape), safe_parse_llm_output used to keep only the first
list entry and silently drop the rest (e.g. "method"/"experiments"), which
made real-paper extraction fail unconditionally.
"""

from unittest.mock import patch

from core.section_splitter import process_text, safe_parse_llm_output


def test_safe_parse_llm_output_keeps_all_sections_from_a_list_response():
    raw = (
        '[{"section": "abstract", "content": "abstract text"},'
        ' {"section": "method", "content": "method text"}]'
    )
    results = safe_parse_llm_output(raw, fallback_text="fallback")

    assert results == [("abstract", "abstract text"), ("method", "method text")]


def test_safe_parse_llm_output_handles_single_dict_response():
    raw = '{"section": "method", "content": "method text"}'
    assert safe_parse_llm_output(raw, fallback_text="fallback") == [("method", "method text")]


def test_safe_parse_llm_output_falls_back_on_malformed_json():
    assert safe_parse_llm_output("not json", fallback_text="fallback") == [
        ("other", "fallback")
    ]


def test_process_text_does_not_drop_method_section_from_multi_section_response():
    multi_section_response = (
        '[{"section": "abstract", "content": "We present ResNet."},'
        ' {"section": "method", "content": "Residual blocks with skip connections."}]'
    )
    with patch(
        "core.section_splitter.classify_section", return_value=multi_section_response
    ):
        sections = process_text("We present ResNet. Residual blocks with skip connections.")

    assert sections["method"] == "Residual blocks with skip connections."
    assert sections["abstract"] == "We present ResNet."
