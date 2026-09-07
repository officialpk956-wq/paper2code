import json

from benchmarks.audit_params import audit_cache, audit_spec


def test_audit_flags_repeated_kernel_as_fabrication_and_missing_channels():
    spec = {
        "layers": [
            {"type": "conv2d", "params": {"kernel_size": 3}}
            for _ in range(22)
        ]
    }

    result = audit_spec(spec)

    assert result["total_layers"] == 22
    assert result["layers_with_params"] == 22
    assert result["fabrication_flags"] == [
        {
            "type": "conv2d",
            "key": "kernel_size",
            "value": "3",
            "count": 22,
            "same_type_layers": 22,
            "fabrication_flag": True,
        }
    ]
    assert len(result["incomplete_layers"]) == 22


def test_audit_cache_reads_cached_specs_without_an_llm(tmp_path, monkeypatch):
    (tmp_path / "paper.production.json").write_text(
        json.dumps({"spec": {"layers": [{"type": "linear", "params": {"channels": 1000}}]}}),
        encoding="utf-8",
    )
    (tmp_path / "paper.production.diag.json").write_text("{}", encoding="utf-8")

    results = audit_cache(tmp_path)

    assert len(results) == 1
    assert results[0]["paper_id"] == "paper.production"
    assert results[0]["incomplete_layers"] == []
