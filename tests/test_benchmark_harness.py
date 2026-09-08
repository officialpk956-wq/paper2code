import json
from pathlib import Path

import pytest

from benchmarks import harness
from benchmarks import diagnose


@pytest.fixture(autouse=True)
def _isolate_cache_dir(tmp_path, monkeypatch):
    """Keep synthetic fixtures out of the real benchmarks/.cache.

    Several tests run the benchmark without patching CACHE_DIR themselves and
    were writing synthetic.*.json into the live cache, where a later --check or
    offline run would happily read them as real extractions. Redirecting here
    covers every test in the module, including ones added later; tests that
    patch CACHE_DIR explicitly still override this.
    """
    monkeypatch.setattr(harness, "CACHE_DIR", tmp_path / "_cache")
    harness.CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _label(tmp_path: Path, expected=None) -> Path:
    path = tmp_path / "synthetic.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": harness.LABEL_SCHEMA_VERSION,
                "paper_id": "synthetic",
                "source": "arxiv:0000.00000",
                "family": "resnet",
                "expected": expected
                or {
                    "layer_types": ["conv2d", "relu"],
                    "key_hyperparams": {"channels": 64},
                    "min_layers": 2,
                    "has_residual": False,
                },
                "notes": "Synthetic label for harness behavior.",
            }
        ),
        encoding="utf-8",
    )
    return path


def _correct_extraction(_label):
    return {
        "family": "resnet",
        "spec": {
            "layers": [
                {"type": "conv2d", "params": {"channels": 64}},
                {"type": "relu", "params": {}},
            ]
        },
    }


def test_known_correct_extraction_scores_one_on_every_available_metric(tmp_path):
    result = harness.run_benchmark([_label(tmp_path)], extractor=_correct_extraction)["per_paper"][0]
    assert result["layer_type_recall"] == 1.0
    assert result["layer_type_precision"] == 1.0
    assert result["hyperparam_accuracy"] == 1.0
    assert result["family_correct"] is True
    assert result["fidelity_score"] is None


def test_benchmark_reports_rule_based_fallback_per_paper_and_in_aggregate(tmp_path):
    result = harness.run_benchmark(
        [_label(tmp_path)],
        extractor=lambda _: {**_correct_extraction({}), "extraction_method": "rule_based_fallback", "extraction_reason": "ValueError: quota"},
    )

    assert result["per_paper"][0]["extraction_method"] == "rule_based_fallback"
    assert result["per_paper"][0]["extraction_reason"] == "ValueError: quota"
    assert result["aggregate"]["rule_based_fallback_count"] == 1


def test_benchmark_reports_provider_fallback_per_paper_and_in_aggregate(tmp_path):
    result = harness.run_benchmark(
        [_label(tmp_path)],
        extractor=lambda _: {
            **_correct_extraction({}),
            "provider_models": ["groq/model", "gemini/model"],
            "provider_fallback": True,
        },
    )

    assert result["per_paper"][0]["provider_models"] == ["groq/model", "gemini/model"]
    assert result["per_paper"][0]["provider_fallback"] is True
    assert result["aggregate"]["provider_fallback_count"] == 1


def test_wrong_extraction_names_specific_layer_and_hyperparameter_misses(tmp_path):
    def wrong(_):
        return {"family": "resnet", "spec": {"layers": [{"type": "conv2d", "params": {"channels": 32}}]}}

    result = harness.run_benchmark([_label(tmp_path)], extractor=wrong)["per_paper"][0]
    assert result["layer_type_recall"] < 1.0
    assert result["hyperparam_accuracy"] < 1.0
    assert any("relu" in miss for miss in result["misses"])
    assert any("channels=64" in miss for miss in result["misses"])


def test_missing_optional_label_key_does_not_count_against_metrics(tmp_path):
    label = _label(tmp_path, {"layer_types": ["relu"]})
    result = harness.run_benchmark(
        [label], extractor=lambda _: {"family": "resnet", "spec": {"layers": [{"type": "relu", "params": {}}]}}
    )["per_paper"][0]
    assert result["hyperparam_accuracy"] is None
    assert result["layer_type_recall"] == 1.0


def test_malformed_label_names_the_file(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json"):
        harness.load_label(broken)


def test_unknown_schema_version_names_the_file(tmp_path):
    label = _label(tmp_path)
    payload = json.loads(label.read_text(encoding="utf-8"))
    payload["schema_version"] = 999
    label.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"synthetic\.json: unsupported schema_version"):
        harness.load_label(label)


def test_offline_mode_uses_cache_and_never_calls_llm(tmp_path, monkeypatch):
    label = _label(tmp_path)
    monkeypatch.setattr(harness, "CACHE_DIR", tmp_path / ".cache")
    harness.CACHE_DIR.mkdir()
    (harness.CACHE_DIR / "synthetic.production.json").write_text(
        json.dumps(_correct_extraction({})), encoding="utf-8"
    )
    monkeypatch.setattr("core.llm_client.llm_complete", lambda *args, **kwargs: pytest.fail("LLM called"))
    result = harness.run_benchmark([label])
    assert result["aggregate"]["hard_failures"] == 0
    assert result["per_paper"][0]["cached"] is True


# Records the kwargs the harness passes to pdfplumber's extract_text, so the
# x_tolerance=1 fix cannot be dropped without a test noticing.
seen_pdf_kwargs: dict = {}


def test_live_adapter_follows_arxiv_pdf_redirects(monkeypatch):
    class Response:
        content = b"pdf bytes"

        def raise_for_status(self):
            return None

    seen = {}
    seen_pdf_kwargs.clear()
    monkeypatch.setattr(harness.httpx, "get", lambda url, **kwargs: seen.update(url=url, **kwargs) or Response())

    class Pdf:
        pages = [type("Page", (), {"extract_text": lambda self, **kw: seen_pdf_kwargs.update(kw) or "A residual architecture."})()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Extractor:
        def __init__(self, chunk_retriever=None):
            self.chunk_retriever = chunk_retriever
            self.diagnostic = {
                "focused_text": "A residual architecture.",
                "raw_llm_response": None,
                "raw_llm_error": None,
                "parsed_spec_pre_normalization": {"name": "ResNet", "layers": []},
                "spec_post_normalization": {"name": "ResNet", "layers": []},
            }

        def extract_from_text(self, text, source_chunks=None):
            assert text == "A residual architecture."
            return {"name": "ResNet", "layers": []}

    monkeypatch.setitem(__import__("sys").modules, "pdfplumber", type("PdfPlumber", (), {"open": lambda stream: Pdf()}))
    monkeypatch.setattr(harness, "_DiagnosticExtractor", Extractor)
    result = harness._live_extractor({"source": "arxiv:1512.03385", "paper_id": "resnet50"})
    assert result["family"] == "resnet"
    assert seen["follow_redirects"] is True
    assert seen_pdf_kwargs["x_tolerance"] == 1


def test_production_live_adapter_invokes_the_chunk_retriever(monkeypatch):
    from core.rag.config_extractor import ConfigExtractor as RealConfigExtractor

    class DiagnosticExtractor(RealConfigExtractor):
        def __init__(self, chunk_retriever=None):
            super().__init__(use_llm=False, verify=False, max_context_chars=10, chunk_retriever=chunk_retriever)
            self.diagnostic = {}

        def extract_from_text(self, text, source_chunks=None):
            spec = super().extract_from_text(text, source_chunks=source_chunks)
            self.diagnostic = {
                "focused_text": text,
                "raw_llm_response": None,
                "raw_llm_error": None,
                "parsed_spec_pre_normalization": spec,
                "spec_post_normalization": spec,
            }
            return spec

    class Response:
        content = b"pdf bytes"

        def raise_for_status(self):
            return None

    class Pdf:
        pages = [type("Page", (), {"extract_text": lambda self, **kw: seen_pdf_kwargs.update(kw) or "architecture " * 20})()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    calls = []

    def retriever(query, texts, top_k):
        calls.append((query, texts, top_k))
        return texts[:top_k]

    monkeypatch.setattr(harness.httpx, "get", lambda *args, **kwargs: Response())
    monkeypatch.setitem(__import__("sys").modules, "pdfplumber", type("PdfPlumber", (), {"open": lambda stream: Pdf()}))
    monkeypatch.setattr("backend.services.vector_service.hybrid_rank_texts", retriever)
    monkeypatch.setattr(harness, "_DiagnosticExtractor", DiagnosticExtractor)

    harness._live_extractor({"source": "arxiv:1512.03385", "paper_id": "resnet50"}, retrieval="production")

    assert calls


def test_legacy_live_adapter_does_not_invoke_the_chunk_retriever(monkeypatch):
    from core.rag.config_extractor import ConfigExtractor as RealConfigExtractor

    class DiagnosticExtractor(RealConfigExtractor):
        def __init__(self, chunk_retriever=None):
            super().__init__(use_llm=False, verify=False, max_context_chars=10, chunk_retriever=chunk_retriever)
            self.diagnostic = {}

        def extract_from_text(self, text, source_chunks=None):
            spec = super().extract_from_text(text, source_chunks=source_chunks)
            self.diagnostic = {
                "focused_text": text,
                "raw_llm_response": None,
                "raw_llm_error": None,
                "parsed_spec_pre_normalization": spec,
                "spec_post_normalization": spec,
            }
            return spec

    class Response:
        content = b"pdf bytes"

        def raise_for_status(self):
            return None

    class Pdf:
        pages = [type("Page", (), {"extract_text": lambda self, **kw: seen_pdf_kwargs.update(kw) or "architecture " * 20})()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(harness.httpx, "get", lambda *args, **kwargs: Response())
    monkeypatch.setitem(__import__("sys").modules, "pdfplumber", type("PdfPlumber", (), {"open": lambda stream: Pdf()}))
    monkeypatch.setattr(
        "backend.services.vector_service.hybrid_rank_texts",
        lambda *args, **kwargs: pytest.fail("legacy mode invoked hybrid retrieval"),
    )
    monkeypatch.setattr(harness, "_DiagnosticExtractor", DiagnosticExtractor)

    harness._live_extractor({"source": "arxiv:1512.03385", "paper_id": "resnet50"}, retrieval="legacy")


def test_cache_entries_are_distinct_per_retrieval_mode(tmp_path, monkeypatch):
    label = _label(tmp_path)
    monkeypatch.setattr(harness, "CACHE_DIR", tmp_path / ".cache")

    # A live run now STAGES rather than publishing: the cache is only written
    # once the run is accepted, so a --strict rejection cannot overwrite good
    # extractions with degraded ones. Mode-distinctness still holds either way.
    harness.run_benchmark([label], extractor=_correct_extraction, retrieval="legacy")
    harness.promote_staged_cache("legacy")
    harness.run_benchmark([label], extractor=_correct_extraction, retrieval="production")
    harness.promote_staged_cache("production")

    assert (harness.CACHE_DIR / "synthetic.legacy.json").exists()
    assert (harness.CACHE_DIR / "synthetic.production.json").exists()


def test_live_adapter_writes_a_companion_diagnostic_without_changing_cache(tmp_path, monkeypatch):
    class Response:
        content = b"pdf bytes"

        def raise_for_status(self):
            return None

    class Pdf:
        pages = [type("Page", (), {"extract_text": lambda self, **kw: seen_pdf_kwargs.update(kw) or "A residual architecture."})()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class Extractor:
        def __init__(self, chunk_retriever=None):
            self.diagnostic = {
                "focused_text": "A residual architecture.",
                "raw_llm_response": '{"name": "ResNet", "layers": []}',
                "raw_llm_error": None,
                "parsed_spec_pre_normalization": {"name": "ResNet", "layers": []},
                "spec_post_normalization": {"name": "ResNet", "layers": []},
            }

        def extract_from_text(self, text, source_chunks=None):
            return self.diagnostic["spec_post_normalization"]

    monkeypatch.setattr(harness, "CACHE_DIR", tmp_path / ".cache")
    monkeypatch.setattr(harness.httpx, "get", lambda *args, **kwargs: Response())
    monkeypatch.setitem(__import__("sys").modules, "pdfplumber", type("PdfPlumber", (), {"open": lambda stream: Pdf()}))
    monkeypatch.setattr(harness, "_DiagnosticExtractor", Extractor)

    harness._live_extractor({"source": "arxiv:1512.03385", "paper_id": "resnet50"})

    path = harness.diagnostic_path("resnet50", "production")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["raw_text_length"] == len("A residual architecture.")
    assert payload["chunk_count_by_type"] == {"text": 1}
    assert payload["raw_llm_response"]
    assert not (tmp_path / ".cache" / "resnet50.production.json").exists()


def test_diagnose_reads_a_live_companion_offline_without_calling_an_llm(tmp_path, monkeypatch):
    path = tmp_path / "ddpm.production.diag.json"
    path.write_text(
        json.dumps(
            {
                "raw_text_length": 42,
                "chunk_count_by_type": {"text": 1},
                "focused_text": "A focused passage.",
                "raw_llm_response": '{"layers": []}',
                "parsed_spec_pre_normalization": {"layers": []},
                "spec_post_normalization": {"layers": []},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(diagnose, "diagnostic_path", lambda *_args: path)
    monkeypatch.setattr(
        "core.llm_client.llm_complete",
        lambda *_args, **_kwargs: pytest.fail("offline diagnostics must not call an LLM"),
    )

    assert diagnose.main(["ddpm", "--retrieval", "production"]) == 0


@pytest.mark.parametrize("label_path", sorted(harness.LABELS_DIR.glob("*.json")))
def test_bundled_labels_validate(label_path):
    assert harness.load_label(label_path)["paper_id"]


# ── Phase 5 closeout: --check regression guard ────────────────────────────────

def _fake_results(recall=0.70, precision=0.70, hyperparam=0.50, family=0.80,
                  retrieval="production", fallbacks=0):
    return {
        "retrieval": retrieval,
        "per_paper": [],
        "aggregate": {
            "layer_type_recall": recall,
            "layer_type_precision": precision,
            "hyperparam_accuracy": hyperparam,
            "family_correct": family,
            "fidelity_score": None,
            "rule_based_fallback_count": fallbacks,
            "hard_failures": 0,
            "papers": 10,
        },
    }


def test_build_baseline_records_mode_and_schema():
    from benchmarks.harness import build_baseline, LABEL_SCHEMA_VERSION

    baseline = build_baseline(_fake_results(), "20260904T000000Z.json")
    assert baseline["schema_version"] == LABEL_SCHEMA_VERSION
    assert baseline["retrieval"] == "production"
    assert baseline["source_results"] == "20260904T000000Z.json"
    assert baseline["metrics"]["layer_type_recall"] == 0.70


def test_build_baseline_refuses_a_run_containing_fallbacks():
    """A run blending LLM and rule-based output must never become the reference."""
    from benchmarks.harness import build_baseline

    with pytest.raises(ValueError, match="rule-based fallback"):
        build_baseline(_fake_results(fallbacks=2), "x.json")


def test_check_passes_when_unchanged():
    from benchmarks.harness import build_baseline, check_against_baseline

    baseline = build_baseline(_fake_results(), "x.json")
    ok, problems = check_against_baseline(_fake_results(), baseline)
    assert ok and problems == []


def test_check_fails_and_names_the_dropped_metric():
    from benchmarks.harness import build_baseline, check_against_baseline

    baseline = build_baseline(_fake_results(recall=0.70), "x.json")
    ok, problems = check_against_baseline(_fake_results(recall=0.50), baseline)
    assert not ok
    assert any("layer_type_recall" in p and "0.700" in p and "0.500" in p for p in problems)


def test_check_tolerates_a_drop_within_tolerance():
    from benchmarks.harness import build_baseline, check_against_baseline

    baseline = build_baseline(_fake_results(recall=0.70), "x.json")
    ok, _ = check_against_baseline(_fake_results(recall=0.66), baseline)
    assert ok


def test_check_never_fails_on_improvement():
    from benchmarks.harness import build_baseline, check_against_baseline

    baseline = build_baseline(_fake_results(recall=0.70), "x.json")
    ok, problems = check_against_baseline(_fake_results(recall=0.95), baseline)
    assert ok and problems == []


def test_check_treats_a_missing_metric_as_an_error_not_a_pass():
    """The failure mode that lets a metric quietly disappear from the report."""
    from benchmarks.harness import build_baseline, check_against_baseline

    baseline = build_baseline(_fake_results(), "x.json")
    degraded = _fake_results()
    degraded["aggregate"]["hyperparam_accuracy"] = None
    ok, problems = check_against_baseline(degraded, baseline)
    assert not ok
    assert any("hyperparam_accuracy" in p and "missing" in p for p in problems)


def test_check_errors_on_retrieval_mode_mismatch():
    from benchmarks.harness import build_baseline, check_against_baseline

    baseline = build_baseline(_fake_results(retrieval="production"), "x.json")
    ok, problems = check_against_baseline(_fake_results(retrieval="legacy"), baseline)
    assert not ok
    assert any("retrieval mode mismatch" in p for p in problems)


def test_check_errors_on_schema_version_mismatch():
    from benchmarks.harness import build_baseline, check_against_baseline

    baseline = build_baseline(_fake_results(), "x.json")
    baseline["schema_version"] = 999
    ok, problems = check_against_baseline(_fake_results(), baseline)
    assert not ok
    assert any("schema_version mismatch" in p and "999" in p for p in problems)


def test_load_baseline_rejects_a_malformed_file(tmp_path):
    from benchmarks.harness import load_baseline

    bad = tmp_path / "baseline.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid baseline"):
        load_baseline(bad)


# ── Cache integrity: a rejected run must not poison the scoring cache ─────────
# Observed 2026-09-07: --strict refused to write the results file for a run
# containing rule-based fallbacks, but the cache had already been written
# inline, so bert_base/dcgan/ddpm were silently replaced with degraded specs
# and every later offline --check scored those instead.

def test_live_extraction_stages_and_does_not_touch_the_cache(tmp_path, monkeypatch):
    from benchmarks import harness

    monkeypatch.setattr(harness, "CACHE_DIR", tmp_path)
    good = {"spec": {"name": "good", "layers": []}, "family": "resnet"}
    harness._cache_path("p1", "production").write_text(
        json.dumps(good), encoding="utf-8"
    )

    label = {"paper_id": "p1", "source": "arxiv:1", "expected": {}}
    harness._load_or_extract(
        label, lambda _l: {"spec": {"name": "fresh", "layers": []}, "family": "vit"}, "production"
    )

    # cache untouched, staged file created
    assert json.loads(harness._cache_path("p1", "production").read_text())["spec"]["name"] == "good"
    assert harness._staged_path("p1", "production").exists()


def test_promote_publishes_staged_and_clears_it(tmp_path, monkeypatch):
    from benchmarks import harness

    monkeypatch.setattr(harness, "CACHE_DIR", tmp_path)
    harness._staged_path("p1", "production").write_text(
        json.dumps({"spec": {"name": "fresh", "layers": []}}), encoding="utf-8"
    )

    assert harness.promote_staged_cache("production") == ["p1"]
    assert json.loads(harness._cache_path("p1", "production").read_text())["spec"]["name"] == "fresh"
    assert not harness._staged_path("p1", "production").exists()


def test_discard_leaves_the_previous_cache_intact(tmp_path, monkeypatch):
    """The whole point: a --strict rejection must not cost us good data."""
    from benchmarks import harness

    monkeypatch.setattr(harness, "CACHE_DIR", tmp_path)
    harness._cache_path("p1", "production").write_text(
        json.dumps({"spec": {"name": "good", "layers": []}}), encoding="utf-8"
    )
    harness._staged_path("p1", "production").write_text(
        json.dumps({"spec": {"name": "degraded", "layers": []}}), encoding="utf-8"
    )

    assert harness.discard_staged_cache("production") == ["p1"]
    assert json.loads(harness._cache_path("p1", "production").read_text())["spec"]["name"] == "good"
    assert not harness._staged_path("p1", "production").exists()


def test_staging_is_keyed_per_retrieval_mode(tmp_path, monkeypatch):
    from benchmarks import harness

    monkeypatch.setattr(harness, "CACHE_DIR", tmp_path)
    harness._staged_path("p1", "production").write_text("{}", encoding="utf-8")
    harness._staged_path("p1", "legacy").write_text("{}", encoding="utf-8")

    assert harness.promote_staged_cache("production") == ["p1"]
    assert harness._staged_path("p1", "legacy").exists(), "legacy staging was clobbered"


# ── Provider purity ──────────────────────────────────────────────────────────
# Enforced 2026-09-07: the cache had accumulated 8 of 10 papers extracted by
# the Gemini fallback while baseline.json was Groq-only, so every offline
# --check compared one model's output against another's. Flagging did not stop
# it; rejection does.

def _run(rule_based=0, provider=0, n=3):
    per = []
    for i in range(n):
        per.append({
            "paper_id": f"p{i}",
            "extraction_method": "rule_based_fallback" if i < rule_based else "llm_verified",
            "provider_fallback": i < provider,
            "layer_type_recall": 0.7, "layer_type_precision": 0.7,
            "hyperparam_accuracy": 0.5, "family_correct": True, "fidelity_score": None,
        })
    return {"retrieval": "production", "per_paper": per, "aggregate": {
        "layer_type_recall": 0.7, "layer_type_precision": 0.7,
        "hyperparam_accuracy": 0.5, "family_correct": 1.0, "fidelity_score": None,
        "rule_based_fallback_count": rule_based, "provider_fallback_count": provider,
        "hard_failures": 0, "papers": n}}


def test_strict_rejects_cross_provider_fallbacks():
    from benchmarks.harness import strict_rejections
    reasons = strict_rejections(_run(provider=2))
    assert "cross-provider fallbacks" in reasons
    assert reasons["cross-provider fallbacks"] == ["p0", "p1"]


def test_strict_reports_both_reasons_independently():
    from benchmarks.harness import strict_rejections
    reasons = strict_rejections(_run(rule_based=1, provider=2))
    assert set(reasons) == {"rule-based fallbacks", "cross-provider fallbacks"}


def test_strict_accepts_a_pure_run():
    from benchmarks.harness import strict_rejections
    assert strict_rejections(_run()) == {}


def test_baseline_refuses_provider_mixed_runs():
    from benchmarks.harness import build_baseline
    with pytest.raises(ValueError, match="cross-provider"):
        build_baseline(_run(provider=1), "x.json")


def test_baseline_records_the_primary_model():
    from benchmarks.harness import build_baseline
    baseline = build_baseline(_run(), "x.json")
    assert baseline["primary_model"], "baseline must record which model produced it"


def _baseline_run(model="openai/gpt-oss-120b"):
    """Minimal (results, baseline) pair that passes every non-model guard."""
    metrics = {name: 0.5 for name in harness._CHECKED_METRICS}
    results = {"retrieval": "production", "aggregate": {"papers": 10, **metrics}}
    baseline = {
        "schema_version": harness.LABEL_SCHEMA_VERSION,
        "retrieval": "production",
        "primary_model": model,
        "papers": 10,
        "metrics": metrics,
    }
    return results, baseline


def test_check_rejects_a_baseline_built_on_a_different_model(monkeypatch):
    monkeypatch.setattr(harness, "_bare_primary_model", lambda: "openai/gpt-oss-120b")
    results, baseline = _baseline_run(model="anthropic/claude-3")
    ok, problems = harness.check_against_baseline(results, baseline)
    assert not ok
    assert any("primary model mismatch" in p for p in problems)


def test_check_rejects_a_baseline_with_no_recorded_model(monkeypatch):
    monkeypatch.setattr(harness, "_bare_primary_model", lambda: "openai/gpt-oss-120b")
    results, baseline = _baseline_run()
    del baseline["primary_model"]
    ok, problems = harness.check_against_baseline(results, baseline)
    assert not ok
    assert any("--write-baseline" in p for p in problems)


def test_check_passes_when_the_model_matches(monkeypatch):
    monkeypatch.setattr(harness, "_bare_primary_model", lambda: "openai/gpt-oss-120b")
    results, baseline = _baseline_run()
    ok, problems = harness.check_against_baseline(results, baseline)
    assert ok, problems
