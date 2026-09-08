"""Offline-first benchmark harness for labeled paper extraction results.

The harness deliberately treats labels as ground truth and never calls an LLM
unless a caller explicitly supplies an extractor (the command-line ``--live``
path does this). Cached results make ordinary regression runs deterministic and
cost-free.
"""

import argparse
import io
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

from core.classification import classify_architecture, infer_family_from_name
from core.fidelity import score_fidelity
from core.orchestrator.pipeline import Paper2CodePipeline
from core.rag import config_extractor as config_extractor_module
from core.rag.config_extractor import ConfigExtractor
from core.rag.normalizer import CANONICAL_TYPES
from core.utils import chunk_pages_with_provenance, extract_caption_chunks, extract_table_chunks


BENCHMARKS_DIR = Path(__file__).resolve().parent
LABELS_DIR = BENCHMARKS_DIR / "labels"
CACHE_DIR = BENCHMARKS_DIR / ".cache"
RESULTS_DIR = BENCHMARKS_DIR / "results"
BASELINE_PATH = BENCHMARKS_DIR / "baseline.json"

# Metrics guarded by --check. fidelity_score is excluded on purpose: it is
# null on the live path (no code is generated there), so guarding it would
# compare None against None and report a false pass.
_CHECKED_METRICS = (
    "layer_type_recall",
    "layer_type_precision",
    "hyperparam_accuracy",
    "family_correct",
)
CHECK_TOLERANCE = 0.05

LABEL_SCHEMA_VERSION = 1
_REQUIRED_LABEL_KEYS = {"schema_version", "paper_id", "source", "family", "expected", "notes"}
_EXPECTED_KEYS = {"layer_types", "key_hyperparams", "min_layers", "has_residual"}


def load_label(path: str | Path) -> dict[str, Any]:
    """Load and validate one conservative benchmark label."""
    label_path = Path(path)
    try:
        payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid benchmark label {label_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid benchmark label {label_path}: expected a JSON object")
    missing = _REQUIRED_LABEL_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Invalid benchmark label {label_path}: missing {', '.join(sorted(missing))}")
    if payload["schema_version"] != LABEL_SCHEMA_VERSION:
        raise ValueError(
            f"Invalid benchmark label {label_path}: unsupported schema_version "
            f"{payload['schema_version']!r}; expected {LABEL_SCHEMA_VERSION}"
        )
    if not isinstance(payload["paper_id"], str) or not payload["paper_id"].strip():
        raise ValueError(f"Invalid benchmark label {label_path}: paper_id must be a non-empty string")
    if not isinstance(payload["expected"], dict):
        raise ValueError(f"Invalid benchmark label {label_path}: expected must be an object")
    unsupported = set(payload["expected"]) - _EXPECTED_KEYS
    if unsupported:
        raise ValueError(
            f"Invalid benchmark label {label_path}: unsupported expected keys "
            f"{', '.join(sorted(unsupported))}"
        )
    for key in ("layer_types", "key_hyperparams", "min_layers", "has_residual"):
        if key in payload["expected"]:
            value = payload["expected"][key]
            if key == "layer_types" and not isinstance(value, list):
                raise ValueError(f"Invalid benchmark label {label_path}: expected.layer_types must be a list")
            if key == "key_hyperparams" and not isinstance(value, dict):
                raise ValueError(f"Invalid benchmark label {label_path}: expected.key_hyperparams must be an object")
    layer_types = payload["expected"].get("layer_types", [])
    if any(not isinstance(layer_type, str) or not layer_type.strip() for layer_type in layer_types):
        raise ValueError(f"Invalid benchmark label {label_path}: layer types must be non-empty strings")
    if len(layer_types) != len(set(layer_types)):
        raise ValueError(f"Invalid benchmark label {label_path}: duplicate expected.layer_types")
    unsupported_types = set(layer_types) - CANONICAL_TYPES
    if unsupported_types:
        raise ValueError(
            f"Invalid benchmark label {label_path}: unsupported layer types "
            f"{', '.join(sorted(unsupported_types))}"
        )
    return payload


def _cache_path(paper_id: str, retrieval: str) -> Path:
    return CACHE_DIR / f"{paper_id}.{retrieval}.json"


def _bare_primary_model() -> str:
    """Model id the baseline was built against, for cross-provider detection."""
    from core.llm_client import PRIMARY_MODEL
    from core.rag.config_extractor import _bare_model_id

    return _bare_model_id(PRIMARY_MODEL)


def strict_rejections(results: dict) -> dict[str, list[str]]:
    """Reasons a live run is not a valid measurement, keyed by reason.

    Two disqualifiers, both of which silently corrupt comparability:

    - **rule-based fallbacks**: the spec came from the keyword extractor, not
      the LLM, so the run blends two different systems.
    - **provider fallbacks**: the paper was served by the cross-provider
      fallback model rather than the primary. Enforced from 2026-09-07 after
      the cache was found to hold 8 of 10 papers extracted by Gemini while
      `baseline.json` was Groq-only -- every offline --check had been
      comparing one model's output against another's. Flagging alone did not
      stop the cache accumulating mixed entries; only rejection does.
    """
    reasons: dict[str, list[str]] = {}
    rule_based = [
        item["paper_id"] for item in results["per_paper"]
        if item.get("extraction_method") == "rule_based_fallback"
    ]
    if rule_based:
        reasons["rule-based fallbacks"] = rule_based
    provider = [
        item["paper_id"] for item in results["per_paper"] if item.get("provider_fallback")
    ]
    if provider:
        reasons["cross-provider fallbacks"] = provider
    return reasons


def _staged_path(paper_id: str, retrieval: str) -> Path:
    """Where a live extraction lands before the run is accepted."""
    return CACHE_DIR / f"{paper_id}.{retrieval}.staged.json"


def promote_staged_cache(retrieval: str) -> list[str]:
    """Publish staged extractions into the scoring cache. Returns paper ids."""
    promoted: list[str] = []
    for staged in sorted(CACHE_DIR.glob(f"*.{retrieval}.staged.json")):
        paper_id = staged.name[: -len(f".{retrieval}.staged.json")]
        _cache_path(paper_id, retrieval).write_text(
            staged.read_text(encoding="utf-8"), encoding="utf-8"
        )
        staged.unlink()
        promoted.append(paper_id)
    return promoted


def discard_staged_cache(retrieval: str) -> list[str]:
    """Drop staged extractions from a rejected run, leaving the cache untouched."""
    discarded: list[str] = []
    for staged in sorted(CACHE_DIR.glob(f"*.{retrieval}.staged.json")):
        discarded.append(staged.name[: -len(f".{retrieval}.staged.json")])
        staged.unlink()
    return discarded


def diagnostic_path(paper_id: str, retrieval: str) -> Path:
    """Companion diagnostic artifact; never part of the scoring cache."""
    return CACHE_DIR / f"{paper_id}.{retrieval}.diag.json"


class _DiagnosticExtractor(ConfigExtractor):
    """Capture extraction stages for benchmark diagnostics without changing output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic: dict[str, Any] = {
            "focused_text": None,
            "raw_llm_response": None,
            "raw_llm_error": None,
            "parsed_spec_pre_normalization": None,
            "spec_post_normalization": None,
        }

    def _focus_text(self, text, source_chunks=None):
        focused = super()._focus_text(text, source_chunks=source_chunks)
        self.diagnostic["focused_text"] = focused
        return focused

    def _extract_with_llm(self, text):
        original_complete = config_extractor_module.llm_complete

        def capture_response(prompt):
            response = original_complete(prompt)
            self.diagnostic["raw_llm_response"] = response
            return response

        config_extractor_module.llm_complete = capture_response
        try:
            raw = super()._extract_with_llm(text)
            self.diagnostic["parsed_spec_pre_normalization"] = raw
            return raw
        except Exception as exc:
            self.diagnostic["raw_llm_error"] = str(exc)
            raise
        finally:
            config_extractor_module.llm_complete = original_complete

    def _verify_extraction(self, original_text, extracted):
        verified = super()._verify_extraction(original_text, extracted)
        self.diagnostic["parsed_spec_pre_normalization"] = verified
        return verified

    def _extract_rule_based(self, text):
        raw = super()._extract_rule_based(text)
        self.diagnostic["parsed_spec_pre_normalization"] = raw
        return raw

    def extract_from_text(self, text, source_chunks=None):
        spec = super().extract_from_text(text, source_chunks=source_chunks)
        self.diagnostic["spec_post_normalization"] = spec
        return spec


def _write_diagnostic(paper_id: str, retrieval: str, payload: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = diagnostic_path(paper_id, retrieval)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _normalise_extraction(extraction: dict[str, Any]) -> dict[str, Any]:
    """Keep the cache portable; graphs and other runtime objects stay out of it."""
    if not isinstance(extraction, dict):
        raise ValueError("extractor returned a non-object result")
    spec = extraction.get("spec", extraction)
    if not isinstance(spec, dict):
        raise ValueError("extractor result has no object spec")
    return {
        "spec": spec,
        "family": extraction.get("family") or spec.get("family") or spec.get("model_family"),
        "code": extraction.get("code") or extraction.get("generated_code_source"),
        "extraction_method": extraction.get("extraction_method") or spec.get("extraction_method"),
        "extraction_reason": extraction.get("extraction_reason") or spec.get("extraction_reason"),
        "provider_models": extraction.get("provider_models") or spec.get("provider_models") or [],
        "provider_fallback": bool(extraction.get("provider_fallback") or spec.get("provider_fallback")),
    }


def _load_or_extract(
    label: dict[str, Any], extractor: Callable[[dict[str, Any]], dict] | None, retrieval: str
) -> tuple[dict[str, Any], bool]:
    cache_path = _cache_path(label["paper_id"], retrieval)
    if extractor is None:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached extraction for {label['paper_id']} at {cache_path}. "
                "Run `python -m benchmarks.harness --live` to create it."
            )
        return _normalise_extraction(json.loads(cache_path.read_text(encoding="utf-8"))), True

    result = _normalise_extraction(extractor(label))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Stage, do not publish. --strict refuses to write the *results* file for a
    # run containing rule-based fallbacks, but the cache used to be written
    # inline and so was overwritten anyway: a rejected run silently replaced
    # good cached extractions with degraded ones, and every later offline
    # --check then scored those. Observed 2026-09-07 -- bert_base, dcgan and
    # ddpm were rejected by --strict yet their cache entries had already been
    # replaced by rule_based_fallback specs. Staged entries are promoted only
    # once the run is accepted; a rejected run leaves the prior cache intact.
    _staged_path(label["paper_id"], retrieval).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result, False


def _layer_types(spec: dict[str, Any]) -> set[str]:
    layers = spec.get("layers", [])
    return {
        layer["type"].lower()
        for layer in layers
        if isinstance(layer, dict) and isinstance(layer.get("type"), str)
    }


def _hyperparameters(value: Any, wanted: set[str], found: dict[str, list[Any]] | None = None) -> dict[str, list[Any]]:
    found = found if found is not None else {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in wanted and isinstance(child, (str, int, float, bool)):
                found.setdefault(key, []).append(child)
            _hyperparameters(child, wanted, found)
    elif isinstance(value, list):
        for child in value:
            _hyperparameters(child, wanted, found)
    return found


def _has_residual(spec: dict[str, Any]) -> bool:
    if {"residualblock", "residual_add"} & _layer_types(spec):
        return True
    connection_types = spec.get("connection_types", [])
    if isinstance(connection_types, str):
        connection_types = [connection_types]
    return any("residual" in str(value).lower() or "skip" in str(value).lower() for value in connection_types)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _score_label(label: dict[str, Any], extraction: dict[str, Any], fidelity: bool) -> dict[str, Any]:
    expected = label["expected"]
    spec = extraction["spec"]
    actual_types = _layer_types(spec)
    expected_types = {str(layer_type).lower() for layer_type in expected.get("layer_types", [])}
    matched = expected_types & actual_types
    missing_types = sorted(expected_types - actual_types)
    unexpected_types = sorted(actual_types - expected_types)
    recall = len(matched) / len(expected_types) if expected_types else None
    precision = len(matched) / len(actual_types) if actual_types else (1.0 if not expected_types else 0.0)

    expected_hyperparameters = expected.get("key_hyperparams", {})
    actual_hyperparameters = _hyperparameters(spec, set(expected_hyperparameters))
    matched_hyperparameters = [
        key
        for key, value in expected_hyperparameters.items()
        if value in actual_hyperparameters.get(key, [])
    ]
    missing_hyperparameters = [
        f"{key}={value!r}"
        for key, value in expected_hyperparameters.items()
        if key not in matched_hyperparameters
    ]
    hyperparam_accuracy = (
        len(matched_hyperparameters) / len(expected_hyperparameters)
        if expected_hyperparameters
        else None
    )

    expected_family = label["family"].lower()
    actual_family = str(extraction.get("family") or "unknown").lower()
    family_correct = actual_family == expected_family
    misses = []
    if missing_types:
        misses.append(f"missing layer types: {', '.join(missing_types)}")
    if unexpected_types:
        misses.append(f"unexpected layer types: {', '.join(unexpected_types)}")
    if missing_hyperparameters:
        misses.append(f"hyperparameter mismatches: {', '.join(missing_hyperparameters)}")
    if not family_correct:
        misses.append(f"family: expected {expected_family}, got {actual_family}")
    if "has_residual" in expected and _has_residual(spec) != expected["has_residual"]:
        misses.append(
            f"residual: expected {expected['has_residual']}, got {_has_residual(spec)}"
        )
    if "min_layers" in expected and len(spec.get("layers", [])) < expected["min_layers"]:
        misses.append(f"layer count: expected at least {expected['min_layers']}, got {len(spec.get('layers', []))}")

    fidelity_score = None
    if fidelity and isinstance(extraction.get("code"), str) and extraction["code"].strip():
        fidelity_score = score_fidelity(spec, extraction["code"])["score"]

    return {
        "paper_id": label["paper_id"],
        "source": label["source"],
        "expected_family": expected_family,
        "actual_family": actual_family,
        "layer_type_recall": recall,
        "layer_type_precision": precision,
        "hyperparam_accuracy": hyperparam_accuracy,
        "family_correct": family_correct,
        "fidelity_score": fidelity_score,
        "extraction_method": extraction.get("extraction_method"),
        "extraction_reason": extraction.get("extraction_reason"),
        "provider_models": extraction.get("provider_models", []),
        "provider_fallback": extraction.get("provider_fallback", False),
        "misses": misses,
    }


def run_benchmark(label_paths: Iterable[str | Path], extractor=None, fidelity: bool = True, retrieval: str = "production", pace_seconds: float = 0.0) -> dict:
    """Run extraction over labeled papers and score it against the labels.

    With no extractor, reads only cached JSON outputs and makes no model calls.
    Supplying an extractor is the explicit live path; its normalized result is
    cached by paper id for later offline regression runs.
    """
    per_paper: list[dict[str, Any]] = []
    hard_failures = 0
    for index, label_path in enumerate(label_paths):
        if extractor is not None and index and pace_seconds > 0:
            time.sleep(pace_seconds)
        label = load_label(label_path)
        try:
            extraction, cached = _load_or_extract(label, extractor, retrieval)
            result = _score_label(label, extraction, fidelity)
            result["cached"] = cached
        except Exception as exc:
            hard_failures += 1
            result = {
                "paper_id": label["paper_id"],
                "source": label["source"],
                "layer_type_recall": None,
                "layer_type_precision": None,
                "hyperparam_accuracy": None,
                # None, not False: a hard failure means the family was never
                # measured. Recording it as False would average into
                # family_correct as a zero and understate real accuracy.
                "family_correct": None,
                "fidelity_score": None,
                "extraction_method": None,
                "extraction_reason": None,
                "provider_models": [],
                "provider_fallback": False,
                "misses": [f"hard failure: {exc}"],
                "cached": False,
            }
        per_paper.append(result)

    aggregate = {
        "layer_type_recall": _mean([item["layer_type_recall"] for item in per_paper if item["layer_type_recall"] is not None]),
        "layer_type_precision": _mean([item["layer_type_precision"] for item in per_paper if item["layer_type_precision"] is not None]),
        "hyperparam_accuracy": _mean([item["hyperparam_accuracy"] for item in per_paper if item["hyperparam_accuracy"] is not None]),
        "family_correct": _mean(
            [float(item["family_correct"]) for item in per_paper if item["family_correct"] is not None]
        ),
        "fidelity_score": _mean([item["fidelity_score"] for item in per_paper if item["fidelity_score"] is not None]),
        "rule_based_fallback_count": sum(
            item.get("extraction_method") == "rule_based_fallback" for item in per_paper
        ),
        "provider_fallback_count": sum(bool(item.get("provider_fallback")) for item in per_paper),
        "hard_failures": hard_failures,
        "papers": len(per_paper),
    }
    return {"retrieval": retrieval, "per_paper": per_paper, "aggregate": aggregate}


def print_table(results: dict) -> None:
    """Print a compact, dependency-free result table."""
    print(f"retrieval mode: {results['retrieval']}")
    print("paper_id                 recall  precision  hyperparams  family  fidelity")
    print("-" * 76)
    for item in results["per_paper"]:
        def value(key: str) -> str:
            number = item.get(key)
            return "-" if number is None else f"{number:.2f}"
        print(
            f"{item['paper_id']:<24} {value('layer_type_recall'):>6}  "
            f"{value('layer_type_precision'):>9}  {value('hyperparam_accuracy'):>11}  "
            f"{str(item.get('family_correct')):>6}  {value('fidelity_score'):>8}"
        )
    print("aggregate                " + json.dumps(results["aggregate"], sort_keys=True))


def write_results(results: dict, timestamp: str) -> Path:
    """Write one comparable result artifact using a caller-provided UTC timestamp."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{timestamp}.json"
    path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_baseline(results: dict, source_results: str) -> dict:
    """Freeze a run's metrics as the regression reference.

    Records retrieval mode and label schema_version alongside the numbers: a
    baseline compared against different labels or a different retrieval path
    is meaningless, and that confound has already produced false conclusions
    in this project once.
    """
    aggregate = results["aggregate"]
    if aggregate.get("rule_based_fallback_count"):
        raise ValueError(
            "refusing to build a baseline from a run containing "
            f"{aggregate['rule_based_fallback_count']} rule-based fallback(s): "
            "the aggregate blends LLM output with rate-limit-degraded output"
        )
    if aggregate.get("provider_fallback_count"):
        raise ValueError(
            "refusing to build a baseline from a run containing "
            f"{aggregate['provider_fallback_count']} cross-provider fallback(s): "
            "papers served by the fallback model are not comparable to "
            "primary-model output"
        )
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "retrieval": results["retrieval"],
        "primary_model": _bare_primary_model(),
        "papers": aggregate["papers"],
        "source_results": source_results,
        "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": {name: aggregate.get(name) for name in _CHECKED_METRICS},
    }


def load_baseline(path: str | Path = BASELINE_PATH) -> dict:
    baseline_path = Path(path)
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid baseline {baseline_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"Invalid baseline {baseline_path}: missing a metrics object")
    return payload


def check_against_baseline(
    results: dict, baseline: dict, tolerance: float = CHECK_TOLERANCE
) -> tuple[bool, list[str]]:
    """Compare a run to the baseline. Returns (ok, problems).

    A metric that improves never fails. A metric present in the baseline but
    absent (or None) in the run is an ERROR, not a silent pass -- that is the
    failure mode that lets a metric quietly disappear from the report.
    """
    problems: list[str] = []

    if baseline.get("schema_version") != LABEL_SCHEMA_VERSION:
        problems.append(
            f"label schema_version mismatch: baseline={baseline.get('schema_version')!r} "
            f"current={LABEL_SCHEMA_VERSION!r}"
        )
    if baseline.get("retrieval") != results.get("retrieval"):
        problems.append(
            f"retrieval mode mismatch: baseline={baseline.get('retrieval')!r} "
            f"run={results.get('retrieval')!r}"
        )
    # build_baseline records primary_model; without comparing it here a run from
    # one model silently validates against a baseline built on another, which is
    # the same class of unverifiable comparison the schema/retrieval guards exist
    # to prevent. A baseline predating the field cannot be checked at all, so it
    # reports as a problem rather than passing by default.
    baseline_model = baseline.get("primary_model")
    current_model = _bare_primary_model()
    if baseline_model is None:
        problems.append(
            "baseline records no primary_model, so provider purity cannot be "
            f"verified against the current model ({current_model!r}); "
            "rebuild it with --write-baseline"
        )
    elif baseline_model != current_model:
        problems.append(
            f"primary model mismatch: baseline={baseline_model!r} current={current_model!r}"
        )

    aggregate = results["aggregate"]
    for name, reference in baseline["metrics"].items():
        if reference is None:
            continue
        actual = aggregate.get(name)
        if actual is None:
            problems.append(f"{name}: present in baseline ({reference:.3f}) but missing from this run")
            continue
        drop = reference - actual
        if drop > tolerance:
            problems.append(f"{name}: {reference:.3f} -> {actual:.3f} (dropped {drop:.3f}, tolerance {tolerance})")

    return (not problems), problems


def _chunk_retriever(query: str, texts: list[str], top_k: int) -> list[str]:
    from backend.services.vector_service import hybrid_rank_texts

    return hybrid_rank_texts(query, texts, top_k=top_k)


def _live_extractor(label: dict[str, Any], retrieval: str = "production") -> dict:
    source = str(label["source"])
    if not source.startswith("arxiv:"):
        raise ValueError(f"live benchmark only supports arXiv labels, got {source}")
    paper_id = source.removeprefix("arxiv:")
    # arXiv redirects the conventional .pdf URL. Keep this compatibility
    # detail inside the benchmark-only live adapter; production behavior is
    # deliberately outside this prompt's scope.
    response = httpx.get(
        f"https://arxiv.org/pdf/{paper_id}.pdf", follow_redirects=True, timeout=60.0
    )
    response.raise_for_status()
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("live benchmark requires pdfplumber") from exc
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        page_texts = [
            (page_number, text)
            for page_number, page in enumerate(pdf.pages[:30], start=1)
            # x_tolerance=1: pdfplumber's default (3) merges adjacent words on these
            # PDFs -- measured across all 10 benchmark papers, spaces ran 3.3-11.1%
            # of characters against ~16% for normal prose, and transformer_base came
            # out at an average letter-run length of 12.4 chars ('Weuseself-
            # attentionat'). That breaks regex word boundaries, BM25 tokenisation
            # and embeddings alike. At x_tolerance=1 the same papers land at
            # 13.4-15.3% spaces and 4.7-5.4 char runs, which is normal English.
            if (text := page.extract_text(x_tolerance=1))
        ]
    text = "\n\n".join(page_text for _, page_text in page_texts)
    if not text.strip():
        raise ValueError("live benchmark could not extract a PDF text layer")

    source_chunks = chunk_pages_with_provenance(page_texts)
    source_chunks.extend(extract_table_chunks(page_texts))
    source_chunks.extend(extract_caption_chunks(page_texts))

    # This baseline measures extraction. It intentionally does not generate
    # code or contact E2B, which would measure a separate downstream stage.
    if retrieval == "production":
        extractor = _DiagnosticExtractor(chunk_retriever=_chunk_retriever)
        spec = extractor.extract_from_text(text, source_chunks=source_chunks)
    elif retrieval == "legacy":
        extractor = _DiagnosticExtractor()
        spec = extractor.extract_from_text(text)
    else:
        raise ValueError(f"Unknown retrieval mode {retrieval!r}")

    diagnostic = {
        "paper_id": label["paper_id"],
        "retrieval": retrieval,
        "raw_text_length": len(text),
        "chunk_count_by_type": dict(sorted(Counter(chunk.get("chunk_type", "text") for chunk in source_chunks).items())),
        **extractor.diagnostic,
    }
    _write_diagnostic(label["paper_id"], retrieval, diagnostic)
    family = infer_family_from_name(str(spec.get("name", "")))
    if family is None:
        try:
            graph = Paper2CodePipeline().run_single(spec)["graph"]
            family = classify_architecture(graph)
        except Exception:
            family = "unknown"
    return {
        "spec": spec,
        "family": family,
        "extraction_method": spec.get("extraction_method"),
        "extraction_reason": spec.get("extraction_reason"),
        "provider_models": spec.get("provider_models", []),
        "provider_fallback": spec.get("provider_fallback", False),
    }


def _live_extractor_with_retry(label: dict[str, Any], retrieval: str = "production", attempts: int = 3) -> dict:
    """Retry a whole paper only when the recorded result proves a rate-limit fallback."""
    result = _live_extractor(label, retrieval=retrieval)
    for attempt in range(1, attempts):
        reason = str(result.get("extraction_reason") or "").lower()
        if result.get("extraction_method") != "rule_based_fallback" or "rate" not in reason:
            break
        time.sleep(8 * attempt)
        result = _live_extractor(label, retrieval=retrieval)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Paper2Code extraction benchmarks")
    parser.add_argument("--live", action="store_true", help="re-extract papers and refresh the cache")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None, help="reject live runs containing rule-based fallbacks")
    parser.add_argument("--pace-seconds", type=float, default=10.0, help="delay between live papers")
    parser.add_argument(
        "--retrieval", choices=("production", "legacy"), default="production", help="extraction retrieval path to measure"
    )
    parser.add_argument("--check", action="store_true", help="offline: compare cached run to benchmarks/baseline.json and fail on regression")
    parser.add_argument("--write-baseline", action="store_true", help="freeze this run as benchmarks/baseline.json")
    parser.add_argument("--baseline", default=str(BASELINE_PATH), help="baseline file path")
    parser.add_argument("--tolerance", type=float, default=CHECK_TOLERANCE, help="allowed absolute metric drop")
    parser.add_argument("--timestamp", help="UTC timestamp used in the result filename")
    parser.add_argument("labels", nargs="*", help="label JSON files (default: all bundled labels)")
    args = parser.parse_args(argv)
    paths = [Path(path) for path in args.labels] or sorted(LABELS_DIR.glob("*.json"))
    timestamp = args.timestamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.check:
        # Offline by construction: extractor=None means cached specs only,
        # so --check never makes an LLM call.
        results = run_benchmark(paths, extractor=None, retrieval=args.retrieval, pace_seconds=0.0)
        print_table(results)
        try:
            baseline = load_baseline(args.baseline)
        except ValueError as exc:
            print(f"check failed: {exc}")
            return 1
        ok, problems = check_against_baseline(results, baseline, tolerance=args.tolerance)
        if ok:
            print(f"check passed against {args.baseline} (tolerance {args.tolerance})")
            return 0
        for problem in problems:
            print(f"check failed: {problem}")
        return 1

    extractor = (lambda label: _live_extractor_with_retry(label, retrieval=args.retrieval)) if args.live else None
    results = run_benchmark(paths, extractor=extractor, retrieval=args.retrieval, pace_seconds=args.pace_seconds if args.live else 0.0)
    print_table(results)
    strict = args.live if args.strict is None else args.strict
    rejections = strict_rejections(results)
    if strict and rejections:
        for reason, papers in rejections.items():
            print(f"strict live benchmark rejected {reason}: {', '.join(papers)}")
        discarded = discard_staged_cache(args.retrieval)
        if discarded:
            print(f"discarded staged cache for {len(discarded)} paper(s); prior cache left intact")
        return 1
    promoted = promote_staged_cache(args.retrieval)
    if promoted:
        print(f"promoted staged cache for {len(promoted)} paper(s)")
    results_path = write_results(results, timestamp)
    print(f"wrote {results_path}")
    if args.write_baseline:
        try:
            baseline = build_baseline(results, results_path.name)
        except ValueError as exc:
            print(f"baseline refused: {exc}")
            return 1
        Path(args.baseline).write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote baseline {args.baseline}")
    return 0 if not results["aggregate"]["hard_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
