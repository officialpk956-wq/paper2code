"""Phase 1 contract and generated-artifact regression tests."""

from unittest.mock import patch

import pytest

from backend.models import Paper
from core.paper_to_code_generator import PaperToCodeGenerator


@pytest.mark.parametrize(
    ("family", "expected_output"),
    [
        ("resnet", [1, 1000]),
        ("unet", [1, 2, 256, 256]),
        ("vit", [1, 1000]),
    ],
)
def test_known_family_source_is_self_contained_and_executable(family, expected_output):
    generator = PaperToCodeGenerator()
    spec = {"model_family": family}

    source = generator._builder_code(family, spec)
    report = generator.validate_generated_code(source, "builder", spec)

    assert "from core." not in source
    assert "import torch" in source
    assert report["passed"] is True, report
    assert report["checks"] == {"syntax": True, "exec": True, "forward": True}
    assert report["output_shape"] == expected_output


def test_run_pipeline_accepts_a_spec_with_no_layers_key():
    """
    Regression test: _run_pipeline used to check spec.get("layers"), but
    extract_architecture's schema (core/schemas_base.py) has no "layers" key
    at all -- it uses model_family/stem/block/stages/head. That made every
    real extraction fail with "No architecture could be detected.", even
    when extraction correctly returned a populated model_family/stages.

    Also exercises _spec_to_config_dict, the translator from that schema to
    the ConfigDict shape ParsingAgentImpl/ConfigParsingAgent actually
    consume ({"name", "layers", "connections"}).
    """
    generator = PaperToCodeGenerator()
    spec_without_layers_key = {"model_family": "resnet", "stages": []}

    # Verify _spec_to_config_dict directly
    translated = generator._spec_to_config_dict(spec_without_layers_key, "test_paper")
    assert "layers" in translated
    assert len(translated["layers"]) >= 2
    assert "connections" in translated

    # Verify fallback path in _run_pipeline
    with (
        patch.object(generator.config_extractor, "extract_from_text", return_value=None),
        patch("core.paper_to_code_generator.process_text", return_value={"method": "..."}),
        patch(
            "core.paper_to_code_generator.extract_architecture",
            return_value=spec_without_layers_key,
        ),
    ):
        result = generator._run_pipeline("irrelevant raw text", "test_paper")

    assert result["family"] == "resnet"
    assert result["generation_status"] == "success"
    assert result["verification_report"]["passed"] is True


@pytest.mark.parametrize(
    ("family", "extracted_stages"),
    [
        # Real extraction sometimes returns partial per-stage dicts (missing
        # in_channels/expansion/stride/downsample/num_blocks for resnet, or
        # missing "repeats" for vit) -- _prepare_builder_schema used to keep
        # them as-is instead of filling in the builder's required keys.
        (
            "resnet",
            [
                {"name": None, "repeats": 1, "out_channels": 64},
                {"name": None, "repeats": 1, "out_channels": 128},
                {"name": None, "repeats": 1, "out_channels": 256},
                {"name": None, "repeats": 1, "out_channels": 512},
            ],
        ),
        ("vit", [{"name": None, "out_channels": 192}]),
    ],
)
def test_builder_schema_fills_in_missing_stage_fields(family, extracted_stages):
    generator = PaperToCodeGenerator()
    spec = {"model_family": family, "stages": extracted_stages}

    source = generator._builder_code(family, spec)
    report = generator.validate_generated_code(source, "builder", spec)

    assert report["passed"] is True, report


def test_paper_detail_and_executable_endpoint_return_persisted_code(client, db_session):
    report = {
        "passed": True,
        "entrypoint_class": "ResNetBuilder",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000],
    }
    paper = Paper(
        title="Phase 1 persisted code",
        visibility="public",
        generated_code_source="import torch\nprint('ready')",
        generated_code_compiled={"language": "python", "framework": "pytorch"},
        generation_status="success",
        verification_report=report,
        architecture_graph={"nodes": [], "edges": [], "ingestion": {}},
    )
    db_session.add(paper)
    db_session.commit()

    detail = client.get(f"/api/papers/{paper.id}")
    assert detail.status_code == 200
    assert detail.json()["generated_code_source"].startswith("import torch")
    assert detail.json()["verification_report"] == report

    executable = client.get(f"/api/papers/{paper.id}/executable-graph")
    assert executable.status_code == 200
    assert executable.json()["status"] == "success"
    assert executable.json()["code"].startswith("import torch")
    assert executable.json()["verification_report"] == report


def test_accepted_upload_returns_async_contract(
    client, db_session, regular_user_headers, tmp_path
):
    stored = tmp_path / "phase1.pdf"
    stored.write_bytes(b"%PDF-1.4 fixture")

    with (
        patch("backend.services.storage_service.store_pdf", return_value=str(stored)),
        patch("backend.routers.papers_pipeline.generate_code_from_pdf_task.delay") as delay,
    ):
        response = client.post(
            "/api/papers/upload",
            headers=regular_user_headers,
            data={"terms_accepted": "true", "visibility": "private"},
            files={"file": ("phase1.pdf", b"%PDF-1.4 fixture", "application/pdf")},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["paper_id"] is None
    assert body["task_id"]
    assert body["poll_url"] == f"/api/tasks/{body['task_id']}"
    delay.assert_called_once()
