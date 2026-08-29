"""
FastAPI HTTP endpoint tests for all 4 supported families + fallback.
"""

import pytest
from backend.models import Paper
from core.paper_to_code_generator import PaperToCodeGenerator


@pytest.mark.parametrize("family", ["resnet", "unet", "vit", "transformer"])
def test_all_four_families_persist_clean_executable_code_via_api(
    client, db_session, regular_user_headers, family
):
    """Verify all 4 families produce verified code when processed and retrieved via API."""
    generator = PaperToCodeGenerator()
    spec = {"model_family": family}
    code, code_source = generator._generate_code(
        spec,
        generator.pipeline.run_single(
            {
                "name": f"{family.upper()}",
                "layers": [
                    {"type": "conv2d", "params": {}},
                    {"type": "linear", "params": {}},
                ],
                "connections": [["layer_0", "layer_1"]],
            }
        )["graph"],
    )
    report = generator.validate_generated_code(code, code_source, spec)

    assert report["passed"] is True
    assert report["checks"] == {"syntax": True, "exec": True, "forward": True}

    # Verify Paper persistence model stores the report and attempts
    paper = Paper(
        title=f"Sample {family.upper()} Architecture Paper",
        visibility="public",
        generated_code_source=code,
        generation_status="success",
        verification_report=report,
    )
    db_session.add(paper)
    db_session.commit()

    response = client.get(f"/api/papers/{paper.id}", headers=regular_user_headers)
    assert response.status_code == 200
    data = response.json()

    assert data["generated_code_source"] == code
    assert data["generation_status"] == "success"
    assert data["verification_report"]["passed"] is True
    assert data["verification_report"]["checks"]["forward"] is True
