"""Opt-in deployed Phase 1 smoke test.

Run only with RUN_LIVE_PAPER_PIPELINE=1, PAPER2CODE_LIVE_API_URL, and
PAPER2CODE_LIVE_TOKEN.  The deployed backend must have Redis/Celery, LLM, and
E2B configured.
"""

import base64
import os
import time
from pathlib import Path

import httpx
import pytest

_ENABLED = os.getenv("RUN_LIVE_PAPER_PIPELINE") == "1"
_BASE = os.getenv("PAPER2CODE_LIVE_API_URL", "").rstrip("/")
_TOKEN = os.getenv("PAPER2CODE_LIVE_TOKEN", "")


@pytest.mark.live
@pytest.mark.skipif(
    not (_ENABLED and _BASE and _TOKEN),
    reason="live paper pipeline environment is not configured",
)
def test_live_upload_to_persisted_workspace_and_sandbox():
    fixture = Path(__file__).parents[1] / "fixtures" / "phase1_architecture.pdf.b64"
    pdf_bytes = base64.b64decode(fixture.read_text(encoding="ascii").strip())
    headers = {"Authorization": f"Bearer {_TOKEN}"}

    with httpx.Client(base_url=_BASE, headers=headers, timeout=90.0) as client:
        upload = client.post(
            "/api/papers/upload",
            data={"terms_accepted": "true", "visibility": "private"},
            files={"file": ("phase1_architecture.pdf", pdf_bytes, "application/pdf")},
        )
        upload.raise_for_status()
        envelope = upload.json()
        assert envelope["paper_id"] is None

        deadline = time.monotonic() + 600
        task = None
        while time.monotonic() < deadline:
            response = client.get(envelope["poll_url"])
            response.raise_for_status()
            task = response.json()
            if task["status"] in ("completed", "failed"):
                break
            time.sleep(1)

        assert task is not None and task["status"] == "completed", task
        paper_id = task["result"]["paper_id"]
        detail = client.get(f"/api/papers/{paper_id}")
        detail.raise_for_status()
        assert detail.json()["generated_code_source"]
        assert detail.json()["verification_report"] is not None

        executable = client.get(f"/api/papers/{paper_id}/executable-graph")
        executable.raise_for_status()
        generated = executable.json()
        assert generated["code"]

        run = client.post("/api/dojo/execute", json={"code": generated["code"], "stdin": ""})
        run.raise_for_status()
        assert run.json()["exit_code"] == 0, run.json()
