"""
backend/services/problem_seed_service.py

Idempotent startup seeding of the dojo Problem table from
backend/scripts/json_dump/dojo_problems_seed.json.

The frontend (src/data/problems.ts) identifies problems by slug — these rows
use id == slug so POST /api/dojo/code-submissions {problem_id: <slug>} resolves.
Each row carries a judge harness in test_cases (type == "harness") that the
execution worker appends to submitted code; passing means the harness's
asserts all hold (exit code 0).

Existing rows are refreshed (title/difficulty/template/test_cases) so harness
fixes ship with a redeploy; user-generated fields (acceptance_rate, version)
are left untouched.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_SEED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "json_dump", "dojo_problems_seed.json"
)


def seed_dojo_problems(db) -> int:
    from backend.models import Problem

    path = os.path.abspath(_SEED_PATH)
    if not os.path.exists(path):
        logger.warning("Dojo problem seed file missing: %s", path)
        return 0

    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    changed = 0
    for e in entries:
        row = db.query(Problem).filter_by(id=e["id"]).first()
        if row is None:
            row = Problem(
                id=e["id"],
                slug=e["slug"],
                title=e["title"],
                category=e.get("category"),
                difficulty=e.get("difficulty"),
                estimated_time=e.get("estimated_time"),
                description=e.get("description", ""),
                tags=e.get("tags", []),
                related_architectures=[],
                related_papers=[],
                related_math=[],
                learning_points=[],
                python_template=e.get("python_template", ""),
                test_cases=e.get("test_cases", []),
                hints={},
                explanation={},
            )
            db.add(row)
            changed += 1
        else:
            row.title = e["title"]
            row.category = e.get("category")
            row.difficulty = e.get("difficulty")
            row.description = e.get("description", "")
            row.python_template = e.get("python_template", "")
            row.test_cases = e.get("test_cases", [])
    db.commit()
    logger.info("Dojo problems seeded (%d new, %d total in seed file)", changed, len(entries))
    return changed
