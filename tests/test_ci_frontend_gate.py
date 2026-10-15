from pathlib import Path

import yaml


def test_ci_runs_real_frontend_gates_with_locked_installs():
    jobs = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())["jobs"]
    commands = "\n".join(str(step.get("run", "")) for job in jobs.values() for step in job["steps"])
    for command in (
        "npm ci",
        "npx tsc --noEmit -p tsconfig.json",
        "npx vitest run",
        "npm run test:e2e:production",
    ):
        assert command in commands, f"Missing CI gate: {command}"
    assert "npm install" not in commands
    assert "Sandbox execution simulation" not in commands
