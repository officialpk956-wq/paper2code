import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("database", ["memory", "file"])
def test_registration_through_real_startup(database, tmp_path):
    url = "sqlite:///:memory:" if database == "memory" else f"sqlite:///{tmp_path / 'fresh.db'}"
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/startup_auth_probe.py"), url],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-3000:]
