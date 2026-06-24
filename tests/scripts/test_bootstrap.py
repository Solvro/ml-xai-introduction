from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def test_bootstrap_dry_run_on_repo() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "bootstrap_project.py"),
            "--name",
            "test_lab_cnn",
            "--dry-run",
            "--yes",
            "--force",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Bootstrap plan:" in result.stdout
    assert "dry-run" in result.stdout
    assert (root / "src" / "ml_xai_introduction").is_dir()


def test_bootstrap_renames_package_in_temp(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    dest = tmp_path / "copy"
    shutil.copytree(
        root,
        dest,
        ignore=shutil.ignore_patterns(".venv", "outputs", ".git", "__pycache__", ".pytest_cache"),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(dest / "scripts" / "bootstrap_project.py"),
            "--name",
            "temp_lab",
            "--yes",
            "--force",
        ],
        cwd=dest,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (dest / "src" / "temp_lab").is_dir()
    assert not (dest / "src" / "ml_xai_introduction").exists()
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "DELETE EVERYTHING ABOVE" not in readme
    assert "temp-lab" in readme
    assert not (dest / "CITATION.cff").exists()
    assert (dest / "LICENSE").exists()


def test_bootstrap_requires_name_when_non_interactive() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "bootstrap_project.py"),
            "--dry-run",
            "--force",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "--name is required" in result.stderr
