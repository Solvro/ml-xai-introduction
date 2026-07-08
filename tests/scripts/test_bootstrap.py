from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PACKAGE = ROOT / "src" / "pytorch_research_template"
IS_TEMPLATE_REPO = TEMPLATE_PACKAGE.is_dir()

requires_template = pytest.mark.skipif(
    not IS_TEMPLATE_REPO,
    reason="Bootstrap tests require the untouched template package (already bootstrapped?)",
)


def _run_bootstrap(
    script: Path,
    args: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        input=input_text,
        check=False,
    )


@requires_template
def test_bootstrap_dry_run_on_repo() -> None:
    result = _run_bootstrap(
        ROOT / "scripts" / "bootstrap_project.py",
        ["--name", "test_lab_cnn", "--dry-run", "--yes", "--force"],
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "Bootstrap plan:" in result.stdout
    assert "dry-run" in result.stdout
    assert TEMPLATE_PACKAGE.is_dir()


@requires_template
def test_bootstrap_renames_package_in_temp(tmp_path: Path) -> None:
    dest = tmp_path / "copy"
    shutil.copytree(
        ROOT,
        dest,
        ignore=shutil.ignore_patterns(".venv", "outputs", ".git", "__pycache__", ".pytest_cache"),
    )
    result = _run_bootstrap(
        dest / "scripts" / "bootstrap_project.py",
        ["--name", "temp_lab", "--yes", "--force"],
        cwd=dest,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (dest / "src" / "temp_lab").is_dir()
    assert not (dest / "src" / "pytorch_research_template").exists()
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "DELETE EVERYTHING ABOVE" not in readme
    assert "temp-lab" in readme
    assert not (dest / "CITATION.cff").exists()
    assert (dest / "LICENSE").exists()


def test_bootstrap_requires_name_when_non_interactive() -> None:
    result = _run_bootstrap(
        ROOT / "scripts" / "bootstrap_project.py",
        ["--dry-run", "--force"],
        cwd=ROOT,
    )
    assert result.returncode == 1
    assert "--name is required" in result.stderr


@requires_template
def test_bootstrap_requires_yes_when_non_interactive(tmp_path: Path) -> None:
    dest = tmp_path / "copy"
    shutil.copytree(
        ROOT,
        dest,
        ignore=shutil.ignore_patterns(".venv", "outputs", ".git", "__pycache__", ".pytest_cache"),
    )
    result = _run_bootstrap(
        dest / "scripts" / "bootstrap_project.py",
        ["--name", "test_lab", "--force"],
        cwd=dest,
    )
    assert result.returncode == 1
    assert "Pass --yes" in result.stderr
    assert (dest / "src" / "pytorch_research_template").is_dir()


@requires_template
def test_bootstrap_piped_interactive(tmp_path: Path) -> None:
    dest = tmp_path / "piped"
    shutil.copytree(
        ROOT,
        dest,
        ignore=shutil.ignore_patterns(".venv", "outputs", ".git", "__pycache__", ".pytest_cache"),
    )
    stdin = (
        "\n".join([
            "piped-lab",
            "Piped Author",
            "piped@example.com",
            "Piped description",
            "Template credit",
            "y",
        ])
        + "\n"
    )
    result = _run_bootstrap(
        dest / "scripts" / "bootstrap_project.py",
        ["--force"],
        cwd=dest,
        input_text=stdin,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (dest / "src" / "piped_lab").is_dir()
    assert "piped-lab" in (dest / "README.md").read_text(encoding="utf-8")


@requires_template
def test_bootstrap_piped_abort(tmp_path: Path) -> None:
    dest = tmp_path / "abort"
    shutil.copytree(
        ROOT,
        dest,
        ignore=shutil.ignore_patterns(".venv", "outputs", ".git", "__pycache__", ".pytest_cache"),
    )
    result = _run_bootstrap(
        dest / "scripts" / "bootstrap_project.py",
        ["--force"],
        cwd=dest,
        input_text="abort-lab\n\n\n\n\nn\n",
    )
    assert result.returncode == 1
    assert "Aborted" in result.stdout
    assert (dest / "src" / "pytorch_research_template").is_dir()
