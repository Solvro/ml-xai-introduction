#!/usr/bin/env python3
"""Print a unified quality report (ruff, pytest, optional smoke train)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _find_project_root() -> Path:
    path = Path(__file__).resolve().parent
    for parent in [path, *path.parents]:
        if (parent / ".project-root").is_file():
            return parent
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def _project_name(root: Path) -> str:
    import tomllib

    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["name"])


def _package_dir(root: Path) -> Path | None:
    src = root / "src"
    if not src.is_dir():
        return None
    packages = [p for p in src.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return packages[0] if len(packages) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run quality checks and print a summary report.")
    parser.add_argument("--full", action="store_true", help="Also run a 1-epoch smoke train.")
    args = parser.parse_args()

    root = _find_project_root()
    name = _project_name(root)
    width = 39
    lines: list[str] = [
        "═" * width,
        f" Quality Report — {name}",
        "═" * width,
    ]
    failed = False

    code, out = _run(["uv", "run", "ruff", "check", "."], root)
    status = "PASS" if code == 0 else "FAIL"
    if code != 0:
        failed = True
    issue_line = out.splitlines()[-1] if out else "ok"
    lines.append(f"Ruff lint     {status:<5}  {issue_line}")

    code, _ = _run(["uv", "run", "ruff", "format", "--check", "."], root)
    status = "PASS" if code == 0 else "FAIL"
    if code != 0:
        failed = True
    lines.append(f"Ruff format   {status}")

    code, out = _run(
        ["uv", "run", "pytest", "-q", "--cov=src", "--cov-report=term-missing:skip-covered"],
        root,
    )
    status = "PASS" if code == 0 else "FAIL"
    if code != 0:
        failed = True
    summary = next((line for line in out.splitlines() if "passed" in line or "failed" in line), out[-80:])
    lines.append(f"Pytest        {status:<5}  {summary}")

    if args.full:
        pkg = _package_dir(root)
        if pkg is None:
            lines.append("Smoke train   SKIP   multiple src packages")
        else:
            train_py = pkg / "train.py"
            smoke_out = "/tmp/ml-template-smoke-out"
            code, _ = _run(
                [
                    "uv",
                    "run",
                    "python",
                    str(train_py),
                    "training.epochs=1",
                    "logging=default",
                    f"paths.output_dir={smoke_out}",
                    f"paths.checkpoint_dir={smoke_out}/checkpoints",
                ],
                root,
            )
            status = "PASS" if code == 0 else "FAIL"
            if code != 0:
                failed = True
            lines.append(f"Smoke train   {status}")

    lines.append("─" * width)
    lines.append(f"Overall       {'FAIL' if failed else 'PASS'}")
    lines.append("═" * width)
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
