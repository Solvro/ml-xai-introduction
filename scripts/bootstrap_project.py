#!/usr/bin/env python3
"""Rebrand the template: rename package, update configs, truncate README."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DELETE_MARKER = "**DELETE EVERYTHING ABOVE FOR YOUR PROJECT**"
OLD_PACKAGE = "pytorch_research_template"
OLD_KEBAB = "pytorch-research-template"
DEFAULT_DESCRIPTION = "Image classification research project"
DEFAULT_AUTHOR = "Your Team"
DEFAULT_EMAIL = "team@example.com"


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    name: str
    author: str
    email: str
    description: str
    attribution_line: str | None

    @property
    def package(self) -> str:
        return to_snake(self.name)

    @property
    def kebab(self) -> str:
        return to_kebab(self.name)


def find_project_root() -> Path:
    path = Path(__file__).resolve().parent
    for parent in [path, *path.parents]:
        if (parent / ".project-root").is_file():
            return parent
    raise RuntimeError("Could not find .project-root marker")


def to_snake(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", normalized):
        raise ValueError(f"Invalid project name: {name!r} (use letters, digits, hyphens)")
    return normalized


def to_kebab(name: str) -> str:
    return to_snake(name).replace("_", "-")


def prompt_value(label: str, default: str | None = None, required: bool = False) -> str:
    if default:
        raw = input(f"{label} [{default}]: ").strip()
        return raw or default
    while True:
        raw = input(f"{label}: ").strip()
        if raw:
            return raw
        if not required:
            return ""
        print("  This field is required.")


def resolve_config(args: argparse.Namespace) -> BootstrapConfig:
    interactive = sys.stdin.isatty()

    if args.name:
        name = args.name
    elif interactive:
        name = prompt_value("Project name (kebab-case or snake_case)", required=True)
    else:
        raise ValueError("--name is required when stdin is not interactive")

    if args.author is not None:
        author = args.author
    elif interactive:
        author = prompt_value("Author", default=DEFAULT_AUTHOR)
    else:
        author = DEFAULT_AUTHOR

    if args.email is not None:
        email = args.email
    elif interactive:
        email = prompt_value("Email", default=DEFAULT_EMAIL)
    else:
        email = DEFAULT_EMAIL

    if args.description is not None:
        description = args.description
    elif interactive:
        description = prompt_value("Description", default=DEFAULT_DESCRIPTION)
    else:
        description = DEFAULT_DESCRIPTION

    attribution = args.attribution_line
    if attribution is None and interactive:
        raw = prompt_value("Attribution line in README (optional, Enter to skip)", default="")
        attribution = raw or None

    return BootstrapConfig(
        name=name,
        author=author,
        email=email,
        description=description,
        attribution_line=attribution,
    )


def print_plan(config: BootstrapConfig, dry_run: bool) -> None:
    print("\nBootstrap plan:")
    print(f"  project:     {config.kebab}")
    print(f"  package:     {config.package}")
    print(f"  author:      {config.author} <{config.email}>")
    print(f"  description: {config.description}")
    print(f"  src/:        {OLD_PACKAGE} -> {config.package}")
    print("  updates:     imports, pyproject.toml, logging config, README stub")
    if dry_run:
        print("\n(dry-run — no files will be written)")


def confirm_proceed() -> bool:
    answer = input("\nProceed? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def git_is_dirty(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def iter_text_files(root: Path) -> list[Path]:
    skip_dirs = {".git", ".venv", "outputs", "__pycache__", ".pytest_cache", ".ruff_cache"}
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix in {".py", ".toml", ".yaml", ".yml", ".md", ".cff", ".example"}:
            files.append(path)
    return files


def replace_in_file(path: Path, old: str, new: str, dry_run: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        return False
    if not dry_run:
        path.write_text(text.replace(old, new), encoding="utf-8")
    return True


def truncate_readme(
    readme: Path,
    project_name: str,
    description: str,
    package: str,
    attribution: str | None,
    dry_run: bool,
) -> None:
    text = readme.read_text(encoding="utf-8")
    if DELETE_MARKER not in text:
        raise RuntimeError(f"README missing marker: {DELETE_MARKER}")
    stub = text.split(DELETE_MARKER, 1)[1].lstrip("\n")
    stub = stub.replace("{project_name}", project_name)
    stub = stub.replace("{description}", description)
    stub = stub.replace("{package}", package)
    if attribution:
        stub = f"<!-- {attribution} -->\n\n{stub}"
    if not dry_run:
        readme.write_text(stub, encoding="utf-8")


def update_pyproject(path: Path, config: BootstrapConfig, dry_run: bool) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'^name = ".*"$', f'name = "{config.kebab}"', text, count=1, flags=re.M)
    text = re.sub(r'^description = ".*"$', f'description = "{config.description}"', text, count=1, flags=re.M)
    text = re.sub(
        r"(\[project\.scripts\]\n)[^\n]+ = .*$",
        rf'\1{config.kebab} = "{config.package}.train:main"',
        text,
        count=1,
        flags=re.M,
    )
    authors_block = f'authors = [\n    {{ name = "{config.author}", email = "{config.email}" }},\n]'
    text = re.sub(r"authors = \[[\s\S]*?\]", authors_block, text, count=1)
    if not dry_run:
        path.write_text(text, encoding="utf-8")


def apply_bootstrap(root: Path, config: BootstrapConfig, dry_run: bool) -> list[str]:
    package = config.package
    kebab = config.kebab
    changed: list[str] = []

    old_src = root / "src" / OLD_PACKAGE
    new_src = root / "src" / package
    if not old_src.is_dir():
        raise RuntimeError(f"Expected {old_src} — already bootstrapped?")

    if not dry_run:
        if new_src.exists():
            raise RuntimeError(f"{new_src} already exists")
        old_src.rename(new_src)
    changed.append(f"src/{OLD_PACKAGE}/ -> src/{package}/")

    for path in iter_text_files(root):
        rel = path.relative_to(root)
        if path == root / "scripts" / "bootstrap_project.py":
            continue
        for old, new in [
            (OLD_PACKAGE, package),
            (OLD_KEBAB, kebab),
            (f'"{OLD_KEBAB}"', f'"{kebab}"'),
            (OLD_KEBAB, kebab),
        ]:
            if replace_in_file(path, old, new, dry_run):
                changed.append(str(rel))

    update_pyproject(root / "pyproject.toml", config, dry_run)
    changed.append("pyproject.toml")

    truncate_readme(root / "README.md", kebab, config.description, package, config.attribution_line, dry_run)
    changed.append("README.md (truncated)")

    citation = root / "CITATION.cff"
    if citation.exists() and not dry_run:
        citation.unlink()
        changed.append("CITATION.cff (removed)")

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap template into your project.",
        epilog="Run without flags to be prompted for missing values.",
    )
    parser.add_argument("--name", default=None, help="Project name (kebab-case or snake_case)")
    parser.add_argument("--author", default=None, help=f"Author name (default: {DEFAULT_AUTHOR})")
    parser.add_argument("--email", default=None, help=f"Author email (default: {DEFAULT_EMAIL})")
    parser.add_argument(
        "--description",
        default=None,
        help=f"Short project description (default: {DEFAULT_DESCRIPTION})",
    )
    parser.add_argument("--attribution-line", default=None, help="Optional HTML comment in README stub")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing files")
    parser.add_argument("--yes", action="store_true", help="Skip final confirmation prompt")
    parser.add_argument("--force", action="store_true", help="Run even if git working tree is dirty")
    args = parser.parse_args()

    root = find_project_root()

    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if config.package == OLD_PACKAGE:
        print("New name matches template package; choose a different name", file=sys.stderr)
        return 1

    if git_is_dirty(root) and not args.force and not args.dry_run:
        print("Git working tree is dirty. Commit or pass --force.", file=sys.stderr)
        return 1

    print_plan(config, args.dry_run)
    if not args.dry_run and not args.yes and not confirm_proceed():
        print("Aborted.")
        return 1

    try:
        changed = apply_bootstrap(root, config, args.dry_run)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    print("\nBootstrap summary:")
    for item in sorted(set(changed)):
        print(f"  - {item}")
    if args.dry_run:
        print("\n(dry-run — no files written)")
    else:
        print("\nTip: If this template helped your research, consider a shoutout in your paper or README.")
        print("Run: uv sync && make quality")
    return 0


if __name__ == "__main__":
    sys.exit(main())
