# Project Setup Guide: Initializing with `uv` Package Manager

This guide explains how to initialize the `ml-xai-introduction` project using `uv` as a Python package with automatic import discovery from the `src` folder.

## Table of Contents

1. [Overview](#overview)
2. [Step-by-Step Setup](#step-by-step-setup)
3. [Project Structure](#project-structure-explained)
4. [How Import Discovery Works](#how-import-discovery-works)
5. [GitHub Actions Workflows](#github-actions-workflows)
6. [Key Commands Reference](#key-commands-reference)
7. [Activation and Usage](#activation-and-usage)

---

## Overview

The project is structured as a distributable Python package with the following layout:

```
ml-xai-introduction/
├── src/
│   └── ml_xai_introduction/
│       ├── __init__.py
│       └── [your modules...]
├── pyproject.toml
├── README.md
├── .python-version
├── .gitignore
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── python-ci.yml
│       └── branch-naming.yml
└── SETUP.md
```

This structure allows your code to be:
- **Installable** as a proper Python package
- **Importable** from anywhere in your project
- **Distributable** to PyPI or other package repositories
- **Maintainable** with automated quality checks

---

## Step-by-Step Setup

### Step 1: Initialize the Project with `uv`

```bash
uv init --package
```

**What this does:**
- Creates `pyproject.toml` with default Python packaging configuration
- Sets up `hatchling` as the build backend (for building `.whl` and `.tar.gz` distributions)
- Creates a `README.md` template
- Configures the project to recognize `src/` as the package source directory

**Why:**
- `--package` flag tells `uv` to create a distribution-ready structure
- `pyproject.toml` is the modern standard for Python packaging (replaces `setup.py`)
- `hatchling` handles building your package when you want to distribute it

### Step 2: Synchronize Your Environment

```bash
uv sync --all-extras
```

**What this does:**
- Installs your project in **editable mode** (development mode)
- Creates a virtual environment if one doesn't exist
- Makes your package importable with `from ml_xai_introduction import ...`
- Locks dependencies for reproducibility in `uv.lock`

**Why:**
- Editable mode means changes to your code are immediately reflected
- You don't need to reinstall after each code change
- `uv sync` creates a `.venv` (or similar) with an isolated Python environment
- `uv.lock` ensures reproducible builds across machines

### Step 3: Install Pre-commit Hooks

```bash
pre-commit install
```

**What this does:**
- Installs git hooks that run before every commit
- Automatically checks code quality (trailing whitespace, YAML, ruff)
- Fixes or prevents commits that don't meet standards

**Why:**
- Catches issues locally before they're pushed
- Saves time in the PR review process
- Enforces consistent code style across the team

### Step 4 (Optional): Add More Development Dependencies

```bash
uv add --dev <package-name>
```

**Common packages:**
```bash
uv add --dev jupyter   # For notebooks
uv add --dev sphinx    # For documentation
```

---

## Project Structure Explained

### 1. `src/ml_xai_introduction/`

This folder contains your actual package code. By putting it in `src/`:
- **Cleaner separation** between your code and configuration files
- **Prevents import confusion** - Python won't accidentally import from the root directory
- **Industry standard** for Python packages

### 2. `pyproject.toml`

The configuration file that tells `uv` and other tools about your project:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ml-xai-introduction"
version = "0.1.0"
description = "ML XAI Introduction Project"

[dependency-groups]
dev = ["pytest", "black", "mypy", "pre-commit", ...]
```

- **build-system**: Specifies how to package your project
- **project**: Metadata about your project (name, version, description, dependencies)
- **dependency-groups**: Separates production and dev dependencies

### 3. `.python-version`

Specifies which Python version this project uses:
```
3.11
```

Tools like `uv`, `pyenv`, and `asdf` automatically use this file.

### 4. `.venv/` (created by `uv sync`)

Virtual environment containing isolated Python packages. **Never commit this to git** - it's in `.gitignore`.

### 5. `.gitignore`

Prevents committing:
- `__pycache__/`, `*.pyc` (Python cache)
- `.venv/`, `uv.lock` (dependencies)
- `.coverage/`, `.pytest_cache/` (testing artifacts)
- `.idea/`, `.vscode/` (IDE settings)
- And more...

### 6. `.pre-commit-config.yaml`

Defines which code quality checks run before each commit:
- `trailing-whitespace` - removes trailing spaces
- `end-of-file-fixer` - ensures files end with newline
- `check-yaml` - validates YAML syntax
- `ruff` - linting and import sorting
- `ruff-format` - code formatting

---

## How Import Discovery Works

After running `uv sync`, Python knows where to find your modules through:

1. **Editable installation**: Your package is installed in development mode
2. **`src/` recognition**: `hatchling` is configured to look in `src/` for packages
3. **Package metadata**: `pyproject.toml` defines the package structure

### Example: Making an Import Work

**Before `uv sync`:**
```python
from ml_xai_introduction import my_module  # ❌ ImportError
```

**After `uv sync`:**
```python
from ml_xai_introduction import my_module  # ✅ Works!
```

The virtual environment and editable installation make this possible.

---

## Key Commands Reference

| Command | Purpose |
|---------|---------|
| `uv init --package` | Initialize project structure with packaging |
| `uv sync --all-extras` | Install project + all dependencies in editable mode |
| `uv add <package>` | Add a production dependency |
| `uv add --dev <package>` | Add a development dependency |
| `uv lock` | Update lock file (pins exact versions) |
| `uv run python` | Run Python in the virtual environment |
| `uv run pytest` | Run tests |
| `uv run ruff check .` | Run linter |
| `uv run ruff format .` | Format code |
| `pre-commit install` | Install local git hooks |
| `pre-commit run --all-files` | Run pre-commit checks manually |
| `uv pip list` | List installed packages |

---

## Activation and Usage

### Using the Virtual Environment

```bash
# Activate manually (optional, uv commands activate automatically)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Deactivate
deactivate

# Run Python with uv (recommended)
uv run python my_script.py

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src
```

### Running Scripts with Proper Imports

```bash
# This ensures imports work correctly
uv run python -c "from ml_xai_introduction import module; print(module)"

# Run a script
uv run python scripts/my_script.py
```

### Working with the Pre-commit Hooks

```bash
# Install hooks (one-time setup)
pre-commit install

# Run hooks on all files
pre-commit run --all-files

# Skip hooks for a commit (not recommended!)
git commit --no-verify
```

---

## GitHub Actions Workflows

The project includes automated workflows that run on every pull request to ensure code quality and consistency.

### 1. `.github/workflows/python-ci.yml`

**Trigger:** Automatically runs on every pull request to `main` branch

**Pipeline Steps:**

#### Step 1: Setup Environment
- Installs Python 3.10 on GitHub's Ubuntu server
- Installs `uv` using the official `astral-sh/setup-uv@v2` action
- Enables caching for faster subsequent runs

#### Step 2: Install Dependencies
```bash
uv sync --all-extras
```
- Reads your `pyproject.toml` and `uv.lock`
- Installs all dependencies (including dev tools like pytest, ruff, pre-commit)
- Creates a reproducible environment identical to your local setup

#### Step 3: Run Pre-commit Hooks
```bash
uv run pre-commit run --all-files
```
- Checks for trailing whitespace
- Validates YAML files
- Runs ruff linter and formatter
- **This is the gatekeeper** - ensures code quality standards

#### Step 4: Run Unit Tests
```bash
uv run pytest
```
- Executes all tests in the `tests/` directory
- Reports failures with detailed error messages
- Ensures new code doesn't break existing functionality

**Why this workflow matters:**
- ✅ **Consistency**: Same tools, same versions both locally and in CI
- ✅ **Quality Gate**: Bad code is caught before merging
- ✅ **Fast Feedback**: Developers know immediately if their code passes checks
- ✅ **Reproducibility**: Uses `uv.lock` to guarantee identical environments

### 2. `.github/workflows/branch-naming.yml`

**Trigger:** Automatically runs on every pull request to `main` branch

**What it does:**
- Validates that branch names follow a naming convention
- Requires branches to start with one of these prefixes:
  - `feat/` - new features
  - `fix/` - bug fixes
  - `docs/` - documentation changes
  - `refactor/` - code refactoring
  - `test/` - test additions/changes
  - `chore/` - maintenance, dependencies
  - `ci/` - CI/CD related changes

**Example valid branch names:**
- `feat/add-explainability-metrics`
- `fix/correct-import-error`
- `docs/setup-final`
- `refactor/simplify-data-loader`

**Why:** Keeps the git history organized and makes it easy to understand what each PR does at a glance.

---

## Summary

### The Initialization Process:

1. **`uv init --package`** → Creates packaging structure with `hatchling`
2. **`uv sync --all-extras`** → Installs project in editable mode, enables imports
3. **`pre-commit install`** → Sets up local code quality checks
4. **Push to GitHub** → CI/CD workflows validate your code automatically

### What This Setup Ensures:

- ✅ Auto-discovery of imports from `src/`
- ✅ Clean, maintainable project structure
- ✅ Easy distribution-ready packaging
- ✅ Isolated, reproducible environment
- ✅ Automated code quality checks (locally + CI)
- ✅ Professional development workflow
- ✅ Modern Python best practices

### Next Steps:

1. Run `uv init --package` to update structure
2. Run `uv sync --all-extras` to install dependencies
3. Run `pre-commit install` to enable local hooks
4. Create a branch: `git checkout -b docs/setup-final`
5. Commit changes: `git add . && git commit -m "docs: finalize setup documentation"`
6. Push: `git push origin docs/setup-final`
7. Open a Pull Request on GitHub
8. Let CI/CD workflows validate your changes
9. Merge once all checks pass ✅
