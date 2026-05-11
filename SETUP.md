# Project Setup Guide: Initializing with `uv` Package Manager

This guide explains how to initialize the `ml-xai-introduction` project using `uv` as a Python package with automatic import discovery from the `src` folder.

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
├── .git/
└── SETUP.md
```

This structure allows your code to be:
- **Installable** as a proper Python package
- **Importable** from anywhere in your project
- **Distributable** to PyPI or other package repositories

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
uv sync
```

**What this does:**
- Installs your project in **editable mode** (development mode)
- Creates a virtual environment if one doesn't exist
- Makes your package importable with `from ml_xai_introduction import ...`
- Locks dependencies for reproducibility

**Why:**
- Editable mode means changes to your code are immediately reflected
- You don't need to reinstall after each code change
- `uv sync` creates a `.venv` (or similar) with an isolated Python environment

### Step 3 (Optional): Add Development Tools

```bash
uv add --dev pytest black isort flake8 mypy
```

**What this does:**
- Installs development dependencies:
  - `pytest`: For unit testing
  - `black`: Code formatter
  - `isort`: Import sorter
  - `flake8`: Linter (code quality check)
  - `mypy`: Type checker

**Why:**
- These tools help maintain code quality and consistency
- `--dev` flag separates them from production dependencies
- They're only installed in the development environment, not in production builds

---

## Project Structure Explained

### `src/ml_xai_introduction/`

This folder contains your actual package code. By putting it in `src/`:
- **Cleaner separation** between your code and configuration files
- **Prevents import confusion** - Python won't accidentally import from the root directory
- **Industry standard** for Python packages

### `pyproject.toml`

The configuration file that tells `uv` and other tools about your project:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ml-xai-introduction"
version = "0.1.0"
description = "ML XAI Introduction Project"
```

- **build-system**: Specifies how to package your project
- **project**: Metadata about your project (name, version, description, dependencies)

### `.venv/` (created by `uv sync`)

Virtual environment containing isolated Python packages. Never commit this to git.

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
| `uv init --package` | Initialize project structure |
| `uv sync` | Install project + dependencies in editable mode |
| `uv add <package>` | Add a production dependency |
| `uv add --dev <package>` | Add a development dependency |
| `uv lock` | Update lock file (pins exact versions) |
| `uv run python` | Run Python in the virtual environment |
| `uv pip list` | List installed packages |

---

## Activation and Usage

### Using the Virtual Environment

```bash
# Activate (optional, uv commands activate automatically)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run Python with uv
uv run python my_script.py

# Run tests
uv run pytest
```

### Running Scripts with Proper Imports

```bash
# This ensures imports work correctly
uv run python -c "from ml_xai_introduction import module; print(module)"
```

---

## GitHub Actions Workflows

The project includes automated workflows that run on every pull request to ensure code quality and consistency.

### `.github/workflows/python-ci.yml`

**When it runs:**
- Automatically triggers on every pull request to `main` branch

**What it does:**

1. **Setup Environment**
   - Installs Python 3.10 on GitHub's Ubuntu server
   - Installs `uv` using the official `astral-sh/setup-uv@v2` action
   - Enables caching for faster subsequent runs

2. **Install Dependencies**
   ```bash
   uv sync --all-extras
   ```
   - Reads your `pyproject.toml` and `uv.lock`
   - Installs all dependencies (including dev tools like pytest, ruff)
   - Creates a reproducible environment identical to your local setup

3. **Run Linter (Ruff Check)**
   ```bash
   uv run ruff check --output-format=github .
   ```
   - Checks for syntax errors and code style issues
   - Detects unused imports and other problems
   - Reports violations in GitHub's format for easy viewing in PR

4. **Run Formatter Check (Ruff Format)**
   ```bash
   uv run ruff format --check .
   ```
   - Verifies code is properly formatted
   - Ensures consistent style across the project
   - Fails if code doesn't match Black-style formatting

5. **Run Unit Tests**
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

### `.github/workflows/branch-naming.yml`

**When it runs:**
- Automatically triggers on every pull request to `main` branch

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
- `docs/update-readme`
- `refactor/simplify-data-loader`

**Why:** Keeps the git history organized and makes it easy to understand what each PR does at a glance.

---

## Summary

The initialization process:

1. **`uv init --package`** → Creates packaging structure with `hatchling`
2. **`uv sync`** → Installs your project in editable mode, enables imports
3. **`uv add --dev [tools]`** → Adds development utilities

This setup ensures:
- ✅ Auto-discovery of imports from `src/`
- ✅ Clean, maintainable project structure
- ✅ Easy distribution-ready packaging
- ✅ Isolated, reproducible environment
- ✅ Modern Python best practices
