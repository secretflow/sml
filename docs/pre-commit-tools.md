# Pre-commit Tools Configuration

This document explains the pre-commit tools used in this project and how to use them.

## Overview

The project uses **Ruff**, a modern and fast Python linter and formatter, to maintain code quality and consistency. Ruff replaces multiple traditional tools and runs automatically on every commit via pre-commit hooks.

## Ruff: All-in-One Tool

**Ruff** is a modern Python linter and formatter written in Rust that combines the functionality of multiple tools:

- **Replaces**: flake8, black, isort, autoflake, pyupgrade
- **Benefits**: Much faster execution, sensible defaults, comprehensive rule set
- **Configuration**: All settings in `pyproject.toml` following SSOT principle

### What Ruff Does

1. **Code Formatting**: Formats Python code consistently (replaces Black)
2. **Import Sorting**: Organizes and sorts import statements (replaces isort)  
3. **Code Linting**: Checks code style, syntax errors, and best practices (replaces flake8)
4. **Code Modernization**: Upgrades code to modern Python syntax (replaces pyupgrade)
5. **Unused Code Removal**: Removes unused imports and variables (replaces autoflake)

## Execution Order

Ruff runs in two phases during pre-commit:

1. **ruff check --fix** - Lints code and fixes auto-fixable issues
2. **ruff format** - Formats code for consistent style

## Usage

### Automatic (Recommended)
```bash
# Ruff runs automatically on commit
git commit -m "your changes"
```

### Manual Execution
```bash
# Run all pre-commit tools on changed files
pre-commit run

# Run all tools on all files
pre-commit run --all-files

# Run ruff linting with auto-fix
ruff check --fix path/to/file.py

# Run ruff formatter
ruff format path/to/file.py

# Check without fixing
ruff check path/to/file.py
```

## Ruff Configuration

All configuration is in `pyproject.toml`:

- **Line length**: 88 characters (Black-compatible)
- **Target version**: Python 3.10+
- **Selected rules**: Comprehensive set including pycodestyle, pyflakes, pyupgrade, isort, flake8-bugbear
- **Ignored rules**: Legacy issues and style preferences

## Import Rules

### Required (Absolute Imports)
```python
from sml.module import function
from sml.parent import function
from sml.submodule import class
```

### Exception
Version imports in `sml/__init__.py` are allowed:
```python
from ._version import version as __version__
```

## Configuration Files

- **`pyproject.toml`**: All Ruff configuration under `[tool.ruff]`
- **`.pre-commit-config.yaml`**: Pre-commit hook execution

## Troubleshooting

If pre-commit hooks fail:
1. Review the error messages from Ruff
2. Fix the reported issues (many can be auto-fixed)
3. Stage your changes: `git add .`
4. Commit again: `git commit -m "your message"`

For auto-fixable issues:
```bash
ruff check --fix .
git add .
git commit -m "your message"
```

To temporarily skip hooks (not recommended):
```bash
git commit --no-verify
```

## Migration from Legacy Tools

This project has migrated from multiple tools (flake8, black, isort, autoflake, pyupgrade) to Ruff for:
- **Better performance**: Ruff is orders of magnitude faster
- **Simplified setup**: One tool instead of five
- **Better integration**: Single configuration, consistent behavior
- **Modern defaults**: Up-to-date rules and best practices
