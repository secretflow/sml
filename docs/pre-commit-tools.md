# Pre-commit Tools Configuration

This document explains the pre-commit tools used in this project and how to use them.

## Overview

The project uses several automated tools to maintain code quality and consistency. These tools run automatically on every commit via pre-commit hooks.

## Tools and Their Purpose

### Code Formatters
- **Black**: Formats Python code consistently (spacing, indentation, line breaks)
- **isort**: Organizes and sorts import statements

### Code Cleaners
- **autoflake**: Removes unused imports and variables
- **pyupgrade**: Upgrades code to modern Python syntax

### Code Analyzers
- **flake8**: Checks code style, syntax errors, and enforces import rules
- **flake8-absolute-import**: Plugin that forbids relative imports

## Execution Order

Tools run in this order to avoid conflicts:

1. **Basic checks** (whitespace, file endings, etc.)
2. **pyupgrade** - Modernize syntax
3. **autoflake** - Remove unused code
4. **isort** - Organize imports
5. **black** - Format code
6. **flake8** - Final validation

## Usage

### Automatic (Recommended)
```bash
# Tools run automatically on commit
git commit -m "your changes"
```

### Manual Execution
```bash
# Run all tools on changed files
pre-commit run

# Run all tools on all files
pre-commit run --all-files

# Run specific tool
pre-commit run black --files path/to/file.py

# Run flake8 directly
flake8 path/to/file.py
```

## Import Rules

### Forbidden (Relative Imports)
```python
from . import module
from ..parent import function
from .submodule import class
```

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

- **`.flake8`**: flake8 configuration and rules
- **`.pre-commit-config.yaml`**: Tool execution and arguments

## Troubleshooting

If pre-commit hooks fail:
1. Review the error messages
2. Fix the reported issues
3. Stage your changes: `git add .`
4. Commit again: `git commit -m "your message"`

To temporarily skip hooks (not recommended):
```bash
git commit --no-verify
```
