# Packaging and Release Guide

This project uses modern Python packaging standards with `pyproject.toml` and `setuptools-scm` for version management.

## Version Management

Project uses `setuptools-scm` to automatically generate version numbers from Git tags:

- **Release versions**: Generated from git tags (e.g., `v1.0.0` → `1.0.0`)
- **Development versions**: Include commit info (e.g., `1.0.0.dev123+g1234567`)
- **Version file**: Auto-generated to `sml/_version.py`

## Development Setup

```bash
# Clone and setup
git clone https://github.com/secretflow/sml.git
cd sml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

## Building

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Verify build
twine check dist/*
```

## Release Process

```bash
# 1. Prepare release
git status  # ensure clean working directory
pytest      # run tests
pylint sml/ # code quality check

# 2. Create release tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# 3. Build and publish
rm -rf dist/ build/
python -m build
twine check dist/*
twine upload dist/*
```

## Version Strategy

- Follow [Semantic Versioning](https://semver.org/)
- Tag format: `v1.0.0`, `v1.0.0rc1`, `v1.0.0a1`
- Branches: `main` (stable), `develop`, `feature/*`, `hotfix/*`

## CI/CD Integration

Example GitHub Actions (`.github/workflows/release.yml`):

```yaml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0  # Required for setuptools-scm

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Build and publish
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        pip install build twine
        python -m build
        twine upload dist/*
```

## Common Issues

### Version shows as `0.1.0.dev0`
- Ensure git repository has tags: `git tag -l`
- Ensure complete git history: `git fetch --unshallow`

### Build failures
- Check `pyproject.toml` syntax
- Install build dependencies: `pip install build`

### Import version in code
```python
try:
    from importlib.metadata import version
    __version__ = version("sf-sml")
except ImportError:
    from importlib_metadata import version
    __version__ = version("sf-sml")
```

## Migration from setup.py

If migrating from `setup.py` + `version.py`:

```bash
# Remove old files
rm setup.py version.py

# Update version imports in code (see above)

# Create initial tag
git tag v0.1.0
```

### Breaking Changes
- No longer supports `$$DATE$$` and `$$COMMIT_ID$$` placeholders
- Build command changed from `python setup.py` to `python -m build`
- Version import mechanism changed

## References

- [Python Packaging User Guide](https://packaging.python.org/)
- [setuptools-scm Documentation](https://setuptools-scm.readthedocs.io/)
- [PEP 517](https://peps.python.org/pep-0517/) - Build system standard
- [PEP 518](https://peps.python.org/pep-0518/) - pyproject.toml standard
