# SML (SecretFlow Secure Machine Learning) Context

## Project Overview
**SML** is a library implementing machine learning algorithms using [JAX](https://github.com/google/jax) for secure training and inference via [SPU](https://github.com/secretflow/spu) (SecretFlow Processing Unit). The goal is to provide a secure, privacy-preserving alternative to scikit-learn with compatible APIs.

## Architecture
- **Backend**: JAX (for computation graph) + SPU (for secure execution).
- **Core Principles**:
  - **Accuracy**: Optimized fixed-point arithmetic, high-precision ops.
  - **Efficiency**: MPC-friendly operations (avoiding heavy non-linear ops where possible).
  - **JIT-ability**: Code must be JIT-compilable by JAX.
  - **Control Flow**: No Python control flow (`if/else`, `while`) on secret data; use `jax.lax.cond`, `jax.lax.while_loop`.

## Directory Structure
- `sml/`: Source code package.
  - `linear_model/`: Linear models (Logistic, Ridge, GLM, etc.).
    - `design/`: Design documents for new implementations (e.g., JAX-GLM).
  - `...`: Other ML modules (cluster, ensemble, etc.).
- `tests/`: Unit tests using the **Simulator** (high speed, no real MPC environment).
- `emulations/`: Tests using the **Emulator** (multi-process/docker, real MPC protocol simulation).

## Development Workflow

### 1. Environment & Installation
The project is set up with a pre-installed virtual environment. To activate it, run:
```bash
source /home/jjzhou/codes/github_dev/sml/.venv/bin/activate
```
The project uses `uv` for dependency management and `setuptools` for building.

### 2. Testing
**Unit Tests (Simulator)** - Run these for correctness:
```bash
# Run all tests
pytest tests

# Run specific module tests
pytest tests/linear_model/glm_test.py
```

**Emulation Tests (Emulator)** - Run these for MPC efficiency/behavior:
```bash
# Run all emulations
python emulations/run_emulations.py

# Run specific module emulation
python emulations/run_emulations.py --module emulations.linear_model.glm_emul
```

### 3. Code Quality
The project uses `pre-commit` with `ruff` (linting/formatting) and `mypy`.
```bash
# Run all checks manually
pre-commit run --all-files
```
*Style*: **NumPy** style docstrings.

## Current Focus: JAX-GLM Implementation
We are currently implementing a new Generalized Linear Model (GLM) architecture in `sml/linear_model/`.

**Design Documents**: `sml/linear_model/design/`
- **Goal**: Decoupled, explicit math (no `jax.grad`), hand-optimized formulas.
- **Components**:
  - `core/`: Link functions, Distributions, Family.
  - `formula/`: Calculation strategies (Generic vs. Optimized).
  - `solvers/`: IRLS, Fisher Scoring (Newton-CG).
  - `model.py`: User-facing `GLM` estimator.

**Key Implementation Constraints**:
1.  **Explicit Math**: Do not use `jax.grad`, `jax.hessian`, or `jax.vmap` for the core GLM logic. Derive gradients/Hessians manually.
2.  **Canonical Links**: Distributions must support default canonical links.
3.  **Optimization**: Support "Hand-Optimized" formulas for specific (Distribution, Link) pairs (e.g., Tweedie + Log).
