# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Protocol

import jax

from sml.linear_model.glm.core.family import Family


class Solver(Protocol):
    """Protocol for GLM solvers (IRLS, Newton-CG, SGD, etc.)."""

    def solve(
        self,
        X: jax.Array,
        y: jax.Array,
        family: Family,
        fit_intercept: bool = True,
        offset: jax.Array | None = None,
        sample_weight: jax.Array | None = None,
        l2: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        stopping_rule: str = "beta",
        learning_rate: float = 1e-2,
        decay_rate: float = 1.0,
        decay_steps: int = 1,
        batch_size: int = 128,
        enable_spu_cache: bool = False,
        enable_spu_reveal: bool = False,
    ) -> tuple[jax.Array, jax.Array | None, dict[str, Any] | None]:
        """
        Solve the GLM optimization problem.

        Parameters
        ----------
        X : jax.Array
            Feature matrix.
        y : jax.Array
            Target values.
        family : Family
            GLM family instance.
        formula : Formula
            Formula strategy for component calculation.
        fit_intercept : bool
            Whether to include an intercept term.
        offset : jax.Array, optional
            Fixed offset added to linear predictor.
        sample_weight : jax.Array, optional
            Sample weights.
        l2 : float
            L2 regularization strength.
        max_iter : int
            Maximum number of iterations (or epochs for SGD).
        tol : float
            Convergence tolerance.
        stopping_rule : str
            The stopping rule to use for convergence. Options:
            - 'beta': based on coefficient change.
        learning_rate : float
            Learning rate for gradient-based solvers (SGD).
        decay_rate : float
            Learning rate decay factor.
        decay_steps : int
            Steps for learning rate decay, counted by epochs.
        batch_size : int
            Batch size for gradient-based solvers (SGD).
        enable_spu_cache : bool
            Whether to enable SPU cache for secure computation.
        enable_spu_reveal : bool
            Whether to reveal intermediate results in SPU for higher performance.

        Returns
        -------
        beta : jax.Array
            Estimated coefficients.
        dispersion : jax.Array
            Estimated dispersion parameter.
        history : Dict[str, Any]
            Optimization history (e.g., deviance trace).
        """
        ...
