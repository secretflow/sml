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
from sml.linear_model.glm.formula.base import Formula


class Solver(Protocol):
    """Protocol for GLM solvers (IRLS, Newton-CG, etc.)."""

    def solve(
        self,
        X: jax.Array,
        y: jax.Array,
        family: Family,
        formula: Formula,
        fit_intercept: bool = True,
        offset: jax.Array | None = None,
        sample_weight: jax.Array | None = None,
        l2: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        clip_eta: tuple[float, float] | None = None,
        clip_mu: tuple[float, float] | None = None,
    ) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
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
            Maximum number of iterations.
        tol : float
            Convergence tolerance.
        clip_eta : Tuple[float, float], optional
            Bounds for eta clipping.
        clip_mu : Tuple[float, float], optional
            Bounds for mu clipping.

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
