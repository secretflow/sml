# Copyright 2025 Ant Group Co., Ltd.
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

"""
Optimized IRLS solver for Gaussian (Normal) distribution with Identity link.

Mathematical Derivation:
------------------------
For Gaussian + Identity:
- V(mu) = 1 (variance function is constant)
- mu = eta (inverse link, identity)
- g'(mu) = 1 (link derivative)

Working weights: W = w / (V(mu) * g'(mu)^2) = w / (1 * 1) = w
Working response: z = eta + (y - mu) * g'(mu) = eta + (y - mu) = y

Key Optimization:
-----------------
1. Working weights W are CONSTANT (equal to sample weights).
2. Working response z = y (original response!).
3. X'WX is constant and can be precomputed.
4. This is essentially weighted least squares solved in ONE iteration.
"""

from typing import Any

import jax
import jax.numpy as jnp

from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.solvers.base import Solver
from sml.linear_model.glm.solvers.utils import (
    add_intercept,
    invert_matrix,
)
from sml.utils import sml_drop_cached_var, sml_make_cached_var


class GaussianIdentityIRLSSolver(Solver):
    """
    Optimized IRLS solver for Gaussian distribution with Identity link.

    Key Optimization:
    -----------------
    For Gaussian + Identity, IRLS converges in exactly ONE iteration.
    This is because W is constant and z = y.

    The solution is the standard weighted least squares:
    beta = (X'WX + lambda*phi*I)^{-1} X'Wy
    """

    def solve(
        self,
        X: jax.Array,
        y: jax.Array,
        family: Family,
        fit_intercept: bool = True,
        offset: jax.Array | None = None,
        sample_weight: jax.Array | None = None,
        l2: float = 0.0,
        max_iter: int = 100,  # Ignored - converges in 1 iteration
        tol: float = 1e-4,  # Ignored
        stopping_rule: str = "beta",  # Ignored
        learning_rate: float = 1e-2,  # Unused
        decay_rate: float = 1.0,  # Unused
        decay_steps: int = 1,  # Unused
        batch_size: int = 128,  # Unused
        enable_spu_cache: bool = False,
        enable_spu_reveal: bool = False,
    ) -> tuple[jax.Array, jax.Array | None, dict[str, Any] | None]:

        # 1. Preprocessing - squeeze y/offset/sample_weight if 2D (from model.py)
        if y.ndim > 1:
            y = jnp.squeeze(y)
        if offset is not None and offset.ndim > 1:
            offset = jnp.squeeze(offset)
        if sample_weight is not None and sample_weight.ndim > 1:
            sample_weight = jnp.squeeze(sample_weight)

        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        n_samples, n_features = X_train.shape

        if enable_spu_cache:
            X_train = sml_make_cached_var(X_train)

        # 2. Setup weights
        if sample_weight is not None:
            w = sample_weight
        else:
            w = jnp.ones(n_samples)

        # 3. Handle offset
        if offset is not None:
            y_adjusted = y - offset
        else:
            y_adjusted = y

        # 4. Compute X'WX (constant)
        xtw = jnp.transpose(X_train * w.reshape((-1, 1)))
        xtwx = jnp.matmul(xtw, X_train)

        # 5. Add L2 regularization
        # For Gaussian, phi = sigma^2. We estimate it after fitting.
        # Initially use phi = 1.0
        reg_matrix = l2 * jnp.eye(n_features)
        if fit_intercept:
            # Don't regularize intercept
            reg_matrix = reg_matrix.at[-1, -1].set(0.0)

        lhs = xtwx + reg_matrix

        # 6. Compute X'Wy (z = y for Gaussian+Identity)
        rhs = jnp.matmul(xtw, y_adjusted)

        # 7. Solve the system (ONE iteration!)
        H_inv = invert_matrix(lhs, iter_round=20)
        beta = jnp.matmul(H_inv, rhs)

        # 8. Compute dispersion parameter (sigma^2)
        eta = jnp.matmul(X_train, beta)
        if offset is not None:
            eta = eta + offset
        mu = eta  # Identity link
        residuals = y - mu
        phi = jnp.sum(w * residuals**2) / (n_samples - n_features)

        # 9. Cleanup
        if enable_spu_cache:
            sml_drop_cached_var(X_train)

        history = {
            "n_iter": 1,
            "converged": True,
            "phi": float(phi),
        }

        return beta, phi, history
