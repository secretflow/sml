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
Optimized IRLS solver for Poisson distribution with Log link (canonical).

Mathematical Derivation:
------------------------
For Poisson + Log (canonical link):
- V(mu) = mu (variance function)
- mu = exp(eta) (inverse link)
- g'(mu) = 1/mu (link derivative)

Working weights: W = w / (V(mu) * g'(mu)^2) = w / (mu * (1/mu)^2) = w * mu
Working response: z = eta + (y - mu) * g'(mu) = eta + (y - mu) / mu = eta + y/mu - 1

Key Optimization:
-----------------
The gradient simplifies beautifully for canonical link:
c_i = w_i (SGD auxiliary vector is just sample weights!)
gradient = X'(w * (y - mu))

For IRLS, W = w * mu is computed directly from eta: W = w * exp(eta)
"""

from typing import Any

import jax
import jax.numpy as jnp

from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.solvers.base import Solver
from sml.linear_model.glm.solvers.utils import (
    add_intercept,
    check_convergence,
    invert_matrix,
)
from sml.utils import sml_drop_cached_var, sml_make_cached_var, sml_reveal


def compute_poisson_log_components(
    y: jax.Array,
    eta: jax.Array,
    sample_weight: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute optimized IRLS components for Poisson + Log.

    For Poisson + Log:
    - W = w * mu = w * exp(eta)
    - z = eta + y * exp(-eta) - 1

    Parameters
    ----------
    y : jax.Array
        Response variable.
    eta : jax.Array
        Current linear predictor.
    sample_weight : jax.Array | None
        Optional sample weights.

    Returns
    -------
    w : jax.Array
        Working weights.
    z : jax.Array
        Working response.
    """
    n_samples = y.shape[0]

    # Working weights: W = w * mu = w * exp(eta)
    mu = jnp.exp(eta)
    if sample_weight is not None:
        w = sample_weight * mu
    else:
        w = mu

    # Working response: z = eta + y/mu - 1 = eta + y * exp(-eta) - 1
    z = eta + y * jnp.exp(-eta) - 1.0

    return w, z


class PoissonLogIRLSSolver(Solver):
    """
    Optimized IRLS solver for Poisson distribution with Log link.

    Key Optimization:
    -----------------
    - Working weights W = w * exp(eta) computed directly
    - Working response z = eta + y * exp(-eta) - 1 avoids intermediate mu
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
        max_iter: int = 100,
        tol: float = 1e-4,
        stopping_rule: str = "beta",
        learning_rate: float = 1e-2,  # Unused
        decay_rate: float = 1.0,  # Unused
        decay_steps: int = 1,  # Unused
        batch_size: int = 128,  # Unused
        enable_spu_cache: bool = False,
        enable_spu_reveal: bool = False,
    ) -> tuple[jax.Array, jax.Array | None, dict[str, Any] | None]:

        is_early_stop_enabled = tol > 0.0

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

        # 2. Initialize beta (use log(mean(y)) for intercept)
        beta = jnp.zeros(n_features)
        if fit_intercept:
            y_mean = jnp.mean(y)
            y_mean = jnp.maximum(y_mean, 1e-6)
            beta = beta.at[-1].set(jnp.log(y_mean))

        # 3. Setup base sample weights
        if sample_weight is not None:
            base_w = sample_weight
        else:
            base_w = jnp.ones(n_samples)

        # 4. L2 regularization matrix
        reg_matrix = l2 * jnp.eye(n_features)
        if fit_intercept:
            reg_matrix = reg_matrix.at[-1, -1].set(0.0)

        # 5. IRLS iterations
        history = {"deviance": [], "beta_diff": []}
        converged = False

        for iteration in range(max_iter):
            beta_old = beta

            # Compute linear predictor
            eta = jnp.matmul(X_train, beta)
            if offset is not None:
                eta = eta + offset

            # Clamp eta for numerical stability
            eta = jnp.clip(eta, -20.0, 20.0)

            # Compute working weights and response
            w, z = compute_poisson_log_components(y, eta, base_w)

            # Build IRLS system: (X'WX + lambda*phi*I) beta = X'Wz
            # For Poisson, phi = 1 (theoretically)
            xtw = jnp.transpose(X_train * w.reshape((-1, 1)))
            xtwx = jnp.matmul(xtw, X_train)
            lhs = xtwx + reg_matrix
            rhs = jnp.matmul(xtw, z)

            # Solve
            H_inv = invert_matrix(
                lhs, iter_round=20, enable_spu_reveal=enable_spu_reveal
            )
            beta = jnp.matmul(H_inv, rhs)

            # Check convergence
            if is_early_stop_enabled:
                beta_diff = jnp.max(jnp.abs(beta - beta_old))
                history["beta_diff"].append(float(beta_diff))

                if enable_spu_reveal:
                    beta_diff_revealed = sml_reveal(beta_diff)
                    if check_convergence(beta, beta_old, stopping_rule, tol):
                        converged = True
                        break
                else:
                    if check_convergence(beta, beta_old, stopping_rule, tol):
                        converged = True
                        break

        # 6. Compute final dispersion (should be ~1 for Poisson)
        eta = jnp.matmul(X_train, beta)
        if offset is not None:
            eta = eta + offset
        mu = jnp.exp(jnp.clip(eta, -20.0, 20.0))

        # Pearson chi-squared for dispersion
        pearson_resid = (y - mu) / jnp.sqrt(mu + 1e-10)
        phi = jnp.sum(base_w * pearson_resid**2) / (n_samples - n_features)

        # 7. Cleanup
        if enable_spu_cache:
            sml_drop_cached_var(X_train)

        history["n_iter"] = iteration + 1
        history["converged"] = converged
        history["phi"] = float(phi)

        return beta, phi, history
