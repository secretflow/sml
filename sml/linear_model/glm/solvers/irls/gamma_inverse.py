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
Optimized IRLS solver for Gamma distribution with Inverse link (canonical).

Mathematical Derivation:
------------------------
For Gamma + Inverse (canonical link):
- V(mu) = mu^2 (variance function)
- eta = 1/mu, so mu = 1/eta (inverse link)
- g'(mu) = -1/mu^2 = -eta^2 (link derivative)

Working weights: W = w / (V(mu) * g'(mu)^2)
               = w / (mu^2 * eta^4)
               = w / ((1/eta^2) * eta^4)
               = w / eta^2 = w * mu^2

Working response: z = eta + (y - mu) * g'(mu)
                = eta - (y - mu) / mu^2
                = eta - (y - mu) * eta^2
                = eta - y*eta^2 + mu*eta^2
                = eta - y*eta^2 + eta  (since mu*eta = 1)
                = 2*eta - y*eta^2

WARNING:
--------
Gamma + Inverse is numerically challenging because:
1. eta must remain positive (eta = 1/mu > 0)
2. Without line search, updates can easily push eta negative
3. The algorithm may diverge without careful initialization

This implementation includes safeguards but may still be unstable.
Consider using Gamma + Log instead when possible.
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


def compute_gamma_inverse_components(
    y: jax.Array,
    eta: jax.Array,
    sample_weight: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute optimized IRLS components for Gamma + Inverse.

    For Gamma + Inverse:
    - W = w * mu^2 = w / eta^2
    - z = 2*eta - y*eta^2

    Parameters
    ----------
    y : jax.Array
        Response variable (must be positive).
    eta : jax.Array
        Current linear predictor (must be positive).
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

    # Ensure eta is positive
    eps = 1e-6
    eta_safe = jnp.maximum(eta, eps)

    # mu = 1/eta
    mu = 1.0 / eta_safe

    # Working weights: W = w * mu^2 = w / eta^2
    if sample_weight is not None:
        w = sample_weight * mu * mu
    else:
        w = mu * mu

    # Working response: z = 2*eta - y*eta^2
    # Alternatively: z = eta - (y - mu) / mu^2 = eta - (y - mu) * eta^2
    z = 2.0 * eta_safe - y * eta_safe * eta_safe

    return w, z


class GammaInverseIRLSSolver(Solver):
    """
    Optimized IRLS solver for Gamma distribution with Inverse link (canonical).

    WARNING:
    --------
    This solver can be numerically unstable. It requires eta > 0 at all times.
    Consider using Gamma + Log link instead for better stability.

    Key Optimization:
    -----------------
    - W = w / eta^2 computed directly
    - z = 2*eta - y*eta^2 avoids intermediate mu computation
    - Uses step size damping to prevent eta from going negative
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

        # 2. Initialize beta
        # For inverse link: eta = 1/mu, so initialize with eta = 1/mean(y)
        beta = jnp.zeros(n_features)
        if fit_intercept:
            y_mean = jnp.mean(y)
            y_mean = jnp.maximum(y_mean, 1e-6)
            # Initial eta = 1/y_mean
            beta = beta.at[-1].set(1.0 / y_mean)

        # 3. Setup base sample weights
        if sample_weight is not None:
            base_w = sample_weight
        else:
            base_w = jnp.ones(n_samples)

        # 4. L2 regularization matrix
        reg_matrix = l2 * jnp.eye(n_features)
        if fit_intercept:
            reg_matrix = reg_matrix.at[-1, -1].set(0.0)

        # 5. IRLS iterations with step size control
        history = {"deviance": [], "beta_diff": []}
        converged = False
        step_size = 1.0  # Damping factor

        for iteration in range(max_iter):
            beta_old = beta

            # Compute linear predictor
            eta = jnp.matmul(X_train, beta)
            if offset is not None:
                eta = eta + offset

            # Ensure eta is positive (critical for inverse link!)
            eta = jnp.maximum(eta, 1e-6)

            # Compute working weights and response
            w, z = compute_gamma_inverse_components(y, eta, base_w)

            # Build IRLS system: (X'WX + lambda*phi*I) beta = X'Wz
            xtw = jnp.transpose(X_train * w.reshape((-1, 1)))
            xtwx = jnp.matmul(xtw, X_train)
            lhs = xtwx + reg_matrix
            rhs = jnp.matmul(xtw, z)

            # Solve
            H_inv = invert_matrix(
                lhs, iter_round=20, enable_spu_reveal=enable_spu_reveal
            )
            beta_new = jnp.matmul(H_inv, rhs)

            # Apply damped update to prevent divergence
            beta = beta_old + step_size * (beta_new - beta_old)

            # Check if eta would go negative with new beta
            eta_new = jnp.matmul(X_train, beta)
            if offset is not None:
                eta_new = eta_new + offset
            min_eta = jnp.min(eta_new)

            # If eta goes negative, reduce step size
            # This is a simple safeguard - more sophisticated line search would be better
            while min_eta < 1e-6 and step_size > 0.1:
                step_size *= 0.5
                beta = beta_old + step_size * (beta_new - beta_old)
                eta_new = jnp.matmul(X_train, beta)
                if offset is not None:
                    eta_new = eta_new + offset
                min_eta = jnp.min(eta_new)

            # Restore step size gradually
            step_size = min(1.0, step_size * 1.1)

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

        # 6. Compute dispersion parameter
        eta = jnp.matmul(X_train, beta)
        if offset is not None:
            eta = eta + offset
        eta = jnp.maximum(eta, 1e-6)
        mu = 1.0 / eta

        # Pearson chi-squared for Gamma dispersion
        pearson_resid = (y - mu) / mu
        phi = jnp.sum(base_w * pearson_resid**2) / (n_samples - n_features)

        # 7. Cleanup
        if enable_spu_cache:
            sml_drop_cached_var(X_train)

        history["n_iter"] = iteration + 1
        history["converged"] = converged
        history["phi"] = float(phi)

        return beta, phi, history
