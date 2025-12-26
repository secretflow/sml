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

Key Optimization:
-----------------
Unlike Gamma + Log, the working weights W are NOT constant for Gamma + Inverse.
W = w / eta^2 changes with each iteration as eta updates.

However, the computation is optimized by:
1. Computing W directly from eta: W = w / eta^2
2. Working response z = 2*eta - y*eta^2 is simple polynomial in eta
3. Using eta directly without computing mu = 1/eta where possible
"""

from typing import Any

import jax
import jax.numpy as jnp

from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.solvers.base import Solver
from sml.linear_model.glm.solvers.utils import (
    add_intercept,
    check_convergence,
    solve_wls,
)
from sml.utils import sml_make_cached_var, sml_reveal


def compute_gamma_inverse_components(
    y: jax.Array,
    eta: jax.Array,
    sample_weight: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute optimized IRLS components for Gamma + Inverse.

    For Gamma + Inverse:
    - W = w / eta^2 = w * mu^2
    - z = 2*eta - y*eta^2

    Parameters
    ----------
    y : jax.Array
        Response variable (must be positive).
    eta : jax.Array
        Current linear predictor (must be positive for inverse link).
    sample_weight : jax.Array | None
        Optional sample weights.

    Returns
    -------
    w : jax.Array
        Working weights.
    z : jax.Array
        Working response.
    """
    # Working weights: W = w / eta^2
    eta_sq = eta * eta
    w = 1.0 / eta_sq

    if sample_weight is not None:
        w = w * sample_weight

    # Working response: z = 2*eta - y*eta^2
    z = 2.0 * eta - y * eta_sq

    return w, z


class GammaInverseIRLSSolver(Solver):
    """
    Optimized IRLS solver for Gamma distribution with Inverse link (canonical).

    Unlike Gamma + Log, the working weights W are NOT constant for Gamma + Inverse.
    W = w / eta^2 changes with each iteration as eta updates.

    However, the computation is optimized by:
    1. Computing W directly from eta: W = w / eta^2
    2. Working response z = 2*eta - y*eta^2 is a simple polynomial
    3. Using eta directly without computing mu where possible
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
        # DEBUG: print solver type for verification
        print(f"[DEBUG] Using solver: GammaInverseIRLSSolver (optimized)")

        is_early_stop_enabled = tol > 0.0

        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        _, n_features = X_train.shape

        if enable_spu_cache:
            X_train = sml_make_cached_var(X_train)
            y = sml_make_cached_var(y)
            if sample_weight is not None:
                sample_weight = sml_make_cached_var(sample_weight)

        # 2. R-style Initialization
        # mu_init from starting_mu, then eta = 1/mu_init (inverse link)
        mu_init = family.distribution.starting_mu(y)
        eta = 1.0 / mu_init

        # Directly compute w and z from mu_init/eta
        # W = 1/eta^2 = mu_init^2, z = 2*eta - y*eta^2
        eta_sq = eta * eta
        if sample_weight is not None:
            w = mu_init * mu_init * sample_weight
        else:
            w = mu_init * mu_init
        z = 2.0 * eta - y * eta_sq
        if offset is not None:
            z = z - offset

        beta = solve_wls(
            X_train,
            z,
            w,
            n_features,
            l2,
            fit_intercept,
            enable_spu_cache,
            enable_spu_reveal,
        )

        # 3. Main optimization loop
        if is_early_stop_enabled:
            init_val = (beta, False, 1)

            def cond_fun(val):
                _, converged, iter_num = val
                return jnp.logical_and(iter_num < max_iter, jnp.logical_not(converged))

            def body_fun(val):
                beta, _, iter_num = val

                # Compute eta from current beta
                eta = jnp.matmul(X_train, beta)
                if offset is not None:
                    eta = eta + offset

                # Compute optimized components
                w, z = compute_gamma_inverse_components(y, eta, sample_weight)
                if offset is not None:
                    z = z - offset

                # Solve WLS
                beta_new = solve_wls(
                    X_train,
                    z,
                    w,
                    n_features,
                    l2,
                    fit_intercept,
                    enable_spu_cache,
                    enable_spu_reveal,
                )

                # Check convergence
                converged = check_convergence(beta_new, beta, stopping_rule, tol)
                converged = sml_reveal(converged)  # type: ignore

                return (beta_new, converged, iter_num + 1)

            beta_final, converged, n_iter = jax.lax.while_loop(
                cond_fun, body_fun, init_val
            )
        else:
            # Fixed iterations using fori_loop
            def fixed_iter_body(_, beta):
                eta = jnp.matmul(X_train, beta)
                if offset is not None:
                    eta = eta + offset

                w, z = compute_gamma_inverse_components(y, eta, sample_weight)
                if offset is not None:
                    z = z - offset
                return solve_wls(
                    X_train,
                    z,
                    w,
                    n_features,
                    l2,
                    fit_intercept,
                    enable_spu_cache,
                    enable_spu_reveal,
                )

            beta_final = jax.lax.fori_loop(0, max_iter, fixed_iter_body, beta)
            converged, n_iter = False, max_iter

        history = {"n_iter": n_iter, "converged": converged}

        return beta_final, None, history
