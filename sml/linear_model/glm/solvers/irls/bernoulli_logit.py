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
Optimized IRLS solver for Bernoulli distribution with Logit link (canonical).

Mathematical Derivation:
------------------------
For Bernoulli + Logit (canonical link):
- V(mu) = mu * (1 - mu) (variance function)
- mu = sigmoid(eta) = 1 / (1 + exp(-eta)) (inverse link)
- g'(mu) = 1 / (mu * (1 - mu)) (link derivative)

Working weights: W = w / (V(mu) * g'(mu)^2)
               = w / (mu*(1-mu) * (1/(mu*(1-mu)))^2)
               = w * mu * (1 - mu)

Working response: z = eta + (y - mu) * g'(mu)
                = eta + (y - mu) / (mu * (1 - mu))

Key Optimization:
-----------------
Unlike Gamma + Log, the working weights W are NOT constant for Bernoulli.
W = w * mu * (1 - mu) changes with each iteration as mu updates.

However, the computation is optimized by:
1. Using JAX's numerically stable sigmoid function
2. Computing var_mu = mu * (1 - mu) once and reusing it
3. Clamping mu to avoid division by zero at boundaries
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
from sml.utils import (
    sml_drop_cached_var,
    sml_make_cached_var,
    sml_reveal,
    sigmoid_remez_fast,
)


def compute_bernoulli_logit_components(
    y: jax.Array,
    eta: jax.Array,
    sample_weight: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute optimized IRLS components for Bernoulli + Logit.

    For Bernoulli + Logit:
    - W = w * mu * (1 - mu)
    - z = eta + (y - mu) / (mu * (1 - mu))

    Parameters
    ----------
    y : jax.Array
        Response variable (0 or 1).
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
    # Compute mu = sigmoid(eta) using quick remez approximation now.
    mu = sigmoid_remez_fast(eta)

    # Compute variance: var_mu = mu * (1 - mu), cache for reuse
    var_mu = mu * (1.0 - mu)

    # Working weights: W = w * mu * (1 - mu)
    if sample_weight is not None:
        w = var_mu * sample_weight
    else:
        w = var_mu

    # Working response: z = eta + (y - mu) / var_mu
    z = eta + (y - mu) / var_mu

    return w, z


class BernoulliLogitIRLSSolver(Solver):
    """
    Optimized IRLS solver for Bernoulli distribution with Logit link.

    Unlike Gamma + Log, the working weights W are NOT constant for Bernoulli.
    W = w * mu * (1 - mu) changes with each iteration as mu updates.

    However, the computation is optimized by:
    1. Using JAX's numerically stable sigmoid function
    2. Computing var_mu once and reusing it for both W and z
    3. Clamping mu to avoid numerical issues at boundaries
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

        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        _, n_features = X_train.shape

        if enable_spu_cache:
            X_train = sml_make_cached_var(X_train)
            if sample_weight is not None:
                sample_weight = sml_make_cached_var(sample_weight)

        # 2. R-style Initialization
        # mu_init from starting_mu, then compute eta = logit(mu_init)
        mu_init = family.distribution.starting_mu(y)

        # eta = logit(mu) = log(mu / (1 - mu))
        eta = jnp.log(mu_init / (1.0 - mu_init))

        # Directly compute w and z from mu_init (avoid redundant sigmoid)
        # var_mu = mu_init * (1 - mu_init)
        var_mu = mu_init * (1.0 - mu_init)
        if sample_weight is not None:
            w = var_mu * sample_weight
        else:
            w = var_mu
        z = eta + (y - mu_init) / var_mu

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
                w, z = compute_bernoulli_logit_components(y, eta, sample_weight)

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

                w, z = compute_bernoulli_logit_components(y, eta, sample_weight)
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

        # 4. Cleanup
        if enable_spu_cache:
            X_train = sml_drop_cached_var(X_train)
            if sample_weight is not None:
                sample_weight = sml_drop_cached_var(sample_weight)

        history = {"n_iter": n_iter, "converged": converged}

        return beta_final, None, history
