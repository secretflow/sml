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
Optimized IRLS solver for Tweedie distribution with Log link.

Mathematical Derivation:
------------------------
For Tweedie + Log with power p (1 < p < 2):
- V(mu) = mu^p (variance function)
- mu = exp(eta) (inverse link)
- g'(mu) = 1/mu (link derivative)

Working weights: W = w / (V(mu) * g'(mu)^2)
               = w / (mu^p * (1/mu)^2)
               = w * mu^(2-p)
               = w * exp(eta * (2-p))

Working response: z = eta + (y - mu) * g'(mu)
                = eta + (y - mu) / mu
                = eta + y/mu - 1
                = eta + y * exp(-eta) - 1

Key Optimization:
-----------------
Compute W directly as w * exp(eta * (2-p)) without intermediate mu^p calculation.
This is more numerically stable and avoids potential overflow issues.
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


def compute_tweedie_log_components(
    y: jax.Array,
    eta: jax.Array,
    power: float,
    sample_weight: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute optimized IRLS components for Tweedie + Log.

    For Tweedie + Log with power p:
    - W = w * exp(eta * (2-p))
    - z = eta + y * exp(-eta) - 1

    Parameters
    ----------
    y : jax.Array
        Response variable.
    eta : jax.Array
        Current linear predictor.
    power : float
        Tweedie power parameter (1 < p < 2 for compound Poisson-Gamma).
    sample_weight : jax.Array | None
        Optional sample weights.

    Returns
    -------
    w : jax.Array
        Working weights.
    z : jax.Array
        Working response.
    """
    # Working weights: W = w * exp(eta * (2-p))
    # More stable than computing mu^(2-p) directly
    exp_factor = 2.0 - power
    w = jnp.exp(eta * exp_factor)

    if sample_weight is not None:
        w = w * sample_weight

    # Working response: z = eta + y * exp(-eta) - 1
    z = eta + y * jnp.exp(-eta) - 1.0

    return w, z


class TweedieLogIRLSSolver(Solver):
    """
    Optimized IRLS solver for Tweedie distribution with Log link.

    Unlike Gamma + Log, the working weights W are NOT constant for Tweedie.
    W = w * mu^(2-p) changes with each iteration as mu updates.

    However, the computation is still optimized by:
    1. Computing W directly from eta: W = w * exp(eta * (2-p))
    2. Avoiding intermediate mu^p calculations
    3. Using numerically stable formulations
    """

    def __init__(self, power: float = 1.5):
        """
        Initialize Tweedie solver with power parameter.

        Parameters
        ----------
        power : float
            Tweedie power parameter. Common values:
            - p=0: Normal/Gaussian
            - p=1: Poisson
            - 1<p<2: Compound Poisson-Gamma (most common for insurance)
            - p=2: Gamma
            - p=3: Inverse Gaussian
        """
        self.power = power

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
        power = self.power

        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        _, n_features = X_train.shape

        if enable_spu_cache:
            X_train = sml_make_cached_var(X_train)

        # 2. R-style Initialization
        # For Tweedie, starting_mu = max(y, small_value)
        mu_init = jnp.maximum(y, 1e-4)
        eta = jnp.log(mu_init)
        if offset is not None:
            eta = eta - offset

        # Compute initial components and solve
        w, z = compute_tweedie_log_components(y, eta, power, sample_weight)
        beta = self._solve_wls(
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
                w, z = compute_tweedie_log_components(y, eta, power, sample_weight)

                # Solve WLS
                beta_new = self._solve_wls(
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
                converged = sml_reveal(converged)

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

                w, z = compute_tweedie_log_components(y, eta, power, sample_weight)
                return self._solve_wls(
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

        history = {"n_iter": n_iter, "converged": converged}

        return beta_final, None, history

    def _solve_wls(
        self,
        X: jax.Array,
        z: jax.Array,
        w: jax.Array,
        n_features: int,
        l2: float,
        fit_intercept: bool,
        enable_spu_cache: bool,
        enable_spu_reveal: bool,
    ) -> jax.Array:
        """Solve weighted least squares: beta = (X'WX + l2*I)^{-1} X'Wz"""
        # Compute X'W
        xtw = jnp.transpose(X * w.reshape((-1, 1)))

        # Compute Hessian H = X'WX and score = X'Wz
        H = jnp.matmul(xtw, X)
        score = jnp.matmul(xtw, z)

        # Apply L2 regularization (not on intercept)
        if l2 > 0:
            diag_indices = jnp.diag_indices(n_features)
            H = H.at[diag_indices].add(l2)
            if fit_intercept:
                H = H.at[n_features - 1, n_features - 1].add(-l2)

        # Solve for beta
        H_inv = invert_matrix(H, iter_round=20, enable_spu_reveal=enable_spu_reveal)
        return jnp.matmul(H_inv, score)
