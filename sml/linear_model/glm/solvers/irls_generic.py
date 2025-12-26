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

from typing import Any

import jax
import jax.numpy as jnp

from sml.linear_model.glm.core.family import Family
from sml.utils import sml_make_cached_var, sml_reveal

from .base import Solver
from .utils import add_intercept, check_convergence, solve_wls


def compute_irls_components(
    y: jax.Array,
    mu: jax.Array,
    eta: jax.Array,
    family: Family,
    sample_weight: jax.Array | None = None,
    enable_spu_cache: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute IRLS working weights (W) and working response (z).

    The IRLS algorithm transforms GLM fitting into iterative weighted least squares:
    - Working weights: W = 1 / (V(mu) * (g'(mu))^2)
    - Working response: z = eta + (y - mu) * g'(mu)

    Parameters
    ----------
    y : jax.Array
        Response variable.
    mu : jax.Array
        Current mean estimate.
    eta : jax.Array
        Current linear predictor.
    family : Family
        GLM family containing distribution and link function.
    sample_weight : jax.Array | None
        Optional sample weights.
    enable_spu_cache : bool
        Whether to enable SPU caching for intermediate variables.

    Returns
    -------
    w : jax.Array
        Working weights.
    z : jax.Array
        Working response (adjusted dependent variable).
    """
    # Compute atomic math components
    v_mu = family.distribution.unit_variance(mu)
    g_prime = family.link.link_deriv(mu)

    if enable_spu_cache:
        g_prime = sml_make_cached_var(g_prime)

    # Compute Working Weights W = 1 / (V(mu) * (g'(mu))^2)
    _g2 = g_prime * g_prime
    w = 1.0 / (v_mu * _g2)
    if sample_weight is not None:
        w = w * sample_weight

    # Compute Working Residuals and Working Response
    # z_resid = (y - mu) * g'(mu), z = eta + z_resid
    z_resid = (y - mu) * g_prime
    z = eta + z_resid

    return w, z


class IRLSSolver(Solver):
    """
    Iteratively Reweighted Least Squares (IRLS) solver for GLMs.

    Mathematical Derivation:
    ------------------------
    IRLS maximizes the log-likelihood L(beta) by using Newton-Raphson (Fisher Scoring) updates.

    1. Newton Update Rule:
       beta_{new} = beta_{old} + H^{-1} * Score
       where H = E[-d^2L/dbeta^2] (Fisher Information) and Score = dL/dbeta (Gradient).

    2. Components:
       - Score = X^T * W * z_resid
       - H = X^T * W * X
       where W are working weights and z_resid are working residuals (see Formula).

    3. Transformation to Weighted Least Squares:
       beta_{new} = beta_{old} + (X^T W X)^{-1} X^T W z_resid
                  = (X^T W X)^{-1} [ (X^T W X) beta_{old} + X^T W z_resid ]
                  = (X^T W X)^{-1} X^T W [ X beta_{old} + z_resid ]
       Let z (adjusted response) = eta + z_resid = X beta_{old} + z_resid.
       Then:
       beta_{new} = (X^T W X)^{-1} X^T W z

    This implementation solves this Weighted Least Squares problem iteratively.

    R-style Initialization (iter_num=0):
    ------------------------------------
    Instead of starting with beta=0, we use R-style initialization:
    1. Compute starting mu from y: mu_init = distribution.starting_mu(y)
    2. Compute eta and components from mu_init using formula.compute_components_from_mu()
    3. Perform one IRLS update to get initial beta

    This approach treats initialization as iteration 0, using the same IRLS update
    formula but with mu/eta derived from starting_mu(y) rather than X @ beta.

    NOTE: This implementation uses naive matrix inversion (inv) instead of
    solving linear systems (solve/cholesky) to meet specific backend (SPU) constraints.
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
        learning_rate: float = 1e-2,  # Unused in IRLS
        decay_rate: float = 1.0,  # Unused in IRLS
        decay_steps: int = 1,  # Unused in IRLS
        batch_size: int = 128,  # Unused in IRLS
        enable_spu_cache: bool = False,
        enable_spu_reveal: bool = False,
    ) -> tuple[jax.Array, jax.Array | None, dict[str, Any] | None]:
        # DEBUG: print solver type for verification
        print(f"[DEBUG] Using solver: IRLSSolver (generic) for {family}")

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

        # 2. R-style Initialization (Iteration 0)
        # Use starting_mu(y) to get initial mu, then perform one IRLS update
        mu = family.distribution.starting_mu(y)
        eta = family.link.link(mu)

        # Compute IRLS components and solve WLS for initial beta
        w, z = compute_irls_components(
            y, mu, eta, family, sample_weight, enable_spu_cache
        )

        # subtract offset from z if provided
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

        # 3. Main optimization loop (only if early stopping is enabled)
        if is_early_stop_enabled:
            # State structure: (beta, converged, n_iter)
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

                # Compute mu from eta
                mu = family.link.inverse(eta)

                # Compute IRLS components and solve WLS
                w, z = compute_irls_components(
                    y, mu, eta, family, sample_weight, enable_spu_cache
                )

                if offset is not None:
                    z = z - offset

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

                return beta_new, converged, iter_num + 1

            beta_final, converged, n_iter = jax.lax.while_loop(
                cond_fun, body_fun, init_val
            )
        else:
            # No early stopping: run fixed max_iter iterations using fori_loop
            def fixed_iter_body(_, beta):
                # Compute eta from current beta
                eta = jnp.matmul(X_train, beta)
                if offset is not None:
                    eta = eta + offset

                # Compute mu from eta
                mu = family.link.inverse(eta)

                # Compute IRLS components and solve WLS
                w, z = compute_irls_components(
                    y, mu, eta, family, sample_weight, enable_spu_cache
                )

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

        # 4. Estimate Dispersion (not implemented yet)
        dispersion = None

        # 5. Build history (not implemented the full history yet)
        history = {
            "n_iter": n_iter,
            "converged": converged,
        }

        return beta_final, dispersion, history
