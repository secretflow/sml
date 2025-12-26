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
Optimized IRLS solver for Gamma distribution with Log link.

Mathematical Derivation:
------------------------
For Gamma + Log:
- V(mu) = mu^2 (variance function)
- mu = exp(eta) (inverse link)
- g'(mu) = 1/mu (link derivative)

Working weights: W = w / (V(mu) * g'(mu)^2) = w / (mu^2 * (1/mu)^2) = w
Working response: z = eta + (y - mu) * g'(mu) = eta + (y - mu) / mu = eta + y/mu - 1

Key Optimization:
-----------------
The working weights W are CONSTANT (equal to sample weights or 1).
This means X'WX is constant across iterations and can be precomputed!
Only X'Wz needs to be recomputed each iteration.
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
from sml.utils import sml_make_cached_var, sml_reveal


class GammaLogIRLSSolver(Solver):
    """
    Optimized IRLS solver for Gamma distribution with Log link.

    Key Optimization:
    -----------------
    Since W is constant (sample weights only), X'WX can be precomputed once
    and reused across all iterations. Only the RHS (X'Wz) changes.

    This reduces computation from O(n*p^2 + p^3) to O(n*p + p^2) per iteration
    after the first iteration.
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
        print(f"[DEBUG] Using solver: GammaLogIRLSSolver (optimized)")

        is_early_stop_enabled = tol > 0.0

        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        n_samples, n_features = X_train.shape

        if enable_spu_cache:
            X_train = sml_make_cached_var(X_train)
            y = sml_make_cached_var(y)
            if sample_weight is not None:
                sample_weight = sml_make_cached_var(sample_weight)

        # 2. Precompute constant working weights
        if sample_weight is not None:
            w = sample_weight
        else:
            w = jnp.ones(n_samples)

        # 3. Precompute X'WX (constant for Gamma + Log!)
        # This is the key optimization
        xtw = jnp.transpose(X_train * w.reshape((-1, 1)))
        if enable_spu_cache:
            xtw = sml_make_cached_var(xtw)

        H = jnp.matmul(xtw, X_train)

        # Apply L2 regularization (not on intercept)
        if l2 > 0:
            diag_indices = jnp.diag_indices(n_features)
            H = H.at[diag_indices].add(l2)
            if fit_intercept:
                H = H.at[n_features - 1, n_features - 1].add(-l2)

        # Precompute H_inv (constant!)
        H_inv = invert_matrix(H, iter_round=20, enable_spu_reveal=enable_spu_reveal)
        if enable_spu_cache:
            H_inv = sml_make_cached_var(H_inv)

        # 4. R-style Initialization
        mu_init = family.distribution.starting_mu(y)
        eta = jnp.log(mu_init)

        # Compute initial z and solve
        z = eta + y * jnp.exp(-eta) - 1.0
        if offset is not None:
            z = z - offset
        score = jnp.matmul(xtw, z)
        beta = jnp.matmul(H_inv, score)

        # 5. Main optimization loop
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

                # Compute optimized z (W is constant, already in xtw)
                z = eta + y * jnp.exp(-eta) - 1.0

                # Solve: beta_new = H_inv @ X'Wz
                if offset is not None:
                    z = z - offset
                score = jnp.matmul(xtw, z)
                beta_new = jnp.matmul(H_inv, score)

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

                z = eta + y * jnp.exp(-eta) - 1.0
                if offset is not None:
                    z = z - offset
                score = jnp.matmul(xtw, z)
                return jnp.matmul(H_inv, score)

            beta_final = jax.lax.fori_loop(0, max_iter, fixed_iter_body, beta)
            converged, n_iter = False, max_iter

        history = {"n_iter": n_iter, "converged": converged}

        return beta_final, None, history
