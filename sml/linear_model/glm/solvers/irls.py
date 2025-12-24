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

from sml.utils import sml_make_cached_var
from sml.linear_model.glm.core.family import Family
from sml.linear_model.glm.formula.base import Formula

from .base import Solver
from .utils import add_intercept, invert_matrix


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
        formula: Formula,
        fit_intercept: bool = True,
        offset: jax.Array | None = None,
        sample_weight: jax.Array | None = None,
        l2: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        learning_rate: float = 1e-2,  # Unused in IRLS
        decay_rate: float = 1.0,  # Unused in IRLS
        decay_steps: int = 100,  # Unused in IRLS
        batch_size: int = 128,  # Unused in IRLS
    ) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        n_samples, n_features = X_train.shape

        # 2. R-style Initialization (Iteration 0)
        # Use starting_mu(y) to get initial mu, then perform one IRLS update
        mu_init = family.distribution.starting_mu(y)

        # Compute components from mu (not from beta)
        w_init, z_resid_init, eta_init, dev_init, _ = (
            formula.compute_components_from_mu(
                y=y, mu=mu_init, family=family, sample_weight=sample_weight
            )
        )

        # Construct working response z = eta + z_resid
        z_init = eta_init + z_resid_init

        # Handle offset
        if offset is not None:
            z_init = z_init - offset

        # Perform WLS to get initial beta: beta = (X'WX + l2*I)^{-1} X'Wz
        w_sqrt_init = jnp.sqrt(w_init)[:, None]
        Xw_init = X_train * w_sqrt_init
        zw_init = z_init * w_sqrt_init.flatten()

        H_init = Xw_init.T @ Xw_init
        score_init = Xw_init.T @ zw_init

        if l2 > 0:
            diag_indices = jnp.diag_indices(n_features)
            H_init = H_init.at[diag_indices].add(l2)
            if fit_intercept:
                H_init = H_init.at[n_features - 1, n_features - 1].add(-l2)

        H_init_inv = invert_matrix(H_init, eps=1e-9)
        beta_init = H_init_inv @ score_init

        # State structure: (beta, converged, n_iter, deviance)
        # Start from iter_num=1 since we already did iteration 0
        init_val = (beta_init, False, 1, dev_init)

        # 3. Define loop body (iterations 1, 2, ...)
        def body_fun(val):
            beta, _, iter_num, _ = val

            # a. Compute components from current beta
            w, z_resid, mu, eta, dev, _ = formula.compute_components(
                X=X_train,
                y=y,
                beta=beta,
                offset=offset,
                family=family,
                sample_weight=sample_weight,
            )

            # b. Construct Working Response z
            z = eta + z_resid

            # Handle offset
            if offset is not None:
                z = z - offset

            # c. Weighted Least Squares: Solve (X'WX + l2*I) beta_new = X'Wz
            w_sqrt = jnp.sqrt(w)[:, None]
            Xw = X_train * w_sqrt
            zw = z * w_sqrt.flatten()

            H = Xw.T @ Xw
            score = Xw.T @ zw

            if l2 > 0:
                diag_indices = jnp.diag_indices(n_features)
                H = H.at[diag_indices].add(l2)
                if fit_intercept:
                    H = H.at[n_features - 1, n_features - 1].add(-l2)

            H_inv = invert_matrix(H, eps=1e-9)
            beta_new = H_inv @ score

            # d. Convergence check
            beta_diff = jnp.linalg.norm(beta_new - beta)
            beta_norm = jnp.linalg.norm(beta)
            rel_change = beta_diff / (beta_norm + 1e-12)
            converged = rel_change < tol

            return beta_new, converged, iter_num + 1, dev

        # 4. Loop condition
        def cond_fun(val):
            _, converged, iter_num, _ = val
            return jnp.logical_and(iter_num < max_iter, jnp.logical_not(converged))

        # 5. Run optimization
        beta_final, converged, n_iter, dev_final = jax.lax.while_loop(
            cond_fun, body_fun, init_val
        )

        # 6. Estimate Dispersion
        dispersion = dev_final / (n_samples - n_features)

        history = {
            "n_iter": n_iter,
            "converged": converged,
            "final_deviance": dev_final,
        }

        return beta_final, dispersion, history
