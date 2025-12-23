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

    R-style Initialization:
    -----------------------
    Instead of starting with beta=0, we use R-style initialization:
    1. Compute starting mu from y: mu_init = distribution.starting_mu(y)
    2. Compute starting eta: eta_init = link(mu_init)
    3. Use mu_init and eta_init for the first IRLS iteration

    NOTE: This implementation uses naive matrix inversion (inv) instead of
    solving linear systems (solve/cholesky) to meet specific backend constraints.
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

        # 2. R-style Initialization
        # Instead of beta=0, we use the distribution's starting_mu to initialize.
        # Step 1: Get starting mu from distribution
        mu_init = family.distribution.starting_mu(y)
        # Step 2: Get starting eta from link
        eta_init = family.link.link(mu_init)
        # Step 3: Handle offset
        if offset is not None:
            eta_init = eta_init - offset
        # Step 4: Solve for initial beta via least squares: eta_init = X @ beta_init
        # Using weighted least squares with unit weights for initialization
        # beta_init = (X^T X)^{-1} X^T eta_init
        H_init = X_train.T @ X_train
        if l2 > 0:
            diag_indices = jnp.diag_indices(n_features)
            H_init = H_init.at[diag_indices].add(l2)
            if fit_intercept:
                H_init = H_init.at[n_features - 1, n_features - 1].add(-l2)
        H_init_inv = invert_matrix(H_init, eps=1e-9)
        beta_init = H_init_inv @ (X_train.T @ eta_init)

        # State structure: (beta, converged, n_iter, deviance)
        init_val = (beta_init, False, 0, jnp.inf)

        # 3. Define loop body
        def body_fun(val):
            beta, _, iter_num, _ = val

            # a. Compute components
            # Returns W and z_resid based on current beta
            w, z_resid, mu, eta, dev, _ = formula.compute_components(
                X=X_train,
                y=y,
                beta=beta,
                offset=offset,
                family=family,
                sample_weight=sample_weight,
            )

            # b. Construct Working Response z
            # z = eta + (y - mu) * g'(mu)
            z = eta + z_resid

            # If offset is present, we are solving for X @ beta ~ z - offset
            if offset is not None:
                z = z - offset

            # c. Weighted Least Squares: Solve (X'WX + l2*I) beta_new = X'Wz
            # Construct Weighted Matrix Xw = sqrt(W) * X
            # Note: W here is "Canonical Weight" (W_can), assuming scale factor phi=1.
            # In standard IRLS, phi cancels out in the update rule:
            # beta_new = (X' W_can X)^{-1} X' W_can z
            w_sqrt = jnp.sqrt(w)[:, None]
            Xw = X_train * w_sqrt
            zw = z * w_sqrt.flatten()

            # Normal Equations Matrix H = Xw.T @ Xw = X^T * W * X
            H = Xw.T @ Xw
            # Score-like vector = X^T * W * z
            # Note on L2 Regularization:
            # The Newton update for L2 penalized objective Q(beta) = L(beta) - 0.5 * lambda * ||beta||^2 is:
            # beta_new = beta_old + (H_Likelihood - lambda*I)^{-1} (Grad_Likelihood - lambda*beta_old)
            # Expanding this reveals that the -lambda*beta term cancels with the term from H*beta_old.
            # Resulting in the WLS form: beta_new = (X^T W X + lambda*I)^{-1} X^T W z
            # Thus, we DO NOT subtract lambda*beta from the score vector here.
            score = Xw.T @ zw

            # Apply L2 Regularization to H
            # H_reg = H + l2 * I (excluding intercept)
            # Here 'l2' parameter is treated as the effective regularization strength (lambda * phi).
            if l2 > 0:
                diag_indices = jnp.diag_indices(n_features)
                H = H.at[diag_indices].add(l2)
                # Don't penalize intercept
                if fit_intercept:
                    H = H.at[n_features - 1, n_features - 1].add(-l2)

            # d. Naive Update with Explicit Inversion
            # beta_new = inv(H + eps*I) @ score
            # We add epsilon jitter inside invert_matrix for stability.
            # We use naive inversion (inv) instead of solve() to accommodate specific backend
            # constraints (e.g., MPC/SPU where triangular solves are expensive or unstable).
            H_inv = invert_matrix(H, eps=1e-9)
            beta_new = H_inv @ score

            # e. Convergence check
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
        # dispersion = deviance / (n - p)
        dispersion = dev_final / (n_samples - n_features)

        history = {
            "n_iter": n_iter,
            "converged": converged,
            "final_deviance": dev_final,
        }

        return beta_final, dispersion, history
