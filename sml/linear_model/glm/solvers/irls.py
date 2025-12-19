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

from typing import Any, Dict, Optional, Tuple

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
        offset: Optional[jax.Array] = None,
        sample_weight: Optional[jax.Array] = None,
        l2: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        learning_rate: float = 1e-2,  # Unused in IRLS
        decay_rate: float = 1.0,      # Unused in IRLS
        decay_steps: int = 100,       # Unused in IRLS
        batch_size: int = 128,        # Unused in IRLS
        random_state: Optional[int] = None, # Unused in IRLS (deterministic)
        clip_eta: Optional[Tuple[float, float]] = None,
        clip_mu: Optional[Tuple[float, float]] = None,
    ) -> Tuple[jax.Array, jax.Array, Dict[str, Any]]:
        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        n_samples, n_features = X_train.shape

        # 2. Initialization
        beta_init = jnp.zeros(n_features, dtype=X.dtype)

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
                clip_eta=clip_eta,
                clip_mu=clip_mu,
            )

            # b. Construct Working Response z
            # z = eta + (y - mu) * g'(mu)
            z = eta + z_resid
            
            # If offset is present, we are solving for X @ beta ~ z - offset
            if offset is not None:
                z = z - offset

            # c. Weighted Least Squares: Solve (X'WX + l2*I) beta_new = X'Wz
            # Construct Weighted Matrix Xw = sqrt(W) * X
            w_sqrt = jnp.sqrt(w)[:, None]
            Xw = X_train * w_sqrt
            zw = z * w_sqrt.flatten()

            # Normal Equations Matrix H = Xw.T @ Xw = X^T * W * X
            H = Xw.T @ Xw
            # Score-like vector = X^T * W * z
            score = Xw.T @ zw

            # Apply L2 Regularization to H
            # H_reg = H + l2 * I (excluding intercept)
            if l2 > 0:
                diag_indices = jnp.diag_indices(n_features)
                H = H.at[diag_indices].add(l2)
                # Don't penalize intercept
                if fit_intercept:
                    H = H.at[n_features - 1, n_features - 1].add(-l2)

            # d. Naive Update with Explicit Inversion
            # beta_new = inv(H + eps*I) @ score
            # We add epsilon jitter inside invert_matrix for stability
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
