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
from .utils import add_intercept


class IRLSSolver(Solver):
    """
    Iteratively Reweighted Least Squares (IRLS) solver for GLMs.

    IRLS is equivalent to Fisher Scoring and is the standard algorithm for GLMs.
    It solves the maximization of the likelihood by iteratively solving weighted
    least squares problems.
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
        clip_eta: tuple[float, float] | None = None,
        clip_mu: tuple[float, float] | None = None,
    ) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
        # 1. Preprocessing
        if fit_intercept:
            X_train = add_intercept(X)
        else:
            X_train = X

        n_samples, n_features = X_train.shape

        # 2. Initialization
        # Initialize beta with zeros or via starting_mu
        # Strategy: Get starting mu -> link(mu) -> Solve linear system for rough beta
        # For simplicity in JIT, we often start with zero or a simple guess.
        # Here we try a simple initialization: beta = 0
        beta_init = jnp.zeros(n_features, dtype=X.dtype)

        # Better initialization:
        # mu_init = family.distribution.starting_mu(y)
        # eta_init = family.link.link(mu_init)
        # However, solving X @ beta = eta_init requires an initial solve.
        # Let's stick to zeros for robustness in pure JAX static graph,
        # or we could do one step of OLS on projected eta.

        # State structure for loop: (beta, converged, n_iter, deviance)
        init_val = (beta_init, False, 0, jnp.inf)

        # 3. Define the loop body
        def body_fun(val):
            beta, _, iter_num, _ = val

            # a. Compute components via Formula
            # We assume formula handles numerical stability inside
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
            # z = eta + z_resid
            z = eta + z_resid

            # c. Weighted Least Squares: Solve (X'WX + l2*I) beta_new = X'Wz
            # Construct Weighted Matrix Xw = sqrt(W) * X
            # Note: W is (N,), sqrt(W) is (N,). Broadcast multiply.
            w_sqrt = jnp.sqrt(w)[:, None]
            Xw = X_train * w_sqrt
            zw = z * w_sqrt.flatten()

            # Normal Equations: (Xw.T @ Xw) beta = Xw.T @ zw
            XtX = Xw.T @ Xw
            Xtz = Xw.T @ zw

            # Apply L2 Regularization
            if l2 > 0:
                # Add l2 to diagonal
                diag_indices = jnp.diag_indices(n_features)
                XtX = XtX.at[diag_indices].add(l2)
                # Don't penalize intercept (last element if fit_intercept=True)
                if fit_intercept:
                    XtX = XtX.at[n_features - 1, n_features - 1].add(-l2)

            # Solve system
            # Use jnp.linalg.solve (assumes XtX is non-singular/SPD)
            # Add small jitter for numerical stability if needed
            XtX = XtX.at[jnp.diag_indices_from(XtX)].add(1e-9)
            beta_new = jnp.linalg.solve(XtX, Xtz)

            # d. Check convergence
            # Relative change in coefficients
            beta_diff = jnp.linalg.norm(beta_new - beta)
            beta_norm = jnp.linalg.norm(beta)
            rel_change = beta_diff / (beta_norm + 1e-12)
            converged = rel_change < tol

            return beta_new, converged, iter_num + 1, dev

        # 4. Define loop condition
        def cond_fun(val):
            _, converged, iter_num, _ = val
            return jnp.logical_and(iter_num < max_iter, jnp.logical_not(converged))

        # 5. Run optimization
        beta_final, converged, n_iter, dev_final = jax.lax.while_loop(
            cond_fun, body_fun, init_val
        )

        # 6. Estimate Dispersion (optional, simplified)
        # dispersion = deviance / (n - p)
        # For now, return 1.0 or implement specific logic
        dispersion = dev_final / (n_samples - n_features)

        history = {
            "n_iter": n_iter,
            "converged": converged,
            "final_deviance": dev_final,
        }

        return beta_final, dispersion, history
