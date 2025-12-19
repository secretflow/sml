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


import jax
import jax.numpy as jnp


def add_intercept(X: jax.Array) -> jax.Array:
    """
    Add a constant column of ones to the feature matrix X.

    Parameters
    ----------
    X : jax.Array
        Input feature matrix of shape (n_samples, n_features).

    Returns
    -------
    X_new : jax.Array
        Feature matrix with intercept column appended (n_samples, n_features + 1).
    """
    n_samples = X.shape[0]
    ones = jnp.ones((n_samples, 1), dtype=X.dtype)
    return jnp.concatenate([X, ones], axis=1)


def split_coef(beta: jax.Array, fit_intercept: bool) -> tuple[jax.Array, jax.Array]:
    """
    Split the coefficient vector into weights and intercept.

    Parameters
    ----------
    beta : jax.Array
        The full coefficient vector.
    fit_intercept : bool
        Whether the model was fitted with an intercept.

    Returns
    -------
    coef : jax.Array
        The feature weights.
    intercept : jax.Array
        The intercept term (0.0 if fit_intercept is False).
    """
    if fit_intercept:
        return beta[:-1], beta[-1]
    return beta, jnp.array(0.0, dtype=beta.dtype)


def solve_cholesky(A: jax.Array, b: jax.Array, l2: float = 0.0) -> jax.Array:
    """
    Solve linear system Ax = b using Cholesky decomposition with L2 regularization.
    Solves (A + l2*I)x = b.

    Parameters
    ----------
    A : jax.Array
        Symmetric positive semi-definite matrix (Hessian approximation).
    b : jax.Array
        Right-hand side vector.
    l2 : float
        L2 regularization strength.

    Returns
    -------
    x : jax.Array
        Solution vector.
    """
    if l2 > 0:
        diag_indices = jnp.diag_indices_from(A)
        # We assume A is already X'WX.
        # Regularization is typically not applied to the intercept (last element).
        # However, for simple implementation here, we might apply to all or handle outside.
        # This function assumes 'l2' is already incorporated or A is raw.
        # Let's add l2 to diagonal for stability.
        A = A.at[diag_indices].add(l2)

    # Use jax.scipy.linalg.solve (which usually uses LU or Cholesky internally)
    # For SPD matrices, we can use solve directly or cholesky + solve_triangular.
    return jnp.linalg.solve(A, b)
