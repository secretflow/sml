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


def invert_matrix(A: jax.Array, eps: float = 1e-9) -> jax.Array:
    """
    Invert a square matrix using naive inversion with jitter for stability.

    This function is used instead of jnp.linalg.solve or cholesky decomposition
    to accommodate specific backend constraints (e.g., MPC).

    Parameters
    ----------
    A : jax.Array
        Square matrix to invert.
    eps : float
        Small value added to the diagonal for numerical stability.

    Returns
    -------
    A_inv : jax.Array
        The inverse of the matrix.
    """
    diag_indices = jnp.diag_indices_from(A)
    # Add jitter to diagonal
    A_stable = A.at[diag_indices].add(eps)
    return jnp.linalg.inv(A_stable)
