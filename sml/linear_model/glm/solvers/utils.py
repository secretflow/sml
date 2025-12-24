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

from sml.utils import sml_reveal


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
        The full coefficient vector, shape (n_features,) or (n_features, 1).
    fit_intercept : bool
        Whether the model was fitted with an intercept.

    Returns
    -------
    coef : jax.Array
        The feature weights, shape (n_features,).
    intercept : jax.Array
        The intercept term (scalar, 0.0 if fit_intercept is False).
    """
    # Squeeze beta to 1D if it's 2D with shape (n, 1)
    beta = jnp.squeeze(beta)
    if fit_intercept:
        return beta[:-1], beta[-1]
    return beta, jnp.array(0.0, dtype=beta.dtype)


def invert_matrix(
    x: jax.Array, iter_round: int = 20, enable_spu_reveal: bool = False
) -> jax.Array:
    """
    computing the inverse of a matrix by newton iteration.
    https://aalexan3.math.ncsu.edu/articles/mat-inv-rep.pdf

    Parameters
    ----------
    x : jax.Array
        The input matrix to be inverted.
    iter_round : int
        The number of iteration rounds.
    enable_spu_reveal : bool
        Whether to reveal intermediate results in SPU for higher performance.

    Returns
    -------
    x_inv : jax.Array
        The inverted matrix.
    """
    assert x.shape[0] == x.shape[1], "x need be a (n x n) matrix"

    if enable_spu_reveal:
        x = sml_reveal(x)  # type: ignore
        return jnp.linalg.inv(x)

    E = jnp.identity(x.shape[0])
    a = (1 / jnp.trace(x)) * E
    for _ in range(iter_round):
        a = jnp.matmul(a, (2 * E - jnp.matmul(x, a)))
    return a


def check_convergence(
    beta_new: jax.Array,
    beta_old: jax.Array,
    stopping_rule: str,
    tol: float,
) -> jax.Array:
    """
    Check convergence based on the specified stopping rule.

    Parameters
    ----------
    beta_new : jax.Array
        New coefficient estimates.
    beta_old : jax.Array
        Previous coefficient estimates.
    stopping_rule : str
        Stopping rule to use. Currently supports "beta".
    tol : float
        Tolerance for convergence.

    Returns
    -------
    converged : jax.Array
        Boolean indicating whether convergence is achieved.
    """
    if stopping_rule == "beta":
        beta_max_delta = jnp.max(jnp.abs(beta_new - beta_old))
        beta_max = jnp.max(jnp.abs(beta_old))
        rel_change = beta_max_delta / beta_max
        return rel_change < tol
    else:
        raise NotImplementedError(f"Stopping rule {stopping_rule} not implemented.")
