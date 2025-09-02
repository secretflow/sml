# Copyright 2025 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from sml.utils.extmath import standardize


@partial(jax.jit, static_argnames=("standardized"))
def pearsonr(X: ArrayLike, standardized: bool = False):
    """
    Compute Pearson correlation coefficient matrix for a given dataset.

    This function calculates the correlation matrix that measures the linear
    relationship between pairs of variables in the input dataset. The diagonal
    elements are set to 1.0, representing perfect self-correlation.

    Parameters
    ----------
    X : ArrayLike
        Input data matrix of shape (n_samples, n_features).
        Each row represents a sample, and each column represents a feature.
    standardized : bool, default=False
        Whether the input data is already standardized (mean=0, std=1).
        If False, the data will be standardized using z-score normalization.

    Returns
    -------
    corr : ArrayLike
        Pearson correlation coefficient matrix of shape (n_features, n_features).
        Each element [i, j] represents the correlation between variable i and variable j.
        Values range from -1 (perfect negative correlation) to 1 (perfect positive correlation).
        Diagonal elements are always 1.0.

    Notes
    -----
    The correlation coefficient is calculated as:
    corr(X, Y) = cov(X, Y) / (std(X) * std(Y))

    This implementation uses the relationship between correlation and covariance:
    corr = X^T * X / (n_samples - 1) for standardized data.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sml.stats import pearsonr
    >>> X = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> corr_matrix = pearsonr(X)
    >>> print(corr_matrix)
    """

    if not standardized:
        X = standardize(X)

    rows = X.shape[0]
    xTx = X.T @ X
    corr = xTx / (rows - 1)

    eye = jnp.eye(corr.shape[0])
    return jnp.where(eye, 1.0, corr)
