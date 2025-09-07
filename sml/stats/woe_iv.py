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


@partial(jax.jit, static_argnames=("n_bins", "positive_value"))
def woe_iv(
    X_binned: jax.Array, y: jax.Array, n_bins: int, positive_value: int = 1
) -> tuple[jax.Array, jax.Array]:
    """
    Calculate Weight of Evidence (WOE) and Information Value (IV) for binned features.

    WOE is a measure of the predictive power of features in relation to a binary target variable.
    It measures the separation between the distribution of positive and negative cases across bins.

    Parameters
    ----------
    X_binned : ArrayLike
        Binned feature values with shape (n_samples, n_features). Each value should be an integer
        in the range [0, n_bins-1] representing the bin assignment for each sample and feature.
    y : ArrayLike
        Target variable with shape (n_samples,). Binary target where positive_value
        indicates the positive class.
    n_bins : int
        Number of bins used for each feature. Must be a positive integer.
    positive_value : int, default=1
        The value in y that represents the positive class. All other values are
        considered negative.

    Returns
    -------
    woe_values : ArrayLike
        WOE values for each feature and bin with shape (n_features, n_bins). Contains the Weight of Evidence
        for each bin of each feature. Bins with no samples will have WOE value of 0.
    iv_values : ArrayLike
        Information Values for each feature with shape (n_features,). Each value represents the overall
        predictive power of the corresponding feature. Higher values indicate stronger predictive power.

    Notes
    -----
    The WOE for each bin is calculated as:
        WOE = ln(% of non-events / % of events)
    where events are the positive class and non-events are the negative class.

    The Information Value (IV) is calculated as:
        IV = Σ (% of non-events - % of events) * WOE

    Edge cases:
    - If a bin has no samples, WOE is set to 0
    - If a bin has only positive or only negative samples, a small epsilon (1e-10)
      is added to avoid division by zero

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sml.stats.woe import woe
    >>> X_binned = jnp.array([[0, 1], [1, 2], [0, 0], [2, 1], [1, 2]])
    >>> y = jnp.array([0, 1, 0, 1, 1])
    >>> woe_values, iv_values = woe(X_binned, y, n_bins=3)
    >>> print(woe_values.shape)  # (2, 3)
    >>> print(iv_values.shape)   # (2,)
    """
    # Validate inputs
    if X_binned.ndim != 2:
        raise ValueError(
            "X_binned must be 2-dimensional with shape (n_samples, n_features)"
        )
    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")
    if X_binned.shape[0] != y.shape[0]:
        raise ValueError("X_binned and y must have the same number of samples")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    # Create binary target (1 for positive, 0 for negative)
    y_binary = (y == positive_value).astype(jnp.int32)

    # Count total positive and negative samples
    total_pos = jnp.sum(y_binary)
    total_neg = y_binary.shape[0] - total_pos

    # Handle edge case where all samples are of one class
    epsilon = 1e-10

    # Create one-hot encoding for bin indices
    # X_binned shape: (n_samples, n_features)
    # one_hot shape: (n_samples, n_features, n_bins)
    one_hot = jax.nn.one_hot(X_binned, n_bins)

    # Calculate positive and negative counts using vectorized operations
    # Broadcast y_binary to match one_hot dimensions for positive counts
    # y_binary shape: (n_samples,) -> (n_samples, 1, 1) -> (n_samples, n_features, n_bins)
    y_binary_expanded = y_binary[:, None, None]
    pos_counts_matrix = one_hot * y_binary_expanded
    neg_counts_matrix = one_hot * (1 - y_binary_expanded)

    # Sum along samples axis to get counts for each feature and bin
    # Result shape: (n_features, n_bins)
    pos_counts = jnp.sum(pos_counts_matrix, axis=0)
    neg_counts = jnp.sum(neg_counts_matrix, axis=0)

    # Calculate percentages
    pos_pct = pos_counts / (total_pos + epsilon)
    neg_pct = neg_counts / (total_neg + epsilon)

    # Calculate WOE for each bin
    woe_values_all = jnp.log((neg_pct + epsilon) / (pos_pct + epsilon))

    # Handle bins with zero samples (WOE = 0)
    total_counts = pos_counts + neg_counts
    woe_values_all = jnp.where(total_counts > 0, woe_values_all, 0.0)

    # Calculate Information Value (IV) for each feature
    iv_values_all = jnp.sum((neg_pct - pos_pct) * woe_values_all, axis=1)

    return woe_values_all, iv_values_all
