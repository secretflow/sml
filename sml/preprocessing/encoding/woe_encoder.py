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


@partial(jax.jit, static_argnames=("n_groups", "positive_value"))
def fit_woe(
    X: jax.Array, y: jax.Array, n_groups: int, positive_value: int = 1
) -> tuple[jax.Array, jax.Array]:
    """
    WOE encoding with fixed lookup table output.

    Args:
        X (jax.Array): (N, D), integer categories in [0, n_bins).
        y (jax.Array): (N,) or (N, 1), target.
        n_groups (int): Number of group used for each feature.It usually represents n_bins
        positive_value (int): The value in y that represents the positive class.

    Returns:
        tuple[jax.Array, jax.Array]:
            - woe_values: WOE values for each feature and bin with shape (n_features, n_bins)
            - iv_values: Information Values for each feature with shape (n_features,)
    """
    # Validate inputs
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional with shape (n_samples, n_features)")
    if y.ndim not in [1, 2]:
        raise ValueError("y must be 1-dimensional or 2-dimensional")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")

    # Ensure y is the right shape
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Create binary target (1 for positive, 0 for negative)
    y_binary = (y.squeeze() == positive_value).astype(jnp.int32)

    # Count total positive and negative samples
    total_pos = jnp.sum(y_binary)
    total_neg = y_binary.shape[0] - total_pos

    # Handle edge case where all samples are of one class
    epsilon = 1e-10

    # Create one-hot encoding for bin indices
    # X shape: (n_samples, n_features)
    # one_hot shape: (n_samples, n_features, n_bins)
    one_hot = jax.nn.one_hot(X, n_groups)

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


def transform_woe(X: jax.Array, table: jax.Array) -> jax.Array:
    """
    Transform X using WOE lookup table.
    Args:
        X (jax.Array): (N, D), category indices.
        table (jax.Array): (D, n_bins), WOE values from fit_woe.
    Returns:
        jax.Array: (N, D), WOE encoded values.
    """
    feature_indices = jnp.arange(X.shape[1])
    return jax.vmap(lambda x, i: table[i][x], in_axes=(1, 0), out_axes=1)(
        X, feature_indices
    )


class WoEEncoder:
    def __init__(self, n_bins: int, positive_value: int = 1):
        self.n_bins = n_bins
        self.positive_value = positive_value

    def fit(self, X: jax.Array, y: jax.Array):
        woe, iv = fit_woe(X, y, self.n_bins, self.positive_value)
        self.woe_ = woe
        self.iv_ = iv

    def transform(self, X: jax.Array) -> jax.Array:
        return transform_woe(X, self.woe_)

    def fit_transform(self, X: jax.Array, y: jax.Array) -> jax.Array:
        self.fit(X, y)
        return self.transform(X)
