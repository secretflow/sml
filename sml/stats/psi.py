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

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def _compute_distribution(data: ArrayLike, bin_edges: ArrayLike, n_bins: int):
    """Compute the sample distribution proportion for each bin using pure JAX and vmap"""

    def _count_feature(feature_data, feature_bins):
        """Count samples in each bin for a single feature"""
        # Find bin indices using searchsorted
        bin_indices = jnp.searchsorted(feature_bins, feature_data, side="right") - 1

        # Clip indices to ensure they are within valid range [0, n_bins-1]
        # Values < 0 go to bin 0, values >= n_bins go to bin n_bins-1
        bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

        # Count samples in each bin using bincount
        counts = jnp.bincount(bin_indices, length=n_bins)

        return counts

    # Use vmap to apply the counting function to all features
    all_features = jax.vmap(_count_feature, in_axes=(1, 1), out_axes=1)(data, bin_edges)

    # Compute distribution proportions
    n_samples = data.shape[0]
    bin_dist = all_features / n_samples

    return bin_dist


def psi(
    actual_data: ArrayLike,
    expect_data: ArrayLike,
    bins: ArrayLike,
    eps: float = 1e-8,
) -> ArrayLike:
    """
    Compute Population Stability Index (PSI) between actual and expected data.

    PSI measures the stability of a population over time by comparing the distribution
    of a variable in two different time periods. A low PSI indicates that the population
    distribution has not changed significantly, while a high PSI indicates a significant
    shift in the population distribution.

    Parameters
    ----------
    actual_data : ArrayLike
        Actual data samples with shape (n_samples, n_features)
    expect_data : ArrayLike
        Expected data samples with shape (n_samples, n_features)
    bins : ArrayLike
        Bin edges with shape (n_bins+1, n_features)
    eps : float, default=1e-8
        Small epsilon value to avoid division by zero

    Returns
    -------
    psi_values : ArrayLike
        PSI values for each feature with shape (n_features,)

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.2: Slight change
        - 0.2 ≤ PSI < 0.5: Moderate change
        - PSI ≥ 0.5: Significant change

    Notes
    -----
    The Population Stability Index (PSI) is calculated as:
    PSI = Σ (actual_proportion - expected_proportion) * ln(actual_proportion / expected_proportion)

    This metric is widely used in credit scoring and risk modeling to monitor
    population drift over time. It compares the distribution of a variable
    between a development sample (expected) and a current sample (actual).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sml.stats import psi
    >>> # Create sample data
    >>> actual_data = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    >>> expect_data = jnp.array([[1.5, 2.5], [2.5, 3.5]])
    >>> bins = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    >>> psi_values = psi(actual_data, expect_data, bins)
    >>> print(psi_values)
    """
    # Ensure 2D arrays
    actual_data = jnp.atleast_2d(actual_data)
    expect_data = jnp.atleast_2d(expect_data)
    bins = jnp.atleast_2d(bins)

    # Validate input dimension consistency
    if (
        actual_data.shape[1] != expect_data.shape[1]
        or actual_data.shape[1] != bins.shape[1]
    ):
        raise ValueError(
            f"actual_data and expect_data and bins must have same number of features, "
            f"got {actual_data.shape} and {expect_data.shape} and {bins.shape}"
        )

    n_bins = bins.shape[0] - 1

    # Compute bin distributions for actual and expected data
    actual_dist = _compute_distribution(actual_data, bins, n_bins)
    expect_dist = _compute_distribution(expect_data, bins, n_bins)

    # Avoid division by zero, replace zero values with eps
    actual_dist = jnp.where(actual_dist == 0, eps, actual_dist)
    expect_dist = jnp.where(expect_dist == 0, eps, expect_dist)

    # Compute PSI values
    # PSI = Σ (actual_proportion - expected_proportion) * jnp.log(actual_dist / expect_dist)
    psi_values = jnp.sum(
        (actual_dist - expect_dist) * jnp.log(actual_dist / expect_dist), axis=0
    )

    return psi_values
