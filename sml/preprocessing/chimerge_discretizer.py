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

from sml.utils.utils import sml_reveal


@partial(jax.jit, static_argnames=("n_bins"))
def _quantile_binning(x: jax.Array, n_bins: int) -> jax.Array:
    """
    Perform quantile-based binning on input features.

    This function computes quantile-based bin edges for each feature in the input array.
    It uses linear interpolation to calculate the quantiles.

    Args:
        x: (n_samples, n_features), input features.
        n_bins: scalar, number of bins to create.

    Returns:
        bin_edges: (n_bins + 1, n_features), bin edges for each feature.
    """

    def _qcut(col: jax.Array) -> jax.Array:
        quantiles = jnp.linspace(0, 1, n_bins + 1)
        bins = jnp.quantile(col, quantiles, method="linear")
        return bins

    return jax.vmap(_qcut, in_axes=(1,), out_axes=1)(x)


@partial(jax.jit, static_argnames=("positive_value"))
def _bincounts(
    bin_edges: jax.Array, x: jax.Array, y: jax.Array, positive_value: int = 1
) -> jax.Array:
    """
    Compute positive and negative sample counts for each bin of each feature using broadcasting.

    Args:
        bin_edges: (n_bins + 1, n_features), bin edges for each feature.
        x: (n_samples, n_features), input features.
        y: (n_samples,), labels.
        positive_value: scalar, the value in y that indicates positive class.

    Returns:
        counts: (n_bins, 2, n_features), where
            counts[:, 0, :] = negative counts per feature and bin,
            counts[:, 1, :] = positive counts per feature and bin.
    """
    n_bins = bin_edges.shape[0] - 1  # because bin_edges is (n_bins+1, n_features)

    # Step 1: Use broadcasting to compute bin indices
    # x: (n_samples, n_features)
    # bin_edges.T: (n_features, n_bins+1)
    # Expand x to (n_samples, n_features, 1), bin_edges.T to (1, n_features, n_bins+1)
    # Then compare: x >= bin_edges (broadcasted comparison)
    x_expanded = x[:, :, None]  # (n_samples, n_features, 1)
    bin_edges_T = bin_edges.T[None, :, :]  # (1, n_features, n_bins+1)

    # Boolean comparison: x >= left edge of each bin
    # Result: (n_samples, n_features, n_bins+1)
    # Sum along last axis: gives the number of edges <= x[i,j], so bin index = sum - 1
    #  bin_indices: (n_samples, n_features)
    bin_indices = (x_expanded >= bin_edges_T).sum(axis=-1) - 1

    # Clip to valid bin range [0, n_bins - 1]
    bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)  # (n_samples, n_features)

    # Step 2: Convert y to binary (1 for positive, 0 for negative)
    y_binary = (y == positive_value).astype(jnp.int32)  # (n_samples,)

    # Step 3: One-hot encode bin indices
    # bin_indices: (n_samples, n_features)
    # one_hot: (n_samples, n_features, n_bins)
    one_hot = jax.nn.one_hot(bin_indices, n_bins)  # (n_samples, n_features, n_bins)

    # Step 4: Expand y_binary for broadcasting
    y_binary_expanded = y_binary[:, None, None]  # (n_samples, 1, 1)
    # Compute contributions to positive
    pos_contrib = one_hot * y_binary_expanded  # (n_samples, n_features, n_bins)
    pos_counts = jnp.sum(pos_contrib, axis=0)  # (n_features, n_bins)

    # Total counts per bin
    total_counts = jnp.sum(one_hot, axis=0)  # (n_features, n_bins)
    # Negative counts = total - positive
    neg_counts = total_counts - pos_counts

    # Stack into (2, n_features, n_bins) then transpose to (n_bins, 2, n_features)
    counts = jnp.stack([neg_counts, pos_counts], axis=0)
    counts = counts.transpose(2, 0, 1)

    return counts


def _chi2(freqs: jax.Array) -> float:
    """
    Compute chi-square statistic between two bins (rows) with class frequencies.

    Args:
        freqs: jnp.array of shape (2, n_classes), frequency counts for two adjacent bins.

    Returns:
        chi2: scalar, chi-square statistic.
    """

    def _calc_chi2(freqs, RA, RB, N):
        C = freqs[0] + freqs[1]
        EA = (RA * C) / N
        EB = (RB * C) / N

        term_A = jnp.where(EA > 0, (freqs[0] - EA) ** 2 / jnp.maximum(EA, 1e-10), 0.0)
        term_B = jnp.where(EB > 0, (freqs[1] - EB) ** 2 / jnp.maximum(EB, 1e-10), 0.0)

        return term_A.sum() + term_B.sum()

    RA, RB = freqs[0].sum(), freqs[1].sum()
    N = RA + RB

    return jax.lax.cond(N > 0, lambda: _calc_chi2(freqs, RA, RB, N), lambda: 0.0)


def _chimerge_step(
    bin_edges: jax.Array, bin_counts: jax.Array, cur_n_bins: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Args:
        bin_edges: (n_bins + 1,), bin edges for the feature (padded with inf).
        bin_counts: (n_bins, 2), bin counts where bin_counts[:, 0] are
            negative counts and bin_counts[:, 1] are positive counts (padded with zeros).
        cur_n_bins: scalar, current number of valid bins.
    Returns:
        new_bin_edges: (n_bins + 1,), updated bin edges after merging (padded with inf).
        new_bin_counts: (n_bins, 2), updated bin counts after merging (padded with zeros).
        min_chi: scalar, the minimum chi-square value found.
    """

    # Use the total number of bins (including invalid ones) for calculations
    n_bins = bin_counts.shape[0]

    # Calculate chi-square values for all adjacent pairs
    def calc_chi2(i):
        # Use lax.dynamic_slice to handle dynamic indexing
        return _chi2(jax.lax.dynamic_slice(bin_counts, (i, 0), (2, 2)))

    # Create indices for all adjacent pairs
    indices = jnp.arange(n_bins - 1)
    chi_vec = jax.vmap(calc_chi2)(indices)

    # Additional constraint: only consider indices < cur_n_bins - 1
    # This ensures we only consider valid adjacent pairs within the current number of bins
    valid_adjacent_mask = indices < (cur_n_bins - 1)
    # Set invalid chi-square values to a large number so they won't be selected
    chi_vec = jnp.where(valid_adjacent_mask, chi_vec, jnp.inf)

    # Find the minimum chi-square value and its index
    min_chi = jnp.min(chi_vec)
    min_idx = jnp.argmin(chi_vec)

    # Merge the bins with minimum chi-square
    # Update bin_counts
    freqs = jax.lax.dynamic_slice(bin_counts, (min_idx, 0), (2, 2))
    new_row = freqs[0] + freqs[1]
    new_bin_counts = bin_counts.at[min_idx, :].set(new_row)

    def _shift_padding(x: jax.Array, idx: int, fill_value: jax.Array) -> jax.Array:
        """
        Shift elements in the array backward starting from idx, and fill the last position with fill_value.
        This function works uniformly for arrays of any dimension, always shifting along axis 0 (rows).

        Args:
            x: Input array to shift
            idx: Starting index for shifting
            fill_value: Value to fill the last position

        Returns:
            Shifted array
        """
        n = x.shape[0]
        i = jnp.arange(n)
        src_row = jnp.where(i < idx, i, jnp.minimum(i + 1, n - 1))
        x = x[src_row]
        return x.at[-1].set(fill_value)

    fill_zero = jnp.zeros(2, dtype=bin_counts.dtype)
    new_bin_counts = _shift_padding(new_bin_counts, min_idx + 1, fill_zero)
    new_bin_edges = _shift_padding(bin_edges, min_idx + 1, bin_edges[-1])

    return new_bin_edges, new_bin_counts, min_chi


def _is_all_bins_valid(cur_bin_counts: jax.Array, min_samples: int) -> bool:
    # Check if all bins meet min_samples requirement
    bin_totals = jnp.sum(cur_bin_counts, axis=1)
    return jnp.all(bin_totals >= min_samples)


@partial(jax.jit, static_argnames=("max_n_bins", "threshold", "min_samples"))
def chimerge(
    bin_edges: jax.Array,
    bin_counts: jax.Array,
    n_bins: jax.Array,
    max_n_bins: int,
    threshold: float,
    min_samples: int,
) -> jax.Array:
    """
    Perform ChiMerge discretization on all features.

    Args:
        bin_edges: (init_bins + 1, n_features), bin edges for each feature.
        bin_counts: (init_bins, 2, n_features), bin counts where bin_counts[:, 0] are
            negative counts and bin_counts[:, 1] are positive counts.
        n_bins: (n_features,), target number of bins for each feature.
        max_n_bins: scalar, max(n_bins).
        threshold: scalar, chi-square threshold for merging bins.
            Threshold to determine whether two adjacent bins should be merged.
            Calculated from chi-square distribution with significance level alpha:
            threshold = chi2.ppf(1 - alpha, df=1)
            Common defaults:
            - 2.706 for alpha=0.1 (90% confidence)
            - 3.841 for alpha=0.05 (95% confidence)
            Higher threshold = stricter merging criteria = more bins.
            Lower threshold = more lenient merging criteria = fewer bins.
        min_samples: scalar, minimum number of samples required to perform merging.

    Returns:
        merged_bin_edges: (max(n_bins) + 1, n_features), merged bin edges for each feature.
    """

    def _chimerge_feature(
        feat_bin_edges: jax.Array,
        feat_bin_counts: jax.Array,
        feat_n_bins: int,
        threshold: float,
        min_samples: int,
    ) -> jax.Array:
        """
        Perform ChiMerge discretization on a single feature.

        Args:
            bin_edges: (max_bins + 1,), bin edges for the feature (padded with inf).
            bin_counts: (max_bins, 2), bin counts where bin_counts[:, 0] are
                negative counts and bin_counts[:, 1] are positive counts (padded with zeros).
            n_bins: scalar, target number of bins for this feature.
            threshold: scalar, chi-square threshold for merging bins.
            min_samples: scalar, minimum number of samples required to perform merging.

        Returns:
            merged_bin_edges: (max_bins + 1,), merged bin edges for the feature (padded with inf).
        """

        def _merge_cond(state):
            # need_stop: (cur_n_bins == 1) or (cur_n_bins <= feat_n_bins and hit_threshold and all_bins_valid)
            cur_bin_edges, cur_bin_counts, cur_n_bins, all_bins_valid, hit_threshold = (
                state
            )

            finished = jnp.logical_and(
                cur_n_bins <= feat_n_bins,
                jnp.logical_and(hit_threshold, all_bins_valid),
            )
            need_stop = jnp.logical_or(cur_n_bins == 1, finished)
            need_next = jnp.logical_not(need_stop)
            return sml_reveal(need_next)

        def _merge_body(state):
            cur_bin_edges, cur_bin_counts, cur_n_bins, _, _ = state
            # bin_counts is already in (max_bins, 2) format for _merge_step
            new_bin_edges, new_bin_counts, min_chi = _chimerge_step(
                cur_bin_edges, cur_bin_counts, cur_n_bins
            )
            # new_bin_counts is already in (max_bins, 2) format
            new_n_bins = cur_n_bins - 1

            all_bins_valid = _is_all_bins_valid(cur_bin_counts, min_samples)
            hit_threshold = min_chi >= threshold

            return (
                new_bin_edges,
                new_bin_counts,
                new_n_bins,
                all_bins_valid,
                hit_threshold,
            )

        init_n_bins = feat_bin_counts.shape[0]
        init_all_bins_valid = _is_all_bins_valid(feat_bin_counts, min_samples)
        # Initialize the state
        initial_state = (
            feat_bin_edges,
            feat_bin_counts,
            init_n_bins,
            jnp.array(init_all_bins_valid),
            jnp.array(False),
        )

        # Run the merging loop
        final_bin_edges, _, _, _, _ = jax.lax.while_loop(
            _merge_cond, _merge_body, initial_state
        )

        return final_bin_edges

    # Use vmap to apply _chimerge_feature to each feature
    # in_axes=(1, 2, 0, None, None) means:
    # - bin_edges: apply vmap to axis 1 (features)
    # - bin_counts: apply vmap to axis 2 (features)
    # - n_bins: apply vmap to axis 0 (features)
    # - threshold: same for all features (None means no vmap)
    # - min_samples: same for all features (None means no vmap)
    merged_bin_edges = jax.vmap(_chimerge_feature, in_axes=(1, 2, 0, None, None))(
        bin_edges, bin_counts, n_bins, threshold, min_samples
    )

    # Transpose to get the correct shape (max_bins + 1, n_features)
    merged_bin_edges = merged_bin_edges.T

    # Clip the output to (max_n_bins+1, n_features)
    clipped_edges = merged_bin_edges[: max_n_bins + 1, :]
    return clipped_edges


class ChiMergeDiscretizer:
    """
    ChiMerge discretizer for supervised discretization of continuous features.

    ChiMerge is a discretization algorithm that uses chi-square statistics to determine
    the optimal binning of continuous features based on their relationship with a target
    variable. It iteratively merges adjacent intervals until the chi-square statistic
    between adjacent intervals exceeds a specified threshold.

    The algorithm works by:
    1. Initializing bins using quantile-based discretization
    2. Computing chi-square statistics between adjacent bins
    3. Merging adjacent bins with the smallest chi-square values
    4. Repeating until the desired number of bins is reached or the chi-square threshold is met

    Parameters
    ----------
    n_bins : int or jax.Array
        The number of bins to produce. Can be:
        - An integer: same number of bins for all features
        - A jax.Array: different number of bins for each feature
    init_bins : int, default=100
        The initial number of bins to create using quantile-based discretization
        before applying ChiMerge. Higher values provide more granular initial bins.
    chi_threshold : float, default=2.706
        The chi-square threshold for merging bins. This corresponds to the critical
        value from the chi-square distribution with 1 degree of freedom.
        Common values:
        - 2.706 for 90% confidence level (alpha=0.1)
        - 3.841 for 95% confidence level (alpha=0.05)
        - 6.635 for 99% confidence level (alpha=0.01)
        Higher thresholds result in more bins (less merging).
    min_samples : int, default=50
        Minimum number of samples required in each bin. Bins with fewer samples
        will not be considered for merging.
    positive_value : int, default=1
        The value in the target variable that indicates the positive class.

    Attributes
    ----------
    n_bins_ : jax.Array
        The actual number of bins for each feature after fitting.
    bin_edges_ : jax.Array
        The bin edges for each feature, shape (max_bins+1, n_features).
    n_features_ : int
        The number of features in the training data.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sml.preprocessing import ChiMergeDiscretizer
    >>> X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = jnp.array([0, 1, 0, 1])
    >>> discretizer = ChiMergeDiscretizer(n_bins=2)
    >>> X_discrete = discretizer.fit_transform(X, y)
    >>> print(X_discrete.shape)
    (4, 2)

    Notes
    -----
    This implementation uses JAX for efficient computation and supports both
    CPU and GPU acceleration. The algorithm is particularly useful for:
    - Feature engineering in machine learning pipelines
    - Creating monotonic relationships between features and target
    - Handling non-linear relationships in logistic regression
    - Reducing the impact of outliers through binning

    References
    ----------
    Kerber, R. (1992). ChiMerge: Discretization of numeric attributes.
    In Proceedings of the tenth national conference on Artificial intelligence
    (pp. 123-128).
    """

    def __init__(
        self,
        n_bins: int | jax.Array,
        init_bins: int = 100,
        chi_threshold: float = 2.706,
        min_samples: int = 50,
        positive_value: int = 1,
    ):
        self.n_bins = n_bins
        self.init_bins = init_bins
        self.chi_threshold = chi_threshold
        self.min_samples = min_samples
        self.positive_value = positive_value

    def fit(
        self,
        X: jax.Array,
        y: jax.Array,
    ) -> "ChiMergeDiscretizer":
        """
        Fit the ChiMerge discretizer to the training data.

        This method learns the optimal bin boundaries for each feature based on
        the chi-square statistic between the feature values and the target variable.
        The algorithm starts with quantile-based initial bins and iteratively merges
        adjacent bins until the chi-square threshold is met or the desired number
        of bins is reached.

        Parameters
        ----------
        X : jax.Array
            Training data of shape (n_samples, n_features) containing continuous
            features to be discretized.
        y : jax.Array
            Target values of shape (n_samples,) used for supervised discretization.
            Should contain binary classification labels (0 and 1, or other values
            specified by positive_value).

        Returns
        -------
        self : ChiMergeDiscretizer
            Returns the instance itself with fitted attributes.

        Raises
        ------
        ValueError
            If the input arrays have incompatible shapes or if min_samples is
            larger than the number of samples in any bin.

        Notes
        -----
        The fitting process involves:
        1. Initial binning using quantiles to create initial_bins
        2. Computing class frequencies for each bin
        3. Iteratively merging adjacent bins based on chi-square statistics
        4. Stopping when the chi-square threshold is reached or target bins achieved

        The fitted bin edges are stored in self.bin_edges_ and can be used for
        transforming new data.
        """
        init_bin_edges = _quantile_binning(X, n_bins=self.init_bins)
        bin_counts = _bincounts(
            init_bin_edges, X, y, positive_value=self.positive_value
        )

        if isinstance(self.n_bins, int):
            max_n_bins = self.n_bins
            n_bins = jnp.full(X.shape[1], self.n_bins)
        else:
            max_n_bins = jnp.max(self.n_bins)
            n_bins = jnp.array(self.n_bins)
            if n_bins.shape[0] != X.shape[1]:
                raise ValueError(
                    f"The length of n_bins ({n_bins.shape[0]}) does not match the number of features in X ({X.shape[1]})."
                )

        bin_edges = chimerge(
            init_bin_edges,
            bin_counts,
            n_bins,
            max_n_bins,
            self.chi_threshold,
            self.min_samples,
        )

        self.n_bins_ = n_bins
        self.bin_edges_ = bin_edges
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: jax.Array) -> jax.Array:
        """Transform data using fitted bin edges.

        Parameters
        ----------
        X : jax.Array
            Input data of shape (n_samples, n_features)

        Returns
        -------
        X_bin : jax.Array
            Discretized data of shape (n_samples, n_features)
        """
        # X shape: (n_samples, n_features)
        # self.bin_edges_ shape: (max_n_bins+1, n_features)

        # because bin_edges is (n_bins+1, n_features)
        n_bins = self.bin_edges_.shape[0] - 1

        # Compute bin indices for all features at once using broadcasting
        # X shape: (n_samples, n_features)
        # self.bin_edges_ shape: (n_bins+1, n_features)
        # After broadcasting: (n_samples, n_features, n_bins+1)
        bin_indices = (X[..., None] >= self.bin_edges_.T[None, ...]).sum(axis=-1) - 1

        # Clip indices to ensure they are within valid range [0, n_bins-1]
        # Values < 0 go to bin 0, values >= n_bins go to bin n_bins-1
        bin_indices = jnp.clip(bin_indices, 0, n_bins - 1)

        return bin_indices

    def fit_transform(self, X: jax.Array, y: jax.Array) -> jax.Array:
        """
        Fit the discretizer to the training data and transform it in one step.

        This is a convenience method that combines fit() and transform() into a single
        operation. It first learns the optimal bin boundaries from the training data
        and then immediately applies the discretization to the same data.

        Parameters
        ----------
        X : jax.Array
            Training data of shape (n_samples, n_features) containing continuous
            features to be discretized.
        y : jax.Array
            Target values of shape (n_samples,) used for supervised discretization.

        Returns
        -------
        X_discrete : jax.Array
            Discretized training data of shape (n_samples, n_features) containing
            bin indices for each feature.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from sml.preprocessing import ChiMergeDiscretizer
        >>> X = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = jnp.array([0, 1, 0, 1])
        >>> discretizer = ChiMergeDiscretizer(n_bins=2)
        >>> X_discrete = discretizer.fit_transform(X, y)
        >>> print(X_discrete)
        [[0 0]
         [1 1]
         [1 1]
         [1 1]]

        Notes
        -----
        This method is equivalent to calling fit(X, y) followed by transform(X),
        but is more efficient as it avoids storing intermediate results.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: jax.Array) -> jax.Array:
        """Inverse transform discretized data back to original numerical values.

        This method converts bin indices back to numerical values by using
        the midpoint of each bin as the representative value.

        Parameters
        ----------
        X : jax.Array
            Discretized data of shape (n_samples, n_features) containing bin indices

        Returns
        -------
        X_original : jax.Array
            Reconstructed numerical values of shape (n_samples, n_features)
        """
        # X shape: (n_samples, n_features) containing bin indices
        # self.bin_edges_ shape: (n_bins+1, n_features)

        # Calculate bin midpoints for all features
        # For each bin, midpoint = (left_edge + right_edge) / 2
        # self.bin_edges_[:-1, :] gives left edges (excluding the last edge)
        # self.bin_edges_[1:, :] gives right edges (excluding the first edge)
        left_edges = self.bin_edges_[:-1, :]  # (n_bins, n_features)
        right_edges = self.bin_edges_[1:, :]  # (n_bins, n_features)
        bin_midpoints = (left_edges + right_edges) / 2  # (n_bins, n_features)

        # Clip input bin indices to valid range [0, n_bins-1]
        n_bins = self.bin_edges_.shape[0] - 1
        clipped_indices = jnp.clip(X, 0, n_bins - 1)

        # Use advanced indexing to select the appropriate midpoints
        # We need to create indices for each sample and feature
        n_samples, n_features = X.shape

        # Create indices for features (0 to n_features-1 for each sample)
        feature_indices = jnp.tile(jnp.arange(n_features), (n_samples, 1))

        # Use clipped_indices and feature_indices to select the correct midpoints
        # bin_midpoints shape: (n_bins, n_features)
        # We want bin_midpoints[clipped_indices[i, j], j] for each i, j
        reconstructed = bin_midpoints[clipped_indices, feature_indices]

        return reconstructed
